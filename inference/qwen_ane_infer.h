// qwen_ane_infer.h — Qwen2.5-0.5B inference on Apple Neural Engine
// Linear projections on ANE (baked-weight conv kernels), CPU for element-wise ops.
// Based on maderix/ANE runtime + MIL generation.
#pragma once

#include "../training/ane_runtime.h"
#include "../training/ane_mil_gen.h"

// Compile a matmul kernel: W[out_ch, in_ch] @ x[in_ch] → y[out_ch]
// Uses the two-input matmul MIL variant (weights passed as input, not baked)
static ANEKernel *compile_matmul_kernel(int in_ch, int out_ch) {
    NSString *mil = mil_gen_matmul(in_ch, out_ch, 1);
    size_t inputSizes[2] = {(size_t)in_ch * 1 * 4, (size_t)out_ch * in_ch * 4};
    size_t outBytes = (size_t)out_ch * 1 * 4;
    return ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, inputSizes, 1, &outBytes);
}

// Compile a baked-weight conv kernel (from model.h)
static ANEKernel *compile_conv_kernel(const float *weights, int in_ch, int out_ch, int spatial) {
    NSData *wb = mil_build_weight_blob(weights, out_ch, in_ch);
    NSString *mil = mil_gen_conv(in_ch, out_ch, spatial);
    size_t inBytes = (size_t)in_ch * spatial * 4;
    size_t outBytes = (size_t)out_ch * spatial * 4;
    return ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], wb, 1, &inBytes, 1, &outBytes);
}

// Compile baked-weight conv with FP16 IOSurfaces (for fused ANE path)
static ANEKernel *compile_conv_kernel_fp16io(const float *weights, int in_ch, int out_ch, int spatial) {
    int saved = g_fp16_io; g_fp16_io = 1;
    NSData *wb = mil_build_weight_blob(weights, out_ch, in_ch);
    NSString *mil = mil_gen_conv(in_ch, out_ch, spatial);
    size_t inBytes = (size_t)in_ch * spatial * sizeof(_Float16);
    size_t outBytes = (size_t)out_ch * spatial * sizeof(_Float16);
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], wb, 1, &inBytes, 1, &outBytes);
    g_fp16_io = saved;
    return k;
}
#include <math.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>

static void *qwen_calloc(size_t count, size_t size, const char *desc) {
    void *p = calloc(count, size);
    if (!p) {
        fprintf(stderr, "FATAL: calloc failed for %s (%.1f MB)\n",
                desc, (double)(count * size) / (1024*1024));
        exit(1);
    }
    return p;
}

// ── Metal GPU context (defined in main.m, used for GPU matmuls) ──────
#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

typedef struct {
    void *device;             // id<MTLDevice>
    void *queue;              // id<MTLCommandQueue>
    void *pipeline_f16;       // id<MTLComputePipelineState>
    void *pipeline_f32;       // id<MTLComputePipelineState>
    void *pipeline_q4;        // id<MTLComputePipelineState> for sgemv_q4
    void *pipeline_rms;       // id<MTLComputePipelineState> for rms_norm
    void *pipeline_rope;      // id<MTLComputePipelineState> for rope_apply
    void *pipeline_silu;      // id<MTLComputePipelineState> for silu_mul
    void *pipeline_add;       // id<MTLComputePipelineState> for vec_add
    void *pipeline_bias;      // id<MTLComputePipelineState> for bias_add
    void *pipeline_embed;     // id<MTLComputePipelineState> for embed_lookup
    void *pipeline_attn_score;    // id<MTLComputePipelineState>
    void *pipeline_softmax;       // id<MTLComputePipelineState>
    void *pipeline_attn_wsum;     // id<MTLComputePipelineState>
    void *pipeline_argmax;        // id<MTLComputePipelineState>
    void *pipeline_copy;          // id<MTLComputePipelineState>
    void *pipeline_zero;          // id<MTLComputePipelineState>
    void *pipeline_q4_fast;       // id<MTLComputePipelineState> for sgemv_q4_fast (SIMD)
    void *pipeline_q4_fused_ffn;  // id<MTLComputePipelineState> for fused gate+up+silu
    void *pipeline_attn_score_b;  // batched attn_score (all heads)
    void *pipeline_softmax_b;     // batched softmax (all heads)
    void *pipeline_attn_wsum_b;   // batched attn weighted sum (all heads)
    void *pipeline_sgemm_q4;      // batched Q4 matmul (prefill)
    void *pipeline_sgemm_q4_fused_ffn; // batched fused FFN (prefill)
    void *pipeline_rms_batched;   // batched RMSNorm (prefill)
    void *pipeline_embed_batched; // batched embed lookup (prefill)
    void *pipeline_rope_batched;  // batched RoPE (prefill)
    void *pipeline_add_batched;   // batched vec_add (prefill)
    void *x_buf;              // id<MTLBuffer> for input vector (reusable)
    void *y_buf;              // id<MTLBuffer> for output vector (reusable)
    int initialized;
} MetalContext;

static MetalContext g_metal = {0};

#ifndef QWEN_DEBUG
#define QWEN_DEBUG 0
#endif

// Qwen2.5-0.5B-Instruct architecture
#define QWEN_DIM         896
#define QWEN_HIDDEN      4864
#define QWEN_LAYERS      24
#define QWEN_HEADS       14
#define QWEN_KV_HEADS    2
#define QWEN_HEAD_DIM    64
#define QWEN_VOCAB       151936
#define QWEN_RMS_EPS     1e-6f
#define QWEN_ROPE_THETA  1000000.0f
#define QWEN_MAX_SEQ     512

// GQA: each KV head serves (HEADS / KV_HEADS) query heads
#define QWEN_GQA_FACTOR  (QWEN_HEADS / QWEN_KV_HEADS)

// Sizes for GQA projections
#define QWEN_Q_DIM       (QWEN_HEADS * QWEN_HEAD_DIM)      // 896
#define QWEN_KV_DIM      (QWEN_KV_HEADS * QWEN_HEAD_DIM)   // 128

typedef struct {
    // Weight format: 0 = F32 everywhere, 1 = F16 projections
    int weight_fmt;

    // Embeddings + norms always F32
    float *embed;                          // [vocab, dim]
    float *rms_att[QWEN_LAYERS];          // [dim]
    float *rms_ffn[QWEN_LAYERS];         // [dim]
    float *rms_final;                      // [dim]

    // Projection weights: F32 or F16 depending on weight_fmt
    // When weight_fmt=1, the f32 pointers are NULL and f16 pointers are set
    float *wq[QWEN_LAYERS];              // [q_dim, dim]  (F32)
    float *wk[QWEN_LAYERS];              // [kv_dim, dim] (F32)
    float *wv[QWEN_LAYERS];              // [kv_dim, dim] (F32)
    float *wo[QWEN_LAYERS];              // [dim, q_dim]  (F32)
    float *w_gate[QWEN_LAYERS];          // [hidden, dim] (F32)
    float *w_up[QWEN_LAYERS];            // [hidden, dim] (F32)
    float *w_down[QWEN_LAYERS];          // [dim, hidden] (F32)

    _Float16 *wq_f16[QWEN_LAYERS];       // (F16)
    _Float16 *wk_f16[QWEN_LAYERS];
    _Float16 *wv_f16[QWEN_LAYERS];
    _Float16 *wo_f16[QWEN_LAYERS];
    _Float16 *wgate_f16[QWEN_LAYERS];
    _Float16 *wup_f16[QWEN_LAYERS];
    _Float16 *wdown_f16[QWEN_LAYERS];

    uint8_t *wq_q8[QWEN_LAYERS];         // (Q8_0 blocks)
    uint8_t *wk_q8[QWEN_LAYERS];
    uint8_t *wv_q8[QWEN_LAYERS];
    uint8_t *wo_q8[QWEN_LAYERS];
    uint8_t *wgate_q8[QWEN_LAYERS];
    uint8_t *wup_q8[QWEN_LAYERS];
    uint8_t *wdown_q8[QWEN_LAYERS];

    // Metal GPU buffers (id<MTLBuffer> cast to void*)
    void *gpu_wq[QWEN_LAYERS];
    void *gpu_wk[QWEN_LAYERS];
    void *gpu_wv[QWEN_LAYERS];
    void *gpu_wo[QWEN_LAYERS];
    void *gpu_wgate[QWEN_LAYERS];
    void *gpu_wup[QWEN_LAYERS];
    void *gpu_wdown[QWEN_LAYERS];
    void *gpu_embed;   // embedding table (F32)
    void *gpu_rms_att[QWEN_LAYERS];   // RMSNorm weights
    void *gpu_rms_ffn[QWEN_LAYERS];
    void *gpu_rms_final;
    void *gpu_q_bias[QWEN_LAYERS];
    void *gpu_k_bias[QWEN_LAYERS];
    void *gpu_v_bias[QWEN_LAYERS];
    void *gpu_kv_cache_k[QWEN_LAYERS];
    void *gpu_kv_cache_v[QWEN_LAYERS];
    int use_gpu;
    // wcls = embed (tied, always F32)

    // ANE kernels -- unfused (one per linear projection per layer)
    ANEKernel *k_q[QWEN_LAYERS];
    ANEKernel *k_k[QWEN_LAYERS];
    ANEKernel *k_v[QWEN_LAYERS];
    ANEKernel *k_o[QWEN_LAYERS];
    ANEKernel *k_gate[QWEN_LAYERS];
    ANEKernel *k_up[QWEN_LAYERS];
    ANEKernel *k_down[QWEN_LAYERS];
    // LM head chunked: vocab too large for single ANE kernel (max 65536)
    #define QWEN_LM_CHUNKS 16
    #define QWEN_LM_CHUNK_SIZE 9496  // 151936 / 16
    ANEKernel *k_lmhead[QWEN_LM_CHUNKS];

    // ANE kernels -- fused (reduces 184 → 112 kernels, under 119 limit)
    ANEKernel *k_qkv[QWEN_LAYERS];     // fused Q+K+V → 3 outputs
    ANEKernel *k_ffn_up[QWEN_LAYERS];  // fused Gate+Up → 2 outputs
    int use_ane;  // 1 = fused ANE matmuls + CPU element-wise

    // Q/K/V biases per layer
    float *q_bias[QWEN_LAYERS];   // [q_dim]
    float *k_bias[QWEN_LAYERS];   // [kv_dim]
    float *v_bias[QWEN_LAYERS];   // [kv_dim]

    // KV cache [layer][kv_heads * head_dim * max_seq]
    float *kv_cache_k[QWEN_LAYERS];
    float *kv_cache_v[QWEN_LAYERS];
    int pos;  // current position in sequence

    // Scratch buffers
    float *x;       // [dim]
    float *xb;      // [dim]
    float *q;       // [q_dim]
    float *k;       // [kv_dim]
    float *v;       // [kv_dim]
    float *att;     // [heads * max_seq]
    float *hb;      // [hidden]
    float *hb2;     // [hidden]
    float *logits;  // [vocab]
} QwenModel;

// ── Precomputed RoPE table ───────────────────────────────────────────

static float g_rope_cos[QWEN_MAX_SEQ][QWEN_HEAD_DIM / 2];
static float g_rope_sin[QWEN_MAX_SEQ][QWEN_HEAD_DIM / 2];
static int g_rope_initialized = 0;

static void qwen_rope_init(void) {
    if (g_rope_initialized) return;
    int half = QWEN_HEAD_DIM / 2;
    for (int pos = 0; pos < QWEN_MAX_SEQ; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(QWEN_ROPE_THETA, (float)(2 * i) / QWEN_HEAD_DIM);
            float angle = pos * freq;
            g_rope_cos[pos][i] = cosf(angle);
            g_rope_sin[pos][i] = sinf(angle);
        }
    }
    g_rope_initialized = 1;
}

// ── CPU ops (vectorized with NEON + vDSP) ────────────────────────────

static void qwen_rmsnorm(float *out, const float *x, const float *w, int D) {
    float ss;
    vDSP_svesq(x, 1, &ss, (vDSP_Length)D);
    ss = 1.0f / sqrtf(ss / D + QWEN_RMS_EPS);
    vDSP_vsmul(x, 1, &ss, out, 1, (vDSP_Length)D);
    vDSP_vmul(out, 1, w, 1, out, 1, (vDSP_Length)D);
}

static void qwen_rope(float *q, float *k, int pos, int n_q_heads, int n_kv_heads, int head_dim) {
    int half = head_dim / 2;
    const float *cv = g_rope_cos[pos];
    const float *sv = g_rope_sin[pos];

    for (int h = 0; h < n_q_heads; h++) {
        float *qh = q + h * head_dim;
        int i = 0;
        for (; i + 3 < half; i += 4) {
            float32x4_t first  = vld1q_f32(qh + i);
            float32x4_t second = vld1q_f32(qh + i + half);
            float32x4_t c = vld1q_f32(cv + i);
            float32x4_t s = vld1q_f32(sv + i);
            vst1q_f32(qh + i,        vmlsq_f32(vmulq_f32(first, c), second, s));
            vst1q_f32(qh + i + half, vmlaq_f32(vmulq_f32(second, c), first, s));
        }
        for (; i < half; i++) {
            float f = qh[i], se = qh[i + half];
            qh[i]        = f * cv[i] - se * sv[i];
            qh[i + half] = se * cv[i] + f * sv[i];
        }
    }

    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        int i = 0;
        for (; i + 3 < half; i += 4) {
            float32x4_t first  = vld1q_f32(kh + i);
            float32x4_t second = vld1q_f32(kh + i + half);
            float32x4_t c = vld1q_f32(cv + i);
            float32x4_t s = vld1q_f32(sv + i);
            vst1q_f32(kh + i,        vmlsq_f32(vmulq_f32(first, c), second, s));
            vst1q_f32(kh + i + half, vmlaq_f32(vmulq_f32(second, c), first, s));
        }
        for (; i < half; i++) {
            float f = kh[i], se = kh[i + half];
            kh[i]        = f * cv[i] - se * sv[i];
            kh[i + half] = se * cv[i] + f * sv[i];
        }
    }
}

static void qwen_silu(float *x, int n) {
    int i = 0;
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float neg[4];
        vst1q_f32(neg, vnegq_f32(v));
        float exp_neg[4];
        for (int j = 0; j < 4; j++) exp_neg[j] = expf(neg[j]);
        float32x4_t denom = vaddq_f32(one, vld1q_f32(exp_neg));
        vst1q_f32(x + i, vdivq_f32(v, denom));
    }
    for (; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

// ── ANE projection helpers ──────────────────────────────────────────
// ANE IOSurfaces are always FP16 at the hardware level.
// We use g_fp16_io=1 MIL (FP16 I/O, no cast ops) and convert F32<->F16 here.

static inline bool ane_run(ANEKernel *k) { return ane_eval(k); }

static void ane_write_f32_as_f16(ANEKernel *kernel, int idx, const float *f32, int n) {
    IOSurfaceLock(kernel->ioInputs[idx], 0, NULL);
    _Float16 *dst = (_Float16 *)IOSurfaceGetBaseAddress(kernel->ioInputs[idx]);
    for (int i = 0; i < n; i++) dst[i] = (_Float16)f32[i];
    IOSurfaceUnlock(kernel->ioInputs[idx], 0, NULL);
}

static void ane_read_f16_to_f32(ANEKernel *kernel, int idx, float *f32, int n) {
    IOSurfaceLock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    const _Float16 *src = (const _Float16 *)IOSurfaceGetBaseAddress(kernel->ioOutputs[idx]);
    for (int i = 0; i < n; i++) f32[i] = (float)src[i];
    IOSurfaceUnlock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

static void ane_project(ANEKernel *kernel, const float *in, float *out,
                        int in_dim, int out_dim) {
    ane_write_f32_as_f16(kernel, 0, in, in_dim);
    ane_run(kernel);
    ane_read_f16_to_f32(kernel, 0, out, out_dim);
}

// Fused QKV: one ANE kernel → 3 outputs (Q, K, V with different dims)
static void ane_project_qkv(ANEKernel *kernel, const float *in,
                             float *q, float *k, float *v,
                             int in_dim, int q_dim, int kv_dim) {
    ane_write_f32_as_f16(kernel, 0, in, in_dim);
    ane_run(kernel);
    ane_read_f16_to_f32(kernel, 0, q, q_dim);
    ane_read_f16_to_f32(kernel, 1, k, kv_dim);
    ane_read_f16_to_f32(kernel, 2, v, kv_dim);
}

// Fused Gate+Up: one ANE kernel → 2 outputs (gate, up)
static void ane_project_ffn_up(ANEKernel *kernel, const float *in,
                                float *gate, float *up,
                                int in_dim, int hidden_dim) {
    ane_write_f32_as_f16(kernel, 0, in, in_dim);
    ane_run(kernel);
    ane_read_f16_to_f32(kernel, 0, gate, hidden_dim);
    ane_read_f16_to_f32(kernel, 1, up, hidden_dim);
}

// Compile fused QKV kernel (GQA-aware: Q=[q_dim,dim], K/V=[kv_dim,dim])
// Uses FP16 IOSurfaces (ANE hardware requirement)
static ANEKernel *compile_qkv_gqa_kernel(const float *wq, const float *wk, const float *wv,
                                          int dim, int q_dim, int kv_dim) {
    int saved = g_fp16_io; g_fp16_io = 1;
    NSData *wb = mil_build_qkv_gqa_weight_blob(wq, q_dim, dim, wk, wv, kv_dim);
    NSString *mil = mil_gen_qkv_gqa(dim, q_dim, kv_dim, 1);
    size_t inBytes = (size_t)dim * sizeof(_Float16);
    size_t outSizes[3] = {
        (size_t)q_dim * sizeof(_Float16),
        (size_t)kv_dim * sizeof(_Float16),
        (size_t)kv_dim * sizeof(_Float16)
    };
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], wb,
                               1, &inBytes, 3, outSizes);
    g_fp16_io = saved;
    return k;
}

// Compile fused FFN up kernel (Gate + Up, both [hidden_dim, dim])
static ANEKernel *compile_ffn_up_kernel(const float *w_gate, const float *w_up,
                                         int dim, int hidden_dim) {
    int saved = g_fp16_io; g_fp16_io = 1;
    NSData *wb = mil_build_ffn_up_weight_blob(w_gate, w_up, hidden_dim, dim);
    NSString *mil = mil_gen_ffn_up(dim, hidden_dim, 1);
    size_t inBytes = (size_t)dim * sizeof(_Float16);
    size_t outSizes[2] = {
        (size_t)hidden_dim * sizeof(_Float16),
        (size_t)hidden_dim * sizeof(_Float16)
    };
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], wb,
                               1, &inBytes, 2, outSizes);
    g_fp16_io = saved;
    return k;
}

// CPU matmul via Accelerate BLAS: y = W @ x, W[out_dim, in_dim]
static void cpu_project(const float *W, const float *x, float *y, int in_dim, int out_dim) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                out_dim, in_dim,
                1.0f, W, in_dim,
                x, 1,
                0.0f, y, 1);
}

// Bulk F16→F32 conversion using NEON vcvt
static void convert_f16_to_f32(const _Float16 *src, float *dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src + i));
        vst1q_f32(dst + i,     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++)
        dst[i] = (float)src[i];
}

// ── Q8_0 quantization support ────────────────────────────────────────
// Block format: 2 bytes F16 scale + 32 bytes int8 values = 34 bytes per block
#define Q8_BLOCK_SIZE 32
#define Q8_BLOCK_BYTES (2 + Q8_BLOCK_SIZE)  // 34

// Q8 matmul: y = W_q8 @ x, dequantize-and-dot using NEON int8
// W is stored as blocks of [f16_scale, 32*int8], row-major
static void cpu_project_q8(const uint8_t *W, const float *x, float *y,
                           int in_dim, int out_dim) {
    int n_blocks = in_dim / Q8_BLOCK_SIZE;
    size_t row_bytes = (size_t)n_blocks * Q8_BLOCK_BYTES;

    for (int r = 0; r < out_dim; r++) {
        const uint8_t *row = W + (size_t)r * row_bytes;
        float sum = 0.0f;

        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *block = row + (size_t)b * Q8_BLOCK_BYTES;
            _Float16 scale_f16;
            memcpy(&scale_f16, block, 2);
            float scale = (float)scale_f16;
            const int8_t *qvals = (const int8_t*)(block + 2);
            const float *xb = x + b * Q8_BLOCK_SIZE;

            // NEON: load 32 int8 values, widen to int16, convert to f32, FMA
            int8x16_t q0 = vld1q_s8(qvals);
            int8x16_t q1 = vld1q_s8(qvals + 16);

            // Widen int8 -> int16 -> int32 -> float32, then FMA with x
            int16x8_t w0 = vmovl_s8(vget_low_s8(q0));
            int16x8_t w1 = vmovl_s8(vget_high_s8(q0));
            int16x8_t w2 = vmovl_s8(vget_low_s8(q1));
            int16x8_t w3 = vmovl_s8(vget_high_s8(q1));

            float32x4_t a0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))),  vld1q_f32(xb));
            float32x4_t a1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))), vld1q_f32(xb + 4));
            float32x4_t a2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))),  vld1q_f32(xb + 8));
            float32x4_t a3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))), vld1q_f32(xb + 12));
            float32x4_t a4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2))),  vld1q_f32(xb + 16));
            float32x4_t a5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2))), vld1q_f32(xb + 20));
            float32x4_t a6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w3))),  vld1q_f32(xb + 24));
            float32x4_t a7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w3))), vld1q_f32(xb + 28));

            float32x4_t s01 = vaddq_f32(a0, a1);
            float32x4_t s23 = vaddq_f32(a2, a3);
            float32x4_t s45 = vaddq_f32(a4, a5);
            float32x4_t s67 = vaddq_f32(a6, a7);
            float32x4_t stot = vaddq_f32(vaddq_f32(s01, s23), vaddq_f32(s45, s67));
            sum += scale * vaddvq_f32(stot);
        }
        y[r] = sum;
    }
}

// ── Q4_0 block constants ─────────────────────────────────────────────
#define Q4_BLOCK_SIZE 32
#define Q4_BLOCK_BYTES 20   // 2(scale) + 2(zero) + 16(packed)

// ── Q4_0 dequantization helper: Q4 blocks to F32 ──
// Dequantizes one weight matrix from Q4 blocks into a caller-provided F32 buffer.
static void dequant_q4_to_f32(const uint8_t *W_q4, float *W_f32,
                               int in_dim, int out_dim) {
    int n_blocks = in_dim / Q4_BLOCK_SIZE;
    size_t row_bytes = (size_t)n_blocks * Q4_BLOCK_BYTES;

    for (int r = 0; r < out_dim; r++) {
        const uint8_t *row = W_q4 + (size_t)r * row_bytes;
        float *out_row = W_f32 + (size_t)r * in_dim;

        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *block = row + (size_t)b * Q4_BLOCK_BYTES;
            _Float16 scale_f16, zero_f16;
            memcpy(&scale_f16, block, 2);
            memcpy(&zero_f16, block + 2, 2);
            float scale = (float)scale_f16;
            float zero  = (float)zero_f16;
            const uint8_t *packed = block + 4;
            float *out = out_row + b * Q4_BLOCK_SIZE;

            for (int i = 0; i < 16; i++) {
                uint8_t byte = packed[i];
                out[i * 2]     = (float)(byte & 0xF) * scale + zero;
                out[i * 2 + 1] = (float)(byte >> 4)  * scale + zero;
            }
        }
    }
}

// Q4 fused NEON dequant-and-dot: reads Q4 from memory, avoids F32 intermediate
// Each block: 2B F16 scale + 2B F16 zero + 16B packed uint8 (32 values)
// Uses NEON to extract nibbles, convert to float, FMA with input vector
static void cpu_project_q4_amx(const uint8_t *W_q4, const float *x, float *y,
                                int in_dim, int out_dim) {
    int n_blocks = in_dim / Q4_BLOCK_SIZE;
    size_t row_bytes = (size_t)n_blocks * Q4_BLOCK_BYTES;

    for (int r = 0; r < out_dim; r++) {
        const uint8_t *row = W_q4 + (size_t)r * row_bytes;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *block = row + (size_t)b * Q4_BLOCK_BYTES;
            _Float16 scale_f16, zero_f16;
            memcpy(&scale_f16, block, 2);
            memcpy(&zero_f16, block + 2, 2);
            float scale = (float)scale_f16;
            float zero  = (float)zero_f16;
            const uint8_t *packed = block + 4;
            const float *xb = x + b * Q4_BLOCK_SIZE;
            float32x4_t vscale = vdupq_n_f32(scale);
            float32x4_t vzero  = vdupq_n_f32(zero);

            // Process 16 packed bytes = 32 values, 8 values at a time
            for (int i = 0; i < 16; i += 4) {
                // Load 4 packed bytes
                uint8x8_t raw = vld1_u8(packed + i);  // only 4 used

                // Extract low and high nibbles
                uint8_t b0 = packed[i], b1 = packed[i+1], b2 = packed[i+2], b3 = packed[i+3];

                // Even indices (low nibbles): b0&0xF, b1&0xF, b2&0xF, b3&0xF
                float32x4_t wlo = vmlaq_f32(vzero, vcvtq_f32_u32((uint32x4_t){
                    b0 & 0xF, b1 & 0xF, b2 & 0xF, b3 & 0xF}), vscale);
                // Odd indices (high nibbles): b0>>4, b1>>4, b2>>4, b3>>4
                float32x4_t whi = vmlaq_f32(vzero, vcvtq_f32_u32((uint32x4_t){
                    b0 >> 4, b1 >> 4, b2 >> 4, b3 >> 4}), vscale);

                // Interleaved dot: x[0]*w[0] + x[1]*w[1] + ... (even/odd pairs)
                int xi = i * 2;
                float32x4_t x_even = {xb[xi], xb[xi+2], xb[xi+4], xb[xi+6]};
                float32x4_t x_odd  = {xb[xi+1], xb[xi+3], xb[xi+5], xb[xi+7]};

                acc0 = vmlaq_f32(acc0, wlo, x_even);
                acc1 = vmlaq_f32(acc1, whi, x_odd);
            }
        }
        y[r] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}

// Q4 batched projection: dequant full matrix to F32, then cblas_sgemm
static void cpu_project_batch_q4_amx(const uint8_t *W_q4, const float *X, float *Y,
                                      int in_dim, int out_dim, int n_tokens) {
    float *W_f32 = (float*)malloc((size_t)out_dim * in_dim * sizeof(float));
    dequant_q4_to_f32(W_q4, W_f32, in_dim, out_dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_tokens, out_dim, in_dim,
                1.0f, X, in_dim,
                W_f32, in_dim,
                0.0f, Y, out_dim);
    free(W_f32);
}

// Toggle: 1 = use ANE for projections, 0 = CPU fallback
#define USE_ANE_PROJECTIONS 0

// ── Metal GPU matmul ─────────────────────────────────────────────────
#ifdef __OBJC__

static int metal_init(void) {
    if (g_metal.initialized) return 0;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "Metal: no GPU device\n"); return -1; }

    NSString *shaderPath = [[NSBundle mainBundle] pathForResource:@"matmul" ofType:@"metallib"];
    NSError *error = nil;
    id<MTLLibrary> lib = nil;

    // Try loading from compiled metallib next to binary
    NSString *execDir = [[[NSProcessInfo processInfo] arguments][0] stringByDeletingLastPathComponent];
    NSString *libPath = [execDir stringByAppendingPathComponent:@"matmul.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
        lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
    }

    // Fall back to compiling from source
    if (!lib) {
        NSString *srcPath = [execDir stringByAppendingPathComponent:@"matmul.metal"];
        NSString *src = [NSString stringWithContentsOfFile:srcPath
                                                 encoding:NSUTF8StringEncoding error:&error];
        if (!src) {
            fprintf(stderr, "Metal: cannot read shader source: %s\n",
                    [[error description] UTF8String]);
            return -1;
        }
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }
        lib = [dev newLibraryWithSource:src options:opts error:&error];
        if (!lib) {
            fprintf(stderr, "Metal: shader compile failed: %s\n",
                    [[error description] UTF8String]);
            return -1;
        }
    }

    // Build all pipeline states
    NSArray *names = @[
        @"sgemv_f16", @"sgemv_f32", @"sgemv_q4",
        @"rms_norm", @"rope_apply", @"silu_mul",
        @"vec_add", @"bias_add", @"embed_lookup",
        @"attn_score", @"softmax_inplace", @"attn_weighted_sum",
        @"argmax_kernel", @"vec_copy", @"vec_zero",
        @"sgemv_q4_fast", @"sgemv_q4_fused_ffn",
        @"attn_score_batched", @"softmax_batched", @"attn_wsum_batched",
        @"sgemm_q4", @"sgemm_q4_fused_ffn",
        @"rms_norm_batched", @"embed_lookup_batched",
        @"rope_apply_batched", @"vec_add_batched"
    ];
    void **pipelines[] = {
        &g_metal.pipeline_f16, &g_metal.pipeline_f32, &g_metal.pipeline_q4,
        &g_metal.pipeline_rms, &g_metal.pipeline_rope, &g_metal.pipeline_silu,
        &g_metal.pipeline_add, &g_metal.pipeline_bias, &g_metal.pipeline_embed,
        &g_metal.pipeline_attn_score, &g_metal.pipeline_softmax, &g_metal.pipeline_attn_wsum,
        &g_metal.pipeline_argmax, &g_metal.pipeline_copy, &g_metal.pipeline_zero,
        &g_metal.pipeline_q4_fast, &g_metal.pipeline_q4_fused_ffn,
        &g_metal.pipeline_attn_score_b, &g_metal.pipeline_softmax_b, &g_metal.pipeline_attn_wsum_b,
        &g_metal.pipeline_sgemm_q4, &g_metal.pipeline_sgemm_q4_fused_ffn,
        &g_metal.pipeline_rms_batched, &g_metal.pipeline_embed_batched,
        &g_metal.pipeline_rope_batched, &g_metal.pipeline_add_batched
    };

    for (int i = 0; i < (int)[names count]; i++) {
        id<MTLFunction> fn = [lib newFunctionWithName:names[i]];
        if (!fn) {
            fprintf(stderr, "Metal: missing shader function '%s'\n", [names[i] UTF8String]);
            return -1;
        }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            fprintf(stderr, "Metal: pipeline for '%s' failed: %s\n",
                    [names[i] UTF8String], [[error description] UTF8String]);
            return -1;
        }
        *pipelines[i] = (__bridge_retained void*)pso;
    }

    g_metal.device = (__bridge_retained void*)dev;
    g_metal.queue = (__bridge_retained void*)[dev newCommandQueue];

    g_metal.initialized = 1;
    printf("Metal GPU initialized (%s)\n", [[dev name] UTF8String]);
    return 0;
}

// GPU projection for F16 weights: dispatches Metal compute shader
// Uses per-call output buffers to allow batching multiple projections
static void gpu_project_f16(id<MTLBuffer> w_buf, const float *x, float *y,
                            int in_dim, int out_dim) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)g_metal.device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)g_metal.queue;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_f16;

    // Shared input/output buffers
    id<MTLBuffer> x_buf = [dev newBufferWithBytes:x
                                           length:in_dim * sizeof(float)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> y_buf = [dev newBufferWithLength:out_dim * sizeof(float)
                                           options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:w_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:y_buf offset:0 atIndex:2];
    uint32_t dims[2] = {(uint32_t)in_dim, (uint32_t)out_dim};
    [enc setBytes:&dims[0] length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&dims[1] length:sizeof(uint32_t) atIndex:4];

    NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)out_dim) tpg = (NSUInteger)out_dim;
    [enc dispatchThreads:MTLSizeMake(out_dim, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(y, [y_buf contents], out_dim * sizeof(float));
}

#endif // __OBJC__

// ── Forward one token ────────────────────────────────────────────────

static int qwen_forward(QwenModel *m, int token) {
    int D = QWEN_DIM, HD = QWEN_HIDDEN;
    int pos = m->pos;

    // Token embedding
    memcpy(m->x, m->embed + token * D, D * sizeof(float));

    for (int l = 0; l < QWEN_LAYERS; l++) {
        // Attention RMSNorm
        qwen_rmsnorm(m->xb, m->x, m->rms_att[l], D);

#if QWEN_DEBUG
        if (l == 0 && pos == 0) {
            float xnorm = 0;
            for (int i = 0; i < D; i++) xnorm += m->xb[i] * m->xb[i];
            printf("  L0 RMSNorm out norm=%.4f (first 4: %.4f %.4f %.4f %.4f)\n",
                   sqrtf(xnorm), m->xb[0], m->xb[1], m->xb[2], m->xb[3]);
        }
#endif

        // QKV projections + bias (CPU path -- GPU overhead too high for small matmuls)
        #if USE_ANE_PROJECTIONS
        ane_project(m->k_q[l], m->xb, m->q, D, QWEN_Q_DIM);
        ane_project(m->k_k[l], m->xb, m->k, D, QWEN_KV_DIM);
        ane_project(m->k_v[l], m->xb, m->v, D, QWEN_KV_DIM);
        #else
        if (m->weight_fmt == 3) {
            cpu_project_q4_amx(m->wq_q8[l], m->xb, m->q, D, QWEN_Q_DIM);
            cpu_project_q4_amx(m->wk_q8[l], m->xb, m->k, D, QWEN_KV_DIM);
            cpu_project_q4_amx(m->wv_q8[l], m->xb, m->v, D, QWEN_KV_DIM);
        } else if (m->weight_fmt == 2) {
            cpu_project_q8(m->wq_q8[l], m->xb, m->q, D, QWEN_Q_DIM);
            cpu_project_q8(m->wk_q8[l], m->xb, m->k, D, QWEN_KV_DIM);
            cpu_project_q8(m->wv_q8[l], m->xb, m->v, D, QWEN_KV_DIM);
        } else {
            cpu_project(m->wq[l], m->xb, m->q, D, QWEN_Q_DIM);
            cpu_project(m->wk[l], m->xb, m->k, D, QWEN_KV_DIM);
            cpu_project(m->wv[l], m->xb, m->v, D, QWEN_KV_DIM);
        }
        #endif
        // Apply Q/K/V biases (vectorized)
        if (m->q_bias[l])
            vDSP_vadd(m->q, 1, m->q_bias[l], 1, m->q, 1, (vDSP_Length)QWEN_Q_DIM);
        if (m->k_bias[l])
            vDSP_vadd(m->k, 1, m->k_bias[l], 1, m->k, 1, (vDSP_Length)QWEN_KV_DIM);
        if (m->v_bias[l])
            vDSP_vadd(m->v, 1, m->v_bias[l], 1, m->v, 1, (vDSP_Length)QWEN_KV_DIM);

#if QWEN_DEBUG
        if (l == 0 && pos == 0) {
            float qn = 0;
            for (int i = 0; i < QWEN_Q_DIM; i++) qn += m->q[i] * m->q[i];
            printf("  L0 ANE Q norm=%.4f (first 4: %.4f %.4f %.4f %.4f)\n",
                   sqrtf(qn), m->q[0], m->q[1], m->q[2], m->q[3]);
            float cpu_q[4] = {0};
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < D; j++)
                    cpu_q[i] += m->wq[0][i * D + j] * m->xb[j];
                cpu_q[i] += m->q_bias[0][i];
            }
            printf("  L0 CPU Q first 4: %.4f %.4f %.4f %.4f\n",
                   cpu_q[0], cpu_q[1], cpu_q[2], cpu_q[3]);
        }
#endif

        // RoPE
        qwen_rope(m->q, m->k, pos, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM);

        // Store K, V in cache
        memcpy(m->kv_cache_k[l] + pos * QWEN_KV_DIM,
               m->k, QWEN_KV_DIM * sizeof(float));
        memcpy(m->kv_cache_v[l] + pos * QWEN_KV_DIM,
               m->v, QWEN_KV_DIM * sizeof(float));

        // GQA attention (CPU — element-wise ops)
        float scale = 1.0f / sqrtf((float)QWEN_HEAD_DIM);
        float *attn_out = m->xb;  // reuse buffer
        memset(attn_out, 0, QWEN_Q_DIM * sizeof(float));

        for (int h = 0; h < QWEN_HEADS; h++) {
            int kv_h = h / QWEN_GQA_FACTOR;
            float *qh = m->q + h * QWEN_HEAD_DIM;
            float *att_h = m->att + h * QWEN_MAX_SEQ;
            int seq_len = pos + 1;

            // Attention scores: Q @ K^T
            float max_score = -1e9f;
            for (int t = 0; t <= pos; t++) {
                float *kt = m->kv_cache_k[l] + t * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                float score = cblas_sdot(QWEN_HEAD_DIM, qh, 1, kt, 1);
                att_h[t] = score * scale;
                if (att_h[t] > max_score) max_score = att_h[t];
            }
            // Softmax: subtract max, exp, normalize (vDSP)
            float neg_max = -max_score;
            vDSP_vsadd(att_h, 1, &neg_max, att_h, 1, (vDSP_Length)seq_len);
            int n_exp = seq_len;
            vvexpf(att_h, att_h, &n_exp);
            float sum;
            vDSP_sve(att_h, 1, &sum, (vDSP_Length)seq_len);
            float inv_sum = 1.0f / sum;
            vDSP_vsmul(att_h, 1, &inv_sum, att_h, 1, (vDSP_Length)seq_len);

            // Weighted sum of V
            for (int t = 0; t <= pos; t++) {
                float a = att_h[t];
                float *vt = m->kv_cache_v[l] + t * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                cblas_saxpy(QWEN_HEAD_DIM, a, vt, 1,
                           attn_out + h * QWEN_HEAD_DIM, 1);
            }
        }

        float o_out[QWEN_DIM];
        #if USE_ANE_PROJECTIONS
        ane_project(m->k_o[l], attn_out, o_out, QWEN_Q_DIM, D);
        #else
        if (m->weight_fmt == 3)
            cpu_project_q4_amx(m->wo_q8[l], attn_out, o_out, QWEN_Q_DIM, D);
        else if (m->weight_fmt == 2)
            cpu_project_q8(m->wo_q8[l], attn_out, o_out, QWEN_Q_DIM, D);
        else
            cpu_project(m->wo[l], attn_out, o_out, QWEN_Q_DIM, D);
        #endif

        // Residual (vectorized)
        vDSP_vadd(m->x, 1, o_out, 1, m->x, 1, (vDSP_Length)D);

#if QWEN_DEBUG
        if (l == 0 && pos == 0) {
            float pan = 0;
            for (int i = 0; i < D; i++) pan += m->x[i] * m->x[i];
            printf("  L0 post-attn norm=%.4f first4=[%.6f, %.6f, %.6f, %.6f]\n",
                   sqrtf(pan), m->x[0], m->x[1], m->x[2], m->x[3]);
            float on = 0;
            for (int i = 0; i < D; i++) on += o_out[i] * o_out[i];
            printf("  L0 o_proj out norm=%.4f first4=[%.6f, %.6f, %.6f, %.6f]\n",
                   sqrtf(on), o_out[0], o_out[1], o_out[2], o_out[3]);
        }
#endif

        // FFN RMSNorm
        qwen_rmsnorm(m->xb, m->x, m->rms_ffn[l], D);

        // SwiGLU FFN
        #if USE_ANE_PROJECTIONS
        ane_project(m->k_gate[l], m->xb, m->hb, D, HD);
        ane_project(m->k_up[l], m->xb, m->hb2, D, HD);
        #else
        if (m->weight_fmt == 3) {
            cpu_project_q4_amx(m->wgate_q8[l], m->xb, m->hb, D, HD);
            cpu_project_q4_amx(m->wup_q8[l], m->xb, m->hb2, D, HD);
        } else if (m->weight_fmt == 2) {
            cpu_project_q8(m->wgate_q8[l], m->xb, m->hb, D, HD);
            cpu_project_q8(m->wup_q8[l], m->xb, m->hb2, D, HD);
        } else {
            cpu_project(m->w_gate[l], m->xb, m->hb, D, HD);
            cpu_project(m->w_up[l], m->xb, m->hb2, D, HD);
        }
        #endif

#if QWEN_DEBUG
        if (l == 0 && pos == 0) {
            float gn = 0, un = 0;
            for (int i = 0; i < HD; i++) { gn += m->hb[i]*m->hb[i]; un += m->hb2[i]*m->hb2[i]; }
            printf("  L0 gate norm=%.4f up norm=%.4f\n", sqrtf(gn), sqrtf(un));
            printf("  L0 gate first4=[%.6f, %.6f, %.6f, %.6f]\n",
                   m->hb[0], m->hb[1], m->hb[2], m->hb[3]);
        }
#endif

        qwen_silu(m->hb, HD);
        // SiLU(gate) * up (vectorized element-wise multiply)
        vDSP_vmul(m->hb, 1, m->hb2, 1, m->hb, 1, (vDSP_Length)HD);

        float ffn_out[QWEN_DIM];
        #if USE_ANE_PROJECTIONS
        ane_project(m->k_down[l], m->hb, ffn_out, HD, D);
        #else
        if (m->weight_fmt == 3)
            cpu_project_q4_amx(m->wdown_q8[l], m->hb, ffn_out, HD, D);
        else if (m->weight_fmt == 2)
            cpu_project_q8(m->wdown_q8[l], m->hb, ffn_out, HD, D);
        else
            cpu_project(m->w_down[l], m->hb, ffn_out, HD, D);
        #endif

        // Residual (vectorized)
        vDSP_vadd(m->x, 1, ffn_out, 1, m->x, 1, (vDSP_Length)D);

#if QWEN_DEBUG
        if (l < 3 && pos == 0) {
            float hn = 0;
            for (int i = 0; i < D; i++) hn += m->x[i] * m->x[i];
            printf("  C hidden[%d] norm=%.4f first4=[%.4f, %.4f, %.4f, %.4f]\n",
                   l+1, sqrtf(hn), m->x[0], m->x[1], m->x[2], m->x[3]);
        }
#endif
    }

    // Final RMSNorm
    qwen_rmsnorm(m->xb, m->x, m->rms_final, D);

#if QWEN_DEBUG
    if (m->pos < 2) {
        float fn = 0;
        for (int i = 0; i < D; i++) fn += m->xb[i] * m->xb[i];
        printf("  Final hidden norm=%.4f (first 4: %.6f %.6f %.6f %.6f)\n",
               sqrtf(fn), m->xb[0], m->xb[1], m->xb[2], m->xb[3]);
    }
#endif

    // LM head via Accelerate BLAS (AMX, fastest for dim<=896)
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                QWEN_VOCAB, D,
                1.0f, m->embed, D,
                m->xb, 1,
                0.0f, m->logits, 1);

#if QWEN_DEBUG
    if (m->pos < 2) {
        float lmax = m->logits[0], lmin = m->logits[0];
        int nonzero = 0;
        for (int i = 0; i < QWEN_VOCAB; i++) {
            if (m->logits[i] > lmax) lmax = m->logits[i];
            if (m->logits[i] < lmin) lmin = m->logits[i];
            if (m->logits[i] != 0.0f) nonzero++;
        }
        printf("  Logits: min=%.4f max=%.4f nonzero=%d/%d\n", lmin, lmax, nonzero, QWEN_VOCAB);
    }
#endif

    m->pos++;

    // Argmax (vDSP, single call over 151936 elements)
    float max_val;
    vDSP_Length max_idx_vdsp;
    vDSP_maxvi(m->logits, 1, &max_val, &max_idx_vdsp, (vDSP_Length)QWEN_VOCAB);
    return (int)max_idx_vdsp;
}

// ── ANE fused forward pass: ANE for matmuls, CPU for element-wise ops ──
// Uses fused QKV and Gate+Up kernels (112 total, under 119 ANE limit).
// O-proj and Down-proj remain as single conv kernels.

static int qwen_forward_ane(QwenModel *m, int token) {
    int D = QWEN_DIM, HD = QWEN_HIDDEN;
    int pos = m->pos;

    memcpy(m->x, m->embed + token * D, D * sizeof(float));

    for (int l = 0; l < QWEN_LAYERS; l++) {
        qwen_rmsnorm(m->xb, m->x, m->rms_att[l], D);

        // Fused QKV projection (1 ANE eval → Q, K, V)
        ane_project_qkv(m->k_qkv[l], m->xb, m->q, m->k, m->v,
                         D, QWEN_Q_DIM, QWEN_KV_DIM);

        // Biases (CPU, vectorized)
        if (m->q_bias[l])
            vDSP_vadd(m->q, 1, m->q_bias[l], 1, m->q, 1, (vDSP_Length)QWEN_Q_DIM);
        if (m->k_bias[l])
            vDSP_vadd(m->k, 1, m->k_bias[l], 1, m->k, 1, (vDSP_Length)QWEN_KV_DIM);
        if (m->v_bias[l])
            vDSP_vadd(m->v, 1, m->v_bias[l], 1, m->v, 1, (vDSP_Length)QWEN_KV_DIM);

        qwen_rope(m->q, m->k, pos, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM);

        memcpy(m->kv_cache_k[l] + pos * QWEN_KV_DIM, m->k, QWEN_KV_DIM * sizeof(float));
        memcpy(m->kv_cache_v[l] + pos * QWEN_KV_DIM, m->v, QWEN_KV_DIM * sizeof(float));

        // GQA attention (CPU)
        float scale = 1.0f / sqrtf((float)QWEN_HEAD_DIM);
        float *attn_out = m->xb;
        memset(attn_out, 0, QWEN_Q_DIM * sizeof(float));

        for (int h = 0; h < QWEN_HEADS; h++) {
            int kv_h = h / QWEN_GQA_FACTOR;
            float *qh = m->q + h * QWEN_HEAD_DIM;
            float *att_h = m->att + h * QWEN_MAX_SEQ;
            int seq_len = pos + 1;

            float max_score = -1e9f;
            for (int t = 0; t <= pos; t++) {
                float *kt = m->kv_cache_k[l] + t * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                float score = cblas_sdot(QWEN_HEAD_DIM, qh, 1, kt, 1);
                att_h[t] = score * scale;
                if (att_h[t] > max_score) max_score = att_h[t];
            }
            float neg_max = -max_score;
            vDSP_vsadd(att_h, 1, &neg_max, att_h, 1, (vDSP_Length)seq_len);
            int n_exp = seq_len;
            vvexpf(att_h, att_h, &n_exp);
            float sum;
            vDSP_sve(att_h, 1, &sum, (vDSP_Length)seq_len);
            float inv_sum = 1.0f / sum;
            vDSP_vsmul(att_h, 1, &inv_sum, att_h, 1, (vDSP_Length)seq_len);

            for (int t = 0; t <= pos; t++) {
                float a = att_h[t];
                float *vt = m->kv_cache_v[l] + t * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                cblas_saxpy(QWEN_HEAD_DIM, a, vt, 1, attn_out + h * QWEN_HEAD_DIM, 1);
            }
        }

        // O projection (single ANE kernel)
        float o_out[QWEN_DIM];
        ane_project(m->k_o[l], attn_out, o_out, QWEN_Q_DIM, D);

        vDSP_vadd(m->x, 1, o_out, 1, m->x, 1, (vDSP_Length)D);

        qwen_rmsnorm(m->xb, m->x, m->rms_ffn[l], D);

        // Fused Gate+Up projection (1 ANE eval → gate, up)
        ane_project_ffn_up(m->k_ffn_up[l], m->xb, m->hb, m->hb2, D, HD);

        qwen_silu(m->hb, HD);
        vDSP_vmul(m->hb, 1, m->hb2, 1, m->hb, 1, (vDSP_Length)HD);

        // Down projection (single ANE kernel)
        float ffn_out[QWEN_DIM];
        ane_project(m->k_down[l], m->hb, ffn_out, HD, D);

        vDSP_vadd(m->x, 1, ffn_out, 1, m->x, 1, (vDSP_Length)D);
    }

    qwen_rmsnorm(m->xb, m->x, m->rms_final, D);

    // LM head: CPU AMX (too large for ANE, 151936 outputs)
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                QWEN_VOCAB, D,
                1.0f, m->embed, D,
                m->xb, 1,
                0.0f, m->logits, 1);

    m->pos++;

    float max_val;
    vDSP_Length max_idx_vdsp;
    vDSP_maxvi(m->logits, 1, &max_val, &max_idx_vdsp, (vDSP_Length)QWEN_VOCAB);
    return (int)max_idx_vdsp;
}

// ── Batched prefill: process all prompt tokens at once ────────────────
// Uses cblas_sgemm (matrix-matrix) instead of sequential sgemv calls.
// Returns the argmax token from the last position's logits.

static void cpu_project_batch(const float *W, const float *X, float *Y,
                               int in_dim, int out_dim, int n_tokens) {
    // X[n_tokens, in_dim], W[out_dim, in_dim], Y[n_tokens, out_dim]
    // Y = X @ W^T  =>  Y(n,out) = sum_k X(n,k) * W(out,k)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_tokens, out_dim, in_dim,
                1.0f, X, in_dim,
                W, in_dim,
                0.0f, Y, out_dim);
}

static int qwen_prefill(QwenModel *m, const int *tokens, int n_tokens) {
    int D = QWEN_DIM, HD = QWEN_HIDDEN, N = n_tokens;

    float *xs   = (float*)qwen_calloc(N * D, sizeof(float), "prefill_xs");
    float *xbs  = (float*)qwen_calloc(N * D, sizeof(float), "prefill_xbs");
    float *qs   = (float*)qwen_calloc(N * QWEN_Q_DIM, sizeof(float), "prefill_qs");
    float *ks   = (float*)qwen_calloc(N * QWEN_KV_DIM, sizeof(float), "prefill_ks");
    float *vs   = (float*)qwen_calloc(N * QWEN_KV_DIM, sizeof(float), "prefill_vs");
    float *hbs  = (float*)qwen_calloc(N * HD, sizeof(float), "prefill_hbs");
    float *hb2s = (float*)qwen_calloc(N * HD, sizeof(float), "prefill_hb2s");
    float *o_outs   = (float*)qwen_calloc(N * D, sizeof(float), "prefill_o_outs");
    float *ffn_outs = (float*)qwen_calloc(N * D, sizeof(float), "prefill_ffn_outs");

    // Load all embeddings
    for (int t = 0; t < N; t++)
        memcpy(xs + t * D, m->embed + tokens[t] * D, D * sizeof(float));

    for (int l = 0; l < QWEN_LAYERS; l++) {
        // Batch RMSNorm
        for (int t = 0; t < N; t++)
            qwen_rmsnorm(xbs + t * D, xs + t * D, m->rms_att[l], D);

        // Batch QKV projections: sgemm
        cpu_project_batch(m->wq[l], xbs, qs, D, QWEN_Q_DIM, N);
        cpu_project_batch(m->wk[l], xbs, ks, D, QWEN_KV_DIM, N);
        cpu_project_batch(m->wv[l], xbs, vs, D, QWEN_KV_DIM, N);

        // Per-token: bias + RoPE + cache + attention
        for (int t = 0; t < N; t++) {
            float *qt = qs + t * QWEN_Q_DIM;
            float *kt = ks + t * QWEN_KV_DIM;
            float *vt = vs + t * QWEN_KV_DIM;
            int pos = m->pos + t;

            // Biases
            if (m->q_bias[l])
                vDSP_vadd(qt, 1, m->q_bias[l], 1, qt, 1, (vDSP_Length)QWEN_Q_DIM);
            if (m->k_bias[l])
                vDSP_vadd(kt, 1, m->k_bias[l], 1, kt, 1, (vDSP_Length)QWEN_KV_DIM);
            if (m->v_bias[l])
                vDSP_vadd(vt, 1, m->v_bias[l], 1, vt, 1, (vDSP_Length)QWEN_KV_DIM);

            // RoPE
            qwen_rope(qt, kt, pos, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM);

            // Store K, V in cache
            memcpy(m->kv_cache_k[l] + pos * QWEN_KV_DIM, kt, QWEN_KV_DIM * sizeof(float));
            memcpy(m->kv_cache_v[l] + pos * QWEN_KV_DIM, vt, QWEN_KV_DIM * sizeof(float));

            // GQA attention
            float scale = 1.0f / sqrtf((float)QWEN_HEAD_DIM);
            float *attn_out = xbs + t * D;
            memset(attn_out, 0, QWEN_Q_DIM * sizeof(float));

            for (int h = 0; h < QWEN_HEADS; h++) {
                int kv_h = h / QWEN_GQA_FACTOR;
                float *qh = qt + h * QWEN_HEAD_DIM;
                float *att_h = m->att + h * QWEN_MAX_SEQ;
                int seq_len = pos + 1;

                float max_score = -1e9f;
                for (int p = 0; p <= pos; p++) {
                    float *kp = m->kv_cache_k[l] + p * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                    float score = cblas_sdot(QWEN_HEAD_DIM, qh, 1, kp, 1);
                    att_h[p] = score * scale;
                    if (att_h[p] > max_score) max_score = att_h[p];
                }
                float neg_max = -max_score;
                vDSP_vsadd(att_h, 1, &neg_max, att_h, 1, (vDSP_Length)seq_len);
                int n_exp = seq_len;
                vvexpf(att_h, att_h, &n_exp);
                float sum;
                vDSP_sve(att_h, 1, &sum, (vDSP_Length)seq_len);
                float inv_sum = 1.0f / sum;
                vDSP_vsmul(att_h, 1, &inv_sum, att_h, 1, (vDSP_Length)seq_len);

                for (int p = 0; p <= pos; p++) {
                    float a = att_h[p];
                    float *vp = m->kv_cache_v[l] + p * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                    cblas_saxpy(QWEN_HEAD_DIM, a, vp, 1,
                               attn_out + h * QWEN_HEAD_DIM, 1);
                }
            }
        }
        // xbs now has [N, Q_DIM] attention outputs

        // Batch O projection (reuses pre-allocated o_outs)
        cpu_project_batch(m->wo[l], xbs, o_outs, QWEN_Q_DIM, D, N);

        for (int t = 0; t < N; t++)
            vDSP_vadd(xs + t * D, 1, o_outs + t * D, 1, xs + t * D, 1, (vDSP_Length)D);

        // Batch FFN RMSNorm
        for (int t = 0; t < N; t++)
            qwen_rmsnorm(xbs + t * D, xs + t * D, m->rms_ffn[l], D);

        // Batch FFN projections
        cpu_project_batch(m->w_gate[l], xbs, hbs,  D, HD, N);
        cpu_project_batch(m->w_up[l],   xbs, hb2s, D, HD, N);

        for (int t = 0; t < N; t++) {
            qwen_silu(hbs + t * HD, HD);
            vDSP_vmul(hbs + t * HD, 1, hb2s + t * HD, 1, hbs + t * HD, 1, (vDSP_Length)HD);
        }

        // Batch down projection (reuses pre-allocated ffn_outs)
        cpu_project_batch(m->w_down[l], hbs, ffn_outs, HD, D, N);

        for (int t = 0; t < N; t++)
            vDSP_vadd(xs + t * D, 1, ffn_outs + t * D, 1, xs + t * D, 1, (vDSP_Length)D);
    }

    // Only need logits for the last token
    float *last_x = xs + (N - 1) * D;
    qwen_rmsnorm(m->xb, last_x, m->rms_final, D);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                QWEN_VOCAB, D,
                1.0f, m->embed, D,
                m->xb, 1,
                0.0f, m->logits, 1);

    m->pos += N;

    float max_val;
    vDSP_Length max_idx_vdsp;
    vDSP_maxvi(m->logits, 1, &max_val, &max_idx_vdsp, (vDSP_Length)QWEN_VOCAB);

    free(xs); free(xbs); free(qs); free(ks); free(vs); free(hbs); free(hb2s);
    free(o_outs); free(ffn_outs);
    return (int)max_idx_vdsp;
}

// Q4 AMX batched prefill: dequantize weight matrices then use sgemm
static int qwen_prefill_q4(QwenModel *m, const int *tokens, int n_tokens) {
    int D = QWEN_DIM, HD = QWEN_HIDDEN, N = n_tokens;

    float *xs   = (float*)qwen_calloc(N * D, sizeof(float), "prefill_q4_xs");
    float *xbs  = (float*)qwen_calloc(N * D, sizeof(float), "prefill_q4_xbs");
    float *qs   = (float*)qwen_calloc(N * QWEN_Q_DIM, sizeof(float), "prefill_q4_qs");
    float *ks   = (float*)qwen_calloc(N * QWEN_KV_DIM, sizeof(float), "prefill_q4_ks");
    float *vs   = (float*)qwen_calloc(N * QWEN_KV_DIM, sizeof(float), "prefill_q4_vs");
    float *hbs  = (float*)qwen_calloc(N * HD, sizeof(float), "prefill_q4_hbs");
    float *hb2s = (float*)qwen_calloc(N * HD, sizeof(float), "prefill_q4_hb2s");
    float *o_outs   = (float*)qwen_calloc(N * D, sizeof(float), "prefill_q4_o_outs");
    float *ffn_outs = (float*)qwen_calloc(N * D, sizeof(float), "prefill_q4_ffn_outs");

    for (int t = 0; t < N; t++)
        memcpy(xs + t * D, m->embed + tokens[t] * D, D * sizeof(float));

    for (int l = 0; l < QWEN_LAYERS; l++) {
        for (int t = 0; t < N; t++)
            qwen_rmsnorm(xbs + t * D, xs + t * D, m->rms_att[l], D);

        cpu_project_batch_q4_amx(m->wq_q8[l], xbs, qs, D, QWEN_Q_DIM, N);
        cpu_project_batch_q4_amx(m->wk_q8[l], xbs, ks, D, QWEN_KV_DIM, N);
        cpu_project_batch_q4_amx(m->wv_q8[l], xbs, vs, D, QWEN_KV_DIM, N);

        for (int t = 0; t < N; t++) {
            float *qt = qs + t * QWEN_Q_DIM;
            float *kt = ks + t * QWEN_KV_DIM;
            float *vt = vs + t * QWEN_KV_DIM;
            int pos = m->pos + t;

            if (m->q_bias[l])
                vDSP_vadd(qt, 1, m->q_bias[l], 1, qt, 1, (vDSP_Length)QWEN_Q_DIM);
            if (m->k_bias[l])
                vDSP_vadd(kt, 1, m->k_bias[l], 1, kt, 1, (vDSP_Length)QWEN_KV_DIM);
            if (m->v_bias[l])
                vDSP_vadd(vt, 1, m->v_bias[l], 1, vt, 1, (vDSP_Length)QWEN_KV_DIM);

            qwen_rope(qt, kt, pos, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM);

            memcpy(m->kv_cache_k[l] + pos * QWEN_KV_DIM, kt, QWEN_KV_DIM * sizeof(float));
            memcpy(m->kv_cache_v[l] + pos * QWEN_KV_DIM, vt, QWEN_KV_DIM * sizeof(float));

            float scale = 1.0f / sqrtf((float)QWEN_HEAD_DIM);
            float *attn_out = xbs + t * D;
            memset(attn_out, 0, QWEN_Q_DIM * sizeof(float));

            for (int h = 0; h < QWEN_HEADS; h++) {
                int kv_h = h / QWEN_GQA_FACTOR;
                float *qh = qt + h * QWEN_HEAD_DIM;
                float *att_h = m->att + h * QWEN_MAX_SEQ;
                int seq_len = pos + 1;

                float max_score = -1e9f;
                for (int p = 0; p <= pos; p++) {
                    float *kp = m->kv_cache_k[l] + p * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                    float score = cblas_sdot(QWEN_HEAD_DIM, qh, 1, kp, 1);
                    att_h[p] = score * scale;
                    if (att_h[p] > max_score) max_score = att_h[p];
                }
                float neg_max = -max_score;
                vDSP_vsadd(att_h, 1, &neg_max, att_h, 1, (vDSP_Length)seq_len);
                int n_exp = seq_len;
                vvexpf(att_h, att_h, &n_exp);
                float sum;
                vDSP_sve(att_h, 1, &sum, (vDSP_Length)seq_len);
                float inv_sum = 1.0f / sum;
                vDSP_vsmul(att_h, 1, &inv_sum, att_h, 1, (vDSP_Length)seq_len);

                for (int p = 0; p <= pos; p++) {
                    float a = att_h[p];
                    float *vp = m->kv_cache_v[l] + p * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                    cblas_saxpy(QWEN_HEAD_DIM, a, vp, 1,
                               attn_out + h * QWEN_HEAD_DIM, 1);
                }
            }
        }

        cpu_project_batch_q4_amx(m->wo_q8[l], xbs, o_outs, QWEN_Q_DIM, D, N);

        for (int t = 0; t < N; t++)
            vDSP_vadd(xs + t * D, 1, o_outs + t * D, 1, xs + t * D, 1, (vDSP_Length)D);

        for (int t = 0; t < N; t++)
            qwen_rmsnorm(xbs + t * D, xs + t * D, m->rms_ffn[l], D);

        cpu_project_batch_q4_amx(m->wgate_q8[l], xbs, hbs,  D, HD, N);
        cpu_project_batch_q4_amx(m->wup_q8[l],   xbs, hb2s, D, HD, N);

        for (int t = 0; t < N; t++) {
            qwen_silu(hbs + t * HD, HD);
            vDSP_vmul(hbs + t * HD, 1, hb2s + t * HD, 1, hbs + t * HD, 1, (vDSP_Length)HD);
        }

        cpu_project_batch_q4_amx(m->wdown_q8[l], hbs, ffn_outs, HD, D, N);

        for (int t = 0; t < N; t++)
            vDSP_vadd(xs + t * D, 1, ffn_outs + t * D, 1, xs + t * D, 1, (vDSP_Length)D);
    }

    float *last_x = xs + (N - 1) * D;
    qwen_rmsnorm(m->xb, last_x, m->rms_final, D);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                QWEN_VOCAB, D,
                1.0f, m->embed, D,
                m->xb, 1,
                0.0f, m->logits, 1);

    m->pos += N;

    float max_val;
    vDSP_Length max_idx_vdsp;
    vDSP_maxvi(m->logits, 1, &max_val, &max_idx_vdsp, (vDSP_Length)QWEN_VOCAB);

    free(xs); free(xbs); free(qs); free(ks); free(vs); free(hbs); free(hb2s);
    free(o_outs); free(ffn_outs);
    return (int)max_idx_vdsp;
}

// ── Full-GPU forward pass (Metal, single command buffer per layer) ────
// Runs entire transformer on GPU using Q4 quantized weights.
// KV cache stays on GPU between calls. Attention runs per-head on GPU.
#ifdef __OBJC__

// SIMD-optimized Q4 matvec with optional bias fusion.
// 2 SIMD groups x 4 rows each = 8 rows/threadgroup, simd_sum reduction.
static void gpu_encode_sgemv_q4_bias(id<MTLComputeCommandEncoder> enc, QwenModel *m,
                                      id<MTLBuffer> w_buf, id<MTLBuffer> x_buf, id<MTLBuffer> y_buf,
                                      uint32_t in_dim, uint32_t out_dim,
                                      id<MTLBuffer> bias_buf) {
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_q4_fast;
    [enc setComputePipelineState:pso];
    [enc setBuffer:w_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:y_buf offset:0 atIndex:2];
    [enc setBytes:&in_dim length:4 atIndex:3];
    [enc setBytes:&out_dim length:4 atIndex:4];

    uint32_t use_bias = (bias_buf != nil) ? 1 : 0;
    if (bias_buf) {
        [enc setBuffer:bias_buf offset:0 atIndex:5];
    } else {
        [enc setBuffer:y_buf offset:0 atIndex:5];
    }
    [enc setBytes:&use_bias length:4 atIndex:6];

    uint32_t rows_per_tg = 8;
    uint32_t n_tg = (out_dim + rows_per_tg - 1) / rows_per_tg;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static void gpu_encode_sgemv_q4(id<MTLComputeCommandEncoder> enc, QwenModel *m,
                                 id<MTLBuffer> w_buf, id<MTLBuffer> x_buf, id<MTLBuffer> y_buf,
                                 uint32_t in_dim, uint32_t out_dim) {
    gpu_encode_sgemv_q4_bias(enc, m, w_buf, x_buf, y_buf, in_dim, out_dim, nil);
}

static void gpu_encode_sgemv_f32(id<MTLComputeCommandEncoder> enc,
                                  id<MTLBuffer> w_buf, id<MTLBuffer> x_buf, id<MTLBuffer> y_buf,
                                  uint32_t in_dim, uint32_t out_dim) {
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_f32;
    [enc setComputePipelineState:pso];
    [enc setBuffer:w_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:y_buf offset:0 atIndex:2];
    [enc setBytes:&in_dim length:4 atIndex:3];
    [enc setBytes:&out_dim length:4 atIndex:4];
    NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
    if (tpg > out_dim) tpg = out_dim;
    [enc dispatchThreads:MTLSizeMake(out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Fused gate+up+silu: reads x once, computes silu(Wg*x)*Wu*x
static void gpu_encode_fused_ffn(id<MTLComputeCommandEncoder> enc,
                                  id<MTLBuffer> wgate_buf, id<MTLBuffer> wup_buf,
                                  id<MTLBuffer> x_buf, id<MTLBuffer> out_buf,
                                  uint32_t in_dim, uint32_t out_dim) {
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_q4_fused_ffn;
    [enc setComputePipelineState:pso];
    [enc setBuffer:wgate_buf offset:0 atIndex:0];
    [enc setBuffer:wup_buf offset:0 atIndex:1];
    [enc setBuffer:x_buf offset:0 atIndex:2];
    [enc setBuffer:out_buf offset:0 atIndex:3];
    [enc setBytes:&in_dim length:4 atIndex:4];
    [enc setBytes:&out_dim length:4 atIndex:5];

    uint32_t rows_per_tg = 4; // FUSED_ROWS_PER_SIMD(2) * FUSED_SIMD_GROUPS(2)
    uint32_t n_tg = (out_dim + rows_per_tg - 1) / rows_per_tg;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static int qwen_forward_gpu(QwenModel *m, int token) {
    int D = QWEN_DIM, HD = QWEN_HIDDEN;
    int pos = m->pos;
    uint32_t uD = (uint32_t)D, uHD = (uint32_t)HD;
    uint32_t uQD = (uint32_t)QWEN_Q_DIM, uKVD = (uint32_t)QWEN_KV_DIM;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)g_metal.device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)g_metal.queue;

    static id<MTLBuffer> gpu_x = nil, gpu_xb = nil;
    static id<MTLBuffer> gpu_q = nil, gpu_k = nil, gpu_v = nil;
    static id<MTLBuffer> gpu_hb = nil, gpu_hb2 = nil;
    static id<MTLBuffer> gpu_attn_out = nil;
    static id<MTLBuffer> gpu_o_out = nil, gpu_ffn_out = nil;
    static id<MTLBuffer> gpu_logits = nil;
    static id<MTLBuffer> gpu_att = nil;
    static id<MTLBuffer> gpu_result = nil;
    static id<MTLBuffer> gpu_rope_cos = nil, gpu_rope_sin = nil;

    if (!gpu_x) {
        gpu_x    = [dev newBufferWithLength:D * 4 options:MTLResourceStorageModeShared];
        gpu_xb   = [dev newBufferWithLength:D * 4 options:MTLResourceStorageModeShared];
        gpu_q    = [dev newBufferWithLength:QWEN_Q_DIM * 4 options:MTLResourceStorageModeShared];
        gpu_k    = [dev newBufferWithLength:QWEN_KV_DIM * 4 options:MTLResourceStorageModeShared];
        gpu_v    = [dev newBufferWithLength:QWEN_KV_DIM * 4 options:MTLResourceStorageModeShared];
        gpu_hb   = [dev newBufferWithLength:HD * 4 options:MTLResourceStorageModeShared];
        gpu_hb2  = [dev newBufferWithLength:HD * 4 options:MTLResourceStorageModeShared];
        gpu_attn_out = [dev newBufferWithLength:QWEN_Q_DIM * 4 options:MTLResourceStorageModeShared];
        gpu_o_out    = [dev newBufferWithLength:D * 4 options:MTLResourceStorageModeShared];
        gpu_ffn_out  = [dev newBufferWithLength:D * 4 options:MTLResourceStorageModeShared];
        gpu_logits   = [dev newBufferWithLength:QWEN_VOCAB * 4 options:MTLResourceStorageModeShared];
        gpu_att      = [dev newBufferWithLength:QWEN_HEADS * QWEN_MAX_SEQ * 4 options:MTLResourceStorageModeShared];
        gpu_result   = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];

        qwen_rope_init();
        gpu_rope_cos = [dev newBufferWithLength:sizeof(g_rope_cos) options:MTLResourceStorageModeShared];
        gpu_rope_sin = [dev newBufferWithLength:sizeof(g_rope_sin) options:MTLResourceStorageModeShared];
        memcpy([gpu_rope_cos contents], g_rope_cos, sizeof(g_rope_cos));
        memcpy([gpu_rope_sin contents], g_rope_sin, sizeof(g_rope_sin));
    }

    id<MTLComputePipelineState> pso_rms  = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_rms;
    id<MTLComputePipelineState> pso_rope = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_rope;
    id<MTLComputePipelineState> pso_silu = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_silu;
    id<MTLComputePipelineState> pso_add  = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_add;
    id<MTLComputePipelineState> pso_bias = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_bias;
    id<MTLComputePipelineState> pso_embed = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_embed;
    id<MTLComputePipelineState> pso_attn_score = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_attn_score;
    id<MTLComputePipelineState> pso_softmax    = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_softmax;
    id<MTLComputePipelineState> pso_attn_wsum  = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_attn_wsum;
    id<MTLComputePipelineState> pso_argmax     = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_argmax;
    id<MTLComputePipelineState> pso_zero       = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_zero;
    id<MTLComputePipelineState> pso_copy       = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_copy;

    float rms_eps = QWEN_RMS_EPS;
    uint32_t utoken = (uint32_t)token;
    uint32_t seq_len = (uint32_t)(pos + 1);
    float attn_scale = 1.0f / sqrtf((float)QWEN_HEAD_DIM);
    uint32_t un_q = QWEN_HEADS, un_kv = QWEN_KV_HEADS, uhd = QWEN_HEAD_DIM;

    // Encode ALL 24 layers + final into ONE command buffer.
    // Metal guarantees sequential execution of dispatches within a command encoder,
    // so data dependencies (KV cache reads after writes) are satisfied by dispatch order.
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // Embedding
    [enc setComputePipelineState:pso_embed];
    [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_embed offset:0 atIndex:0];
    [enc setBuffer:gpu_x offset:0 atIndex:1];
    [enc setBytes:&utoken length:4 atIndex:2];
    [enc setBytes:&uD length:4 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)D, pso_embed.maxTotalThreadsPerThreadgroup), 1, 1)];

    for (int l = 0; l < QWEN_LAYERS; l++) {
        // RMSNorm attention
        [enc setComputePipelineState:pso_rms];
        [enc setBuffer:gpu_x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_rms_att[l] offset:0 atIndex:1];
        [enc setBuffer:gpu_xb offset:0 atIndex:2];
        [enc setBytes:&uD length:4 atIndex:3];
        [enc setBytes:&rms_eps length:4 atIndex:4];
        { NSUInteger p = 1; while (p < (NSUInteger)D) p <<= 1; if (p > 1024) p = 1024;
          [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(p, 1, 1)]; }

        // QKV with fused bias (saves 3 bias_add dispatches per layer)
        gpu_encode_sgemv_q4_bias(enc, m, (__bridge id<MTLBuffer>)m->gpu_wq[l], gpu_xb, gpu_q, uD, uQD,
                                  (__bridge id<MTLBuffer>)m->gpu_q_bias[l]);
        gpu_encode_sgemv_q4_bias(enc, m, (__bridge id<MTLBuffer>)m->gpu_wk[l], gpu_xb, gpu_k, uD, uKVD,
                                  (__bridge id<MTLBuffer>)m->gpu_k_bias[l]);
        gpu_encode_sgemv_q4_bias(enc, m, (__bridge id<MTLBuffer>)m->gpu_wv[l], gpu_xb, gpu_v, uD, uKVD,
                                  (__bridge id<MTLBuffer>)m->gpu_v_bias[l]);

        // RoPE
        uint32_t rope_offset = (uint32_t)pos * (QWEN_HEAD_DIM / 2);
        [enc setComputePipelineState:pso_rope];
        [enc setBuffer:gpu_q offset:0 atIndex:0];
        [enc setBuffer:gpu_k offset:0 atIndex:1];
        [enc setBuffer:gpu_rope_cos offset:rope_offset * 4 atIndex:2];
        [enc setBuffer:gpu_rope_sin offset:rope_offset * 4 atIndex:3];
        [enc setBytes:&un_q length:4 atIndex:4];
        [enc setBytes:&un_kv length:4 atIndex:5];
        [enc setBytes:&uhd length:4 atIndex:6];
        { uint32_t total = (QWEN_HEADS + QWEN_KV_HEADS) * (QWEN_HEAD_DIM / 2);
          [enc dispatchThreads:MTLSizeMake(total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)total, pso_rope.maxTotalThreadsPerThreadgroup), 1, 1)]; }

        // Store K, V into KV cache
        [enc setComputePipelineState:pso_copy];
        [enc setBuffer:gpu_k offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_kv_cache_k[l] offset:(NSUInteger)pos * QWEN_KV_DIM * 4 atIndex:1];
        [enc setBytes:&uKVD length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(QWEN_KV_DIM, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)QWEN_KV_DIM, pso_copy.maxTotalThreadsPerThreadgroup), 1, 1)];

        [enc setBuffer:gpu_v offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_kv_cache_v[l] offset:(NSUInteger)pos * QWEN_KV_DIM * 4 atIndex:1];
        [enc setBytes:&uKVD length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(QWEN_KV_DIM, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)QWEN_KV_DIM, pso_copy.maxTotalThreadsPerThreadgroup), 1, 1)];

        // Batched attention: all 14 Q heads in 3 dispatches (was 42)
        {
            uint32_t un_q_heads = QWEN_HEADS;
            uint32_t u_gqa = QWEN_GQA_FACTOR;
            uint32_t u_max_seq = QWEN_MAX_SEQ;

            id<MTLComputePipelineState> pso_score_b = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_attn_score_b;
            id<MTLComputePipelineState> pso_soft_b = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_softmax_b;
            id<MTLComputePipelineState> pso_wsum_b = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_attn_wsum_b;

            // 1. Batched attn score: grid (seq_len, n_q_heads)
            [enc setComputePipelineState:pso_score_b];
            [enc setBuffer:gpu_q offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_kv_cache_k[l] offset:0 atIndex:1];
            [enc setBuffer:gpu_att offset:0 atIndex:2];
            [enc setBytes:&uhd length:4 atIndex:3];
            [enc setBytes:&uKVD length:4 atIndex:4];
            [enc setBytes:&un_q_heads length:4 atIndex:5];
            [enc setBytes:&u_gqa length:4 atIndex:6];
            [enc setBytes:&attn_scale length:4 atIndex:7];
            [enc setBytes:&seq_len length:4 atIndex:8];
            [enc setBytes:&u_max_seq length:4 atIndex:9];
            { NSUInteger tpg_x = MIN((NSUInteger)seq_len, (NSUInteger)256);
              NSUInteger tpg_y = MIN((NSUInteger)QWEN_HEADS, (NSUInteger)(pso_score_b.maxTotalThreadsPerThreadgroup / tpg_x));
              if (tpg_y < 1) tpg_y = 1;
              [enc dispatchThreads:MTLSizeMake(seq_len, QWEN_HEADS, 1)
             threadsPerThreadgroup:MTLSizeMake(tpg_x, tpg_y, 1)]; }

            // 2. Batched softmax: one threadgroup per head
            [enc setComputePipelineState:pso_soft_b];
            [enc setBuffer:gpu_att offset:0 atIndex:0];
            [enc setBytes:&seq_len length:4 atIndex:1];
            [enc setBytes:&u_max_seq length:4 atIndex:2];
            [enc setBytes:&un_q_heads length:4 atIndex:3];
            { NSUInteger p = 1; while (p < (NSUInteger)seq_len && p < 1024) p <<= 1;
              [enc dispatchThreadgroups:MTLSizeMake(QWEN_HEADS, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(p, 1, 1)]; }

            // 3. Batched weighted sum: grid (head_dim, n_q_heads)
            [enc setComputePipelineState:pso_wsum_b];
            [enc setBuffer:gpu_att offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_kv_cache_v[l] offset:0 atIndex:1];
            [enc setBuffer:gpu_attn_out offset:0 atIndex:2];
            [enc setBytes:&uhd length:4 atIndex:3];
            [enc setBytes:&uKVD length:4 atIndex:4];
            [enc setBytes:&un_q_heads length:4 atIndex:5];
            [enc setBytes:&u_gqa length:4 atIndex:6];
            [enc setBytes:&seq_len length:4 atIndex:7];
            [enc setBytes:&u_max_seq length:4 atIndex:8];
            { NSUInteger tpg_x = MIN((NSUInteger)QWEN_HEAD_DIM, (NSUInteger)64);
              NSUInteger tpg_y = MIN((NSUInteger)QWEN_HEADS, (NSUInteger)(pso_wsum_b.maxTotalThreadsPerThreadgroup / tpg_x));
              if (tpg_y < 1) tpg_y = 1;
              [enc dispatchThreads:MTLSizeMake(QWEN_HEAD_DIM, QWEN_HEADS, 1)
             threadsPerThreadgroup:MTLSizeMake(tpg_x, tpg_y, 1)]; }
        }

        // O projection + residual
        gpu_encode_sgemv_q4(enc, m, (__bridge id<MTLBuffer>)m->gpu_wo[l], gpu_attn_out, gpu_o_out, uQD, uD);

        [enc setComputePipelineState:pso_add];
        [enc setBuffer:gpu_x offset:0 atIndex:0];
        [enc setBuffer:gpu_o_out offset:0 atIndex:1];
        [enc setBuffer:gpu_x offset:0 atIndex:2];
        [enc setBytes:&uD length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(D, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)D, pso_add.maxTotalThreadsPerThreadgroup), 1, 1)];

        // FFN
        [enc setComputePipelineState:pso_rms];
        [enc setBuffer:gpu_x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_rms_ffn[l] offset:0 atIndex:1];
        [enc setBuffer:gpu_xb offset:0 atIndex:2];
        [enc setBytes:&uD length:4 atIndex:3];
        [enc setBytes:&rms_eps length:4 atIndex:4];
        { NSUInteger p = 1; while (p < (NSUInteger)D) p <<= 1; if (p > 1024) p = 1024;
          [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(p, 1, 1)]; }

        // Fused gate+up+silu: one kernel reads xb, computes silu(Wg*xb)*Wu*xb
        gpu_encode_fused_ffn(enc,
                             (__bridge id<MTLBuffer>)m->gpu_wgate[l],
                             (__bridge id<MTLBuffer>)m->gpu_wup[l],
                             gpu_xb, gpu_hb, uD, uHD);

        gpu_encode_sgemv_q4(enc, m, (__bridge id<MTLBuffer>)m->gpu_wdown[l], gpu_hb, gpu_ffn_out, uHD, uD);

        [enc setComputePipelineState:pso_add];
        [enc setBuffer:gpu_x offset:0 atIndex:0];
        [enc setBuffer:gpu_ffn_out offset:0 atIndex:1];
        [enc setBuffer:gpu_x offset:0 atIndex:2];
        [enc setBytes:&uD length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(D, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)D, pso_add.maxTotalThreadsPerThreadgroup), 1, 1)];
    }

    // Final RMSNorm + LM Head + argmax (still in the SAME command buffer)
    [enc setComputePipelineState:pso_rms];
    [enc setBuffer:gpu_x offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_rms_final offset:0 atIndex:1];
    [enc setBuffer:gpu_xb offset:0 atIndex:2];
    [enc setBytes:&uD length:4 atIndex:3];
    [enc setBytes:&rms_eps length:4 atIndex:4];
    { NSUInteger p = 1; while (p < (NSUInteger)D) p <<= 1; if (p > 1024) p = 1024;
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(p, 1, 1)]; }

    uint32_t uVocab = QWEN_VOCAB;
    gpu_encode_sgemv_f32(enc, (__bridge id<MTLBuffer>)m->gpu_embed, gpu_xb, gpu_logits, uD, uVocab);

    [enc setComputePipelineState:pso_argmax];
    [enc setBuffer:gpu_logits offset:0 atIndex:0];
    [enc setBuffer:gpu_result offset:0 atIndex:1];
    [enc setBytes:&uVocab length:4 atIndex:2];
    { NSUInteger tpg = MIN((NSUInteger)1024, pso_argmax.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)]; }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    m->pos++;

    int *result_ptr = (int*)[gpu_result contents];
    return result_ptr[0];
}

// ── GPU batched prefill: all N prompt tokens in one command buffer ────
// Uses sgemm_q4 (matrix-matrix) instead of sequential sgemv calls.
// Reads each weight matrix once for all N tokens instead of N times.
static int qwen_prefill_gpu(QwenModel *m, const int *tokens, int n_tokens) {
    int D = QWEN_DIM, HD = QWEN_HIDDEN, N = n_tokens;
    uint32_t uD = (uint32_t)D, uHD = (uint32_t)HD;
    uint32_t uQD = (uint32_t)QWEN_Q_DIM, uKVD = (uint32_t)QWEN_KV_DIM;
    uint32_t uN = (uint32_t)N;
    float rms_eps = QWEN_RMS_EPS;
    float attn_scale = 1.0f / sqrtf((float)QWEN_HEAD_DIM);
    uint32_t uhd = QWEN_HEAD_DIM;
    uint32_t un_q = QWEN_HEADS, un_kv = QWEN_KV_HEADS;
    uint32_t u_gqa = QWEN_GQA_FACTOR;
    uint32_t u_max_seq = QWEN_MAX_SEQ;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)g_metal.device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)g_metal.queue;

    // Static batch buffers: allocated once at QWEN_MAX_SEQ size, reused across calls
    static id<MTLBuffer> gpu_xs = nil, gpu_xbs = nil;
    static id<MTLBuffer> gpu_qs = nil, gpu_ks = nil, gpu_vs = nil;
    static id<MTLBuffer> gpu_hbs = nil;
    static id<MTLBuffer> gpu_attn_outs = nil, gpu_o_outs = nil, gpu_ffn_outs = nil;
    static id<MTLBuffer> gpu_att = nil, gpu_logits = nil, gpu_result = nil;
    static id<MTLBuffer> gpu_token_ids = nil;
    static id<MTLBuffer> gpu_rope_cos = nil, gpu_rope_sin = nil;
    static id<MTLBuffer> gpu_xb_last = nil;

    if (!gpu_xs) {
        NSUInteger maxN = QWEN_MAX_SEQ;
        gpu_xs        = [dev newBufferWithLength:maxN * D * 4 options:MTLResourceStorageModeShared];
        gpu_xbs       = [dev newBufferWithLength:maxN * D * 4 options:MTLResourceStorageModeShared];
        gpu_qs        = [dev newBufferWithLength:maxN * QWEN_Q_DIM * 4 options:MTLResourceStorageModeShared];
        gpu_ks        = [dev newBufferWithLength:maxN * QWEN_KV_DIM * 4 options:MTLResourceStorageModeShared];
        gpu_vs        = [dev newBufferWithLength:maxN * QWEN_KV_DIM * 4 options:MTLResourceStorageModeShared];
        gpu_hbs       = [dev newBufferWithLength:maxN * HD * 4 options:MTLResourceStorageModeShared];
        gpu_attn_outs = [dev newBufferWithLength:maxN * QWEN_Q_DIM * 4 options:MTLResourceStorageModeShared];
        gpu_o_outs    = [dev newBufferWithLength:maxN * D * 4 options:MTLResourceStorageModeShared];
        gpu_ffn_outs  = [dev newBufferWithLength:maxN * D * 4 options:MTLResourceStorageModeShared];
        gpu_att       = [dev newBufferWithLength:QWEN_HEADS * QWEN_MAX_SEQ * 4 options:MTLResourceStorageModeShared];
        gpu_logits    = [dev newBufferWithLength:QWEN_VOCAB * 4 options:MTLResourceStorageModeShared];
        gpu_result    = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        gpu_token_ids = [dev newBufferWithLength:maxN * sizeof(int) options:MTLResourceStorageModeShared];
        gpu_xb_last   = [dev newBufferWithLength:D * 4 options:MTLResourceStorageModeShared];

        qwen_rope_init();
        gpu_rope_cos = [dev newBufferWithLength:sizeof(g_rope_cos) options:MTLResourceStorageModeShared];
        gpu_rope_sin = [dev newBufferWithLength:sizeof(g_rope_sin) options:MTLResourceStorageModeShared];
        memcpy([gpu_rope_cos contents], g_rope_cos, sizeof(g_rope_cos));
        memcpy([gpu_rope_sin contents], g_rope_sin, sizeof(g_rope_sin));
    }

    memcpy([gpu_token_ids contents], tokens, (NSUInteger)N * sizeof(int));

    // Pipeline states
    id<MTLComputePipelineState> pso_sgemm_q4      = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_sgemm_q4;
    id<MTLComputePipelineState> pso_sgemm_ffn     = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_sgemm_q4_fused_ffn;
    id<MTLComputePipelineState> pso_rms_b         = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_rms_batched;
    id<MTLComputePipelineState> pso_embed_b       = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_embed_batched;
    id<MTLComputePipelineState> pso_rope_b        = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_rope_batched;
    id<MTLComputePipelineState> pso_add_b         = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_add_batched;
    id<MTLComputePipelineState> pso_copy           = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_copy;
    id<MTLComputePipelineState> pso_rms            = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_rms;
    id<MTLComputePipelineState> pso_argmax         = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_argmax;
    id<MTLComputePipelineState> pso_score_b        = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_attn_score_b;
    id<MTLComputePipelineState> pso_soft_b         = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_softmax_b;
    id<MTLComputePipelineState> pso_wsum_b         = (__bridge id<MTLComputePipelineState>)g_metal.pipeline_attn_wsum_b;

    // Helper: encode sgemm_q4 dispatch
    #define ENCODE_SGEMM_Q4(enc, w_buf, x_buf, y_buf, in_d, out_d, bias_buf, n_tok) do { \
        [enc setComputePipelineState:pso_sgemm_q4]; \
        [enc setBuffer:w_buf offset:0 atIndex:0]; \
        [enc setBuffer:x_buf offset:0 atIndex:1]; \
        [enc setBuffer:y_buf offset:0 atIndex:2]; \
        uint32_t _id = (in_d), _od = (out_d), _ub = ((bias_buf) != nil) ? 1 : 0, _nt = (n_tok); \
        [enc setBytes:&_id length:4 atIndex:3]; \
        [enc setBytes:&_od length:4 atIndex:4]; \
        if (bias_buf) [enc setBuffer:bias_buf offset:0 atIndex:5]; \
        else [enc setBuffer:y_buf offset:0 atIndex:5]; \
        [enc setBytes:&_ub length:4 atIndex:6]; \
        [enc setBytes:&_nt length:4 atIndex:7]; \
        uint32_t _tg_x = (_od + 7) / 8; \
        [enc dispatchThreadgroups:MTLSizeMake(_tg_x, _nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; \
    } while(0)

    // Single command buffer for entire prefill
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // 1. Batched embedding: load all N token embeddings
    [enc setComputePipelineState:pso_embed_b];
    [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_embed offset:0 atIndex:0];
    [enc setBuffer:gpu_xs offset:0 atIndex:1];
    [enc setBuffer:gpu_token_ids offset:0 atIndex:2];
    [enc setBytes:&uD length:4 atIndex:3];
    { NSUInteger tpg_x = MIN((NSUInteger)D, pso_embed_b.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreads:MTLSizeMake(D, N, 1) threadsPerThreadgroup:MTLSizeMake(tpg_x, 1, 1)]; }

    for (int l = 0; l < QWEN_LAYERS; l++) {
        // 2. Batched RMSNorm (attention): N threadgroups, one per token
        [enc setComputePipelineState:pso_rms_b];
        [enc setBuffer:gpu_xs offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_rms_att[l] offset:0 atIndex:1];
        [enc setBuffer:gpu_xbs offset:0 atIndex:2];
        [enc setBytes:&uD length:4 atIndex:3];
        [enc setBytes:&rms_eps length:4 atIndex:4];
        [enc setBytes:&uN length:4 atIndex:5];
        { NSUInteger p = 1; while (p < (NSUInteger)D) p <<= 1; if (p > 1024) p = 1024;
          [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(p, 1, 1)]; }

        // 3. Batched QKV projections with fused bias (3 sgemm_q4 dispatches)
        ENCODE_SGEMM_Q4(enc, (__bridge id<MTLBuffer>)m->gpu_wq[l], gpu_xbs, gpu_qs,
                         uD, uQD, (__bridge id<MTLBuffer>)m->gpu_q_bias[l], uN);
        ENCODE_SGEMM_Q4(enc, (__bridge id<MTLBuffer>)m->gpu_wk[l], gpu_xbs, gpu_ks,
                         uD, uKVD, (__bridge id<MTLBuffer>)m->gpu_k_bias[l], uN);
        ENCODE_SGEMM_Q4(enc, (__bridge id<MTLBuffer>)m->gpu_wv[l], gpu_xbs, gpu_vs,
                         uD, uKVD, (__bridge id<MTLBuffer>)m->gpu_v_bias[l], uN);

        // 4. Batched RoPE: apply to all N tokens' Q and K
        uint32_t base_pos = (uint32_t)m->pos;
        uint32_t q_stride_val = QWEN_Q_DIM;
        uint32_t k_stride_val = QWEN_KV_DIM;
        [enc setComputePipelineState:pso_rope_b];
        [enc setBuffer:gpu_qs offset:0 atIndex:0];
        [enc setBuffer:gpu_ks offset:0 atIndex:1];
        [enc setBuffer:gpu_rope_cos offset:0 atIndex:2];
        [enc setBuffer:gpu_rope_sin offset:0 atIndex:3];
        [enc setBytes:&un_q length:4 atIndex:4];
        [enc setBytes:&un_kv length:4 atIndex:5];
        [enc setBytes:&uhd length:4 atIndex:6];
        [enc setBytes:&base_pos length:4 atIndex:7];
        [enc setBytes:&q_stride_val length:4 atIndex:8];
        [enc setBytes:&k_stride_val length:4 atIndex:9];
        { uint32_t total_pairs = (QWEN_HEADS + QWEN_KV_HEADS) * (QWEN_HEAD_DIM / 2);
          NSUInteger tpg = MIN((NSUInteger)total_pairs, pso_rope_b.maxTotalThreadsPerThreadgroup);
          [enc dispatchThreads:MTLSizeMake(total_pairs, N, 1)
         threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)]; }

        // 5. Store K, V into cache for all N tokens (copy from batched buffers)
        for (int t = 0; t < N; t++) {
            int pos = m->pos + t;
            [enc setComputePipelineState:pso_copy];
            [enc setBuffer:gpu_ks offset:(NSUInteger)t * QWEN_KV_DIM * 4 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_kv_cache_k[l] offset:(NSUInteger)pos * QWEN_KV_DIM * 4 atIndex:1];
            [enc setBytes:&uKVD length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(QWEN_KV_DIM, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)QWEN_KV_DIM, pso_copy.maxTotalThreadsPerThreadgroup), 1, 1)];

            [enc setBuffer:gpu_vs offset:(NSUInteger)t * QWEN_KV_DIM * 4 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_kv_cache_v[l] offset:(NSUInteger)pos * QWEN_KV_DIM * 4 atIndex:1];
            [enc setBytes:&uKVD length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(QWEN_KV_DIM, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)QWEN_KV_DIM, pso_copy.maxTotalThreadsPerThreadgroup), 1, 1)];
        }

        // 6. Per-token causal attention on GPU (each token sees only preceding tokens)
        for (int t = 0; t < N; t++) {
            uint32_t seq_len = (uint32_t)(m->pos + t + 1);

            // Attn score: all heads
            [enc setComputePipelineState:pso_score_b];
            [enc setBuffer:gpu_qs offset:(NSUInteger)t * QWEN_Q_DIM * 4 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_kv_cache_k[l] offset:0 atIndex:1];
            [enc setBuffer:gpu_att offset:0 atIndex:2];
            [enc setBytes:&uhd length:4 atIndex:3];
            [enc setBytes:&uKVD length:4 atIndex:4];
            [enc setBytes:&un_q length:4 atIndex:5];
            [enc setBytes:&u_gqa length:4 atIndex:6];
            [enc setBytes:&attn_scale length:4 atIndex:7];
            [enc setBytes:&seq_len length:4 atIndex:8];
            [enc setBytes:&u_max_seq length:4 atIndex:9];
            { NSUInteger tpg_x = MIN((NSUInteger)seq_len, (NSUInteger)256);
              NSUInteger tpg_y = MIN((NSUInteger)QWEN_HEADS, pso_score_b.maxTotalThreadsPerThreadgroup / tpg_x);
              if (tpg_y < 1) tpg_y = 1;
              [enc dispatchThreads:MTLSizeMake(seq_len, QWEN_HEADS, 1)
             threadsPerThreadgroup:MTLSizeMake(tpg_x, tpg_y, 1)]; }

            // Softmax: one threadgroup per head
            [enc setComputePipelineState:pso_soft_b];
            [enc setBuffer:gpu_att offset:0 atIndex:0];
            [enc setBytes:&seq_len length:4 atIndex:1];
            [enc setBytes:&u_max_seq length:4 atIndex:2];
            [enc setBytes:&un_q length:4 atIndex:3];
            { NSUInteger p = 1; while (p < (NSUInteger)seq_len && p < 1024) p <<= 1;
              [enc dispatchThreadgroups:MTLSizeMake(QWEN_HEADS, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(p, 1, 1)]; }

            // Weighted sum: all heads
            [enc setComputePipelineState:pso_wsum_b];
            [enc setBuffer:gpu_att offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_kv_cache_v[l] offset:0 atIndex:1];
            [enc setBuffer:gpu_attn_outs offset:(NSUInteger)t * QWEN_Q_DIM * 4 atIndex:2];
            [enc setBytes:&uhd length:4 atIndex:3];
            [enc setBytes:&uKVD length:4 atIndex:4];
            [enc setBytes:&un_q length:4 atIndex:5];
            [enc setBytes:&u_gqa length:4 atIndex:6];
            [enc setBytes:&seq_len length:4 atIndex:7];
            [enc setBytes:&u_max_seq length:4 atIndex:8];
            { NSUInteger tpg_x = MIN((NSUInteger)QWEN_HEAD_DIM, (NSUInteger)64);
              NSUInteger tpg_y = MIN((NSUInteger)QWEN_HEADS, pso_wsum_b.maxTotalThreadsPerThreadgroup / tpg_x);
              if (tpg_y < 1) tpg_y = 1;
              [enc dispatchThreads:MTLSizeMake(QWEN_HEAD_DIM, QWEN_HEADS, 1)
             threadsPerThreadgroup:MTLSizeMake(tpg_x, tpg_y, 1)]; }
        }

        // 7. Batched O projection
        ENCODE_SGEMM_Q4(enc, (__bridge id<MTLBuffer>)m->gpu_wo[l], gpu_attn_outs, gpu_o_outs,
                         uQD, uD, nil, uN);

        // 8. Batched residual: xs += o_outs
        uint32_t total_add = uN * uD;
        [enc setComputePipelineState:pso_add_b];
        [enc setBuffer:gpu_xs offset:0 atIndex:0];
        [enc setBuffer:gpu_o_outs offset:0 atIndex:1];
        [enc setBuffer:gpu_xs offset:0 atIndex:2];
        [enc setBytes:&total_add length:4 atIndex:3];
        { NSUInteger tpg = MIN((NSUInteger)total_add, pso_add_b.maxTotalThreadsPerThreadgroup);
          [enc dispatchThreads:MTLSizeMake(total_add, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)]; }

        // 9. Batched RMSNorm (FFN)
        [enc setComputePipelineState:pso_rms_b];
        [enc setBuffer:gpu_xs offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_rms_ffn[l] offset:0 atIndex:1];
        [enc setBuffer:gpu_xbs offset:0 atIndex:2];
        [enc setBytes:&uD length:4 atIndex:3];
        [enc setBytes:&rms_eps length:4 atIndex:4];
        [enc setBytes:&uN length:4 atIndex:5];
        { NSUInteger p = 1; while (p < (NSUInteger)D) p <<= 1; if (p > 1024) p = 1024;
          [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(p, 1, 1)]; }

        // 10. Batched fused Gate+Up+SiLU
        [enc setComputePipelineState:pso_sgemm_ffn];
        [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_wgate[l] offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_wup[l] offset:0 atIndex:1];
        [enc setBuffer:gpu_xbs offset:0 atIndex:2];
        [enc setBuffer:gpu_hbs offset:0 atIndex:3];
        [enc setBytes:&uD length:4 atIndex:4];
        [enc setBytes:&uHD length:4 atIndex:5];
        [enc setBytes:&uN length:4 atIndex:6];
        { uint32_t ffn_tg_x = (uHD + 3) / 4;
          [enc dispatchThreadgroups:MTLSizeMake(ffn_tg_x, N, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; }

        // 11. Batched down projection
        ENCODE_SGEMM_Q4(enc, (__bridge id<MTLBuffer>)m->gpu_wdown[l], gpu_hbs, gpu_ffn_outs,
                         uHD, uD, nil, uN);

        // 12. Batched FFN residual: xs += ffn_outs
        [enc setComputePipelineState:pso_add_b];
        [enc setBuffer:gpu_xs offset:0 atIndex:0];
        [enc setBuffer:gpu_ffn_outs offset:0 atIndex:1];
        [enc setBuffer:gpu_xs offset:0 atIndex:2];
        [enc setBytes:&total_add length:4 atIndex:3];
        { NSUInteger tpg = MIN((NSUInteger)total_add, pso_add_b.maxTotalThreadsPerThreadgroup);
          [enc dispatchThreads:MTLSizeMake(total_add, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)]; }
    }

    // Final: RMSNorm + LM head + argmax on LAST token only
    NSUInteger last_off = (NSUInteger)(N - 1) * D * 4;

    [enc setComputePipelineState:pso_rms];
    [enc setBuffer:gpu_xs offset:last_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)m->gpu_rms_final offset:0 atIndex:1];
    [enc setBuffer:gpu_xb_last offset:0 atIndex:2];
    [enc setBytes:&uD length:4 atIndex:3];
    [enc setBytes:&rms_eps length:4 atIndex:4];
    { NSUInteger p = 1; while (p < (NSUInteger)D) p <<= 1; if (p > 1024) p = 1024;
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(p, 1, 1)]; }

    uint32_t uVocab = QWEN_VOCAB;
    gpu_encode_sgemv_f32(enc, (__bridge id<MTLBuffer>)m->gpu_embed, gpu_xb_last, gpu_logits, uD, uVocab);

    [enc setComputePipelineState:pso_argmax];
    [enc setBuffer:gpu_logits offset:0 atIndex:0];
    [enc setBuffer:gpu_result offset:0 atIndex:1];
    [enc setBytes:&uVocab length:4 atIndex:2];
    { NSUInteger tpg = MIN((NSUInteger)1024, pso_argmax.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)]; }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    #undef ENCODE_SGEMM_Q4

    m->pos += N;
    int *result_ptr = (int*)[gpu_result contents];
    return result_ptr[0];
}

#endif // __OBJC__

// ── Compile all ANE kernels ──────────────────────────────────────────

static void qwen_compile_kernels(QwenModel *m) {
#if USE_ANE_PROJECTIONS
    int D = QWEN_DIM, HD = QWEN_HIDDEN;
    printf("Compiling %d ANE kernels...\n", QWEN_LAYERS * 7 + 1);
    for (int l = 0; l < QWEN_LAYERS; l++) {
        m->k_q[l]    = compile_conv_kernel(m->wq[l],    D, QWEN_Q_DIM,  1);
        m->k_k[l]    = compile_conv_kernel(m->wk[l],    D, QWEN_KV_DIM, 1);
        m->k_v[l]    = compile_conv_kernel(m->wv[l],    D, QWEN_KV_DIM, 1);
        m->k_o[l]    = compile_conv_kernel(m->wo[l],    QWEN_Q_DIM, D,  1);
        m->k_gate[l] = compile_conv_kernel(m->w_gate[l], D, HD,          1);
        m->k_up[l]   = compile_conv_kernel(m->w_up[l],   D, HD,          1);
        m->k_down[l] = compile_conv_kernel(m->w_down[l], HD, D,          1);
        printf("  Layer %d/%d compiled\r", l+1, QWEN_LAYERS);
        fflush(stdout);
    }
    for (int c = 0; c < QWEN_LM_CHUNKS; c++) {
        float *chunk_weights = m->embed + c * QWEN_LM_CHUNK_SIZE * D;
        m->k_lmhead[c] = compile_conv_kernel(chunk_weights, D, QWEN_LM_CHUNK_SIZE, 1);
        if (!m->k_lmhead[c]) {
            printf("  LM head chunk %d FAILED to compile\n", c);
        }
    }
    printf("\nAll kernels compiled.\n");
#else
    printf("CPU-only mode (ANE kernel compilation skipped).\n");
    (void)m;
#endif
}

// Fused ANE compilation: QKV fused + Gate/Up fused + separate O, Down
// Total: 24*(1 QKV + 1 O + 1 FFN_up + 1 Down) = 96 kernels + 16 LM head = 112 (< 119)
static void qwen_compile_kernels_fused(QwenModel *m) {
    int D = QWEN_DIM, HD = QWEN_HIDDEN;
    int total = QWEN_LAYERS * 4 + QWEN_LM_CHUNKS;
    int compiled = 0, failed = 0;
    printf("Compiling %d fused ANE kernels (QKV+FFN_up fused)...\n", total);

    for (int l = 0; l < QWEN_LAYERS; l++) {
        m->k_qkv[l] = compile_qkv_gqa_kernel(
            m->wq[l], m->wk[l], m->wv[l],
            D, QWEN_Q_DIM, QWEN_KV_DIM);
        if (m->k_qkv[l]) compiled++; else { failed++; printf("  Layer %d QKV FAILED\n", l); }

        m->k_o[l] = compile_conv_kernel_fp16io(m->wo[l], QWEN_Q_DIM, D, 1);
        if (m->k_o[l]) compiled++; else { failed++; printf("  Layer %d O FAILED\n", l); }

        m->k_ffn_up[l] = compile_ffn_up_kernel(m->w_gate[l], m->w_up[l], D, HD);
        if (m->k_ffn_up[l]) compiled++; else { failed++; printf("  Layer %d FFN_up FAILED\n", l); }

        m->k_down[l] = compile_conv_kernel_fp16io(m->w_down[l], HD, D, 1);
        if (m->k_down[l]) compiled++; else { failed++; printf("  Layer %d Down FAILED\n", l); }

        printf("  Layer %d/%d compiled (%d/%d ok)\r", l+1, QWEN_LAYERS, compiled, compiled+failed);
        fflush(stdout);
    }

    for (int c = 0; c < QWEN_LM_CHUNKS; c++) {
        float *chunk_w = m->embed + c * QWEN_LM_CHUNK_SIZE * D;
        m->k_lmhead[c] = compile_conv_kernel_fp16io(chunk_w, D, QWEN_LM_CHUNK_SIZE, 1);
        if (m->k_lmhead[c]) compiled++; else { failed++; printf("  LM head chunk %d FAILED\n", c); }
    }

    printf("\nFused ANE: %d/%d compiled, %d failed\n", compiled, total, failed);
    if (failed > 0)
        printf("WARNING: some kernels failed — ANE inference will fall back to CPU for those projections\n");
}

// ── Allocate buffers ─────────────────────────────────────────────────

static void qwen_alloc(QwenModel *m) {
    m->x      = (float*)qwen_calloc(QWEN_DIM, sizeof(float), "x");
    m->xb     = (float*)qwen_calloc(QWEN_DIM, sizeof(float), "xb");
    m->q      = (float*)qwen_calloc(QWEN_Q_DIM, sizeof(float), "q");
    m->k      = (float*)qwen_calloc(QWEN_KV_DIM, sizeof(float), "k");
    m->v      = (float*)qwen_calloc(QWEN_KV_DIM, sizeof(float), "v");
    m->att    = (float*)qwen_calloc(QWEN_HEADS * QWEN_MAX_SEQ, sizeof(float), "att");
    m->hb     = (float*)qwen_calloc(QWEN_HIDDEN, sizeof(float), "hb");
    m->hb2    = (float*)qwen_calloc(QWEN_HIDDEN, sizeof(float), "hb2");
    m->logits = (float*)qwen_calloc(QWEN_VOCAB, sizeof(float), "logits");
    for (int l = 0; l < QWEN_LAYERS; l++) {
        m->kv_cache_k[l] = (float*)qwen_calloc(QWEN_MAX_SEQ * QWEN_KV_DIM, sizeof(float), "kv_cache_k");
        m->kv_cache_v[l] = (float*)qwen_calloc(QWEN_MAX_SEQ * QWEN_KV_DIM, sizeof(float), "kv_cache_v");
    }
    m->pos = 0;
}

static void qwen_reset(QwenModel *m) {
    for (int l = 0; l < QWEN_LAYERS; l++) {
        memset(m->kv_cache_k[l], 0, QWEN_MAX_SEQ * QWEN_KV_DIM * sizeof(float));
        memset(m->kv_cache_v[l], 0, QWEN_MAX_SEQ * QWEN_KV_DIM * sizeof(float));
    }
    m->pos = 0;
}
