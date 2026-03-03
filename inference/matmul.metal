#include <metal_stdlib>
using namespace metal;

// ── Q4_0 block format ────────────────────────────────────────────────
// Block of 32 values: 2 bytes F16 scale + 2 bytes F16 zero + 16 bytes packed uint8
// Each uint8 stores 2 values: low nibble = even index, high nibble = odd index
// Total: 20 bytes per block of 32 weights
#define Q4_BLOCK_SIZE 32
#define Q4_BLOCK_BYTES 20

// ── Q4 Matrix-vector multiply (legacy, 1 thread per row) ────────────
// Kept as fallback for edge cases.
kernel void sgemv_q4(
    device const uint8_t *W  [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float *y          [[buffer(2)]],
    constant uint &in_dim    [[buffer(3)]],
    constant uint &out_dim   [[buffer(4)]],
    uint gid                 [[thread_position_in_grid]])
{
    if (gid >= out_dim) return;

    uint n_blocks = in_dim / Q4_BLOCK_SIZE;
    uint row_bytes = n_blocks * Q4_BLOCK_BYTES;
    device const uint8_t *row = W + uint64_t(gid) * row_bytes;

    float sum = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        device const uint8_t *block = row + b * Q4_BLOCK_BYTES;
        half scale_h, zero_h;
        scale_h = *reinterpret_cast<device const half*>(block);
        zero_h  = *reinterpret_cast<device const half*>(block + 2);
        float scale = float(scale_h);
        float zero  = float(zero_h);

        device const uint8_t *packed = block + 4;
        uint base = b * Q4_BLOCK_SIZE;

        for (uint i = 0; i < 16; i++) {
            uint8_t byte = packed[i];
            float w0 = float(byte & 0xF) * scale + zero;
            float w1 = float(byte >> 4)  * scale + zero;
            sum += w0 * x[base + i * 2];
            sum += w1 * x[base + i * 2 + 1];
        }
    }
    y[gid] = sum;
}

// ── Q4 SIMD-optimized matrix-vector multiply ─────────────────────────
// MLX-style cooperative SIMD kernel: 2 SIMD groups per threadgroup,
// each SIMD group handles ROWS_PER_SIMD output rows cooperatively.
// 32 threads in a SIMD group split the K (input) dimension, then
// reduce via simd_sum(). No threadgroup memory needed.
//
// Threadgroup layout: 64 threads = 2 SIMD groups of 32
// Grid: (ceil(out_dim / ROWS_PER_TG), 1, 1) threadgroups
//
// Optional bias: if bias pointer is non-null (use_bias != 0),
// y[r] = dot(W[r], x) + bias[r]
#define ROWS_PER_SIMD 4
#define SIMD_GROUPS   2
#define ROWS_PER_TG   (ROWS_PER_SIMD * SIMD_GROUPS)

kernel void sgemv_q4_fast(
    device const uint8_t *W     [[buffer(0)]],
    device const float   *x     [[buffer(1)]],
    device       float   *y     [[buffer(2)]],
    constant     uint    &in_dim   [[buffer(3)]],
    constant     uint    &out_dim  [[buffer(4)]],
    device const float   *bias  [[buffer(5)]],
    constant     uint    &use_bias [[buffer(6)]],
    uint  tgid      [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    uint base_row = tgid * ROWS_PER_TG + simd_gid * ROWS_PER_SIMD;
    if (base_row >= out_dim) return;

    uint n_blocks = in_dim / Q4_BLOCK_SIZE;
    uint row_bytes = n_blocks * Q4_BLOCK_BYTES;

    uint rows_this = min((uint)ROWS_PER_SIMD, out_dim - base_row);

    float accum[ROWS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};
    float zero_accum[ROWS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each of 32 SIMD lanes processes a stripe of blocks.
    // Lane i processes blocks i, i+32, i+64, ...
    for (uint b = simd_lid; b < n_blocks; b += 32) {
        uint k_base = b * Q4_BLOCK_SIZE;

        // Load input vector segment for this block (32 floats)
        float xv[Q4_BLOCK_SIZE];
        for (uint j = 0; j < 16; j++) {
            xv[j * 2]     = x[k_base + j * 2];
            xv[j * 2 + 1] = x[k_base + j * 2 + 1];
        }

        for (uint r = 0; r < rows_this; r++) {
            device const uint8_t *block =
                W + uint64_t(base_row + r) * row_bytes + uint64_t(b) * Q4_BLOCK_BYTES;

            half scale_h = *reinterpret_cast<device const half*>(block);
            half zero_h  = *reinterpret_cast<device const half*>(block + 2);
            float scale = float(scale_h);
            float zero  = float(zero_h);

            device const uint8_t *packed = block + 4;

            float dot = 0.0f;
            float xsum = 0.0f;
            for (uint j = 0; j < 16; j++) {
                uint8_t byte = packed[j];
                float w0 = float(byte & 0xF);
                float w1 = float(byte >> 4);
                dot += w0 * xv[j * 2] + w1 * xv[j * 2 + 1];
                xsum += xv[j * 2] + xv[j * 2 + 1];
            }
            accum[r] += dot * scale;
            zero_accum[r] += xsum * zero;
        }
    }

    // SIMD reduction across 32 lanes
    for (uint r = 0; r < rows_this; r++) {
        float result = simd_sum(accum[r]) + simd_sum(zero_accum[r]);
        if (simd_lid == 0) {
            if (use_bias != 0) {
                result += bias[base_row + r];
            }
            y[base_row + r] = result;
        }
    }
}

// ── Fused Gate+Up+SiLU: reads x once, computes gate=silu(Wg*x)*Wu*x ──
// Combines two Q4 matvecs + silu_mul into one kernel.
// W_gate and W_up have the same dimensions [out_dim, in_dim].
// Output: gate[r] = silu(dot(W_gate[r], x)) * dot(W_up[r], x)
#define FUSED_ROWS_PER_SIMD 2
#define FUSED_SIMD_GROUPS   2
#define FUSED_ROWS_PER_TG   (FUSED_ROWS_PER_SIMD * FUSED_SIMD_GROUPS)

kernel void sgemv_q4_fused_ffn(
    device const uint8_t *W_gate [[buffer(0)]],
    device const uint8_t *W_up   [[buffer(1)]],
    device const float   *x      [[buffer(2)]],
    device       float   *out    [[buffer(3)]],
    constant     uint    &in_dim    [[buffer(4)]],
    constant     uint    &out_dim   [[buffer(5)]],
    uint  tgid      [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    uint base_row = tgid * FUSED_ROWS_PER_TG + simd_gid * FUSED_ROWS_PER_SIMD;
    if (base_row >= out_dim) return;

    uint n_blocks = in_dim / Q4_BLOCK_SIZE;
    uint row_bytes = n_blocks * Q4_BLOCK_BYTES;

    uint rows_this = min((uint)FUSED_ROWS_PER_SIMD, out_dim - base_row);

    float gate_acc[FUSED_ROWS_PER_SIMD] = {0.0f, 0.0f};
    float gate_zacc[FUSED_ROWS_PER_SIMD] = {0.0f, 0.0f};
    float up_acc[FUSED_ROWS_PER_SIMD] = {0.0f, 0.0f};
    float up_zacc[FUSED_ROWS_PER_SIMD] = {0.0f, 0.0f};

    for (uint b = simd_lid; b < n_blocks; b += 32) {
        uint k_base = b * Q4_BLOCK_SIZE;

        float xv[Q4_BLOCK_SIZE];
        for (uint j = 0; j < 16; j++) {
            xv[j * 2]     = x[k_base + j * 2];
            xv[j * 2 + 1] = x[k_base + j * 2 + 1];
        }

        for (uint r = 0; r < rows_this; r++) {
            uint64_t row_off = uint64_t(base_row + r) * row_bytes + uint64_t(b) * Q4_BLOCK_BYTES;

            // Gate weight block
            device const uint8_t *g_block = W_gate + row_off;
            float g_scale = float(*reinterpret_cast<device const half*>(g_block));
            float g_zero  = float(*reinterpret_cast<device const half*>(g_block + 2));
            device const uint8_t *g_packed = g_block + 4;

            // Up weight block
            device const uint8_t *u_block = W_up + row_off;
            float u_scale = float(*reinterpret_cast<device const half*>(u_block));
            float u_zero  = float(*reinterpret_cast<device const half*>(u_block + 2));
            device const uint8_t *u_packed = u_block + 4;

            float g_dot = 0.0f, g_xsum = 0.0f;
            float u_dot = 0.0f, u_xsum = 0.0f;
            for (uint j = 0; j < 16; j++) {
                float x0 = xv[j * 2];
                float x1 = xv[j * 2 + 1];
                float xs = x0 + x1;

                uint8_t gb = g_packed[j];
                g_dot += float(gb & 0xF) * x0 + float(gb >> 4) * x1;
                g_xsum += xs;

                uint8_t ub = u_packed[j];
                u_dot += float(ub & 0xF) * x0 + float(ub >> 4) * x1;
                u_xsum += xs;
            }
            gate_acc[r] += g_dot * g_scale;
            gate_zacc[r] += g_xsum * g_zero;
            up_acc[r] += u_dot * u_scale;
            up_zacc[r] += u_xsum * u_zero;
        }
    }

    for (uint r = 0; r < rows_this; r++) {
        float g = simd_sum(gate_acc[r]) + simd_sum(gate_zacc[r]);
        float u = simd_sum(up_acc[r]) + simd_sum(up_zacc[r]);
        if (simd_lid == 0) {
            float s = g / (1.0f + exp(-g));
            out[base_row + r] = s * u;
        }
    }
}

// ── Q4 batched matrix-matrix multiply (SGEMM) for prefill ────────────
// Y[t, r] = sum_k(dequant(W[r, k]) * X[t, k])  for t in [0, n_tokens), r in [0, out_dim)
// Grid: (ceil(out_dim / GEMM_TILE_M), n_tokens, 1)
// Each threadgroup: 2 SIMD groups, each handles GEMM_TILE_M/2 output rows for one token.
#define GEMM_TILE_M 8
#define GEMM_SIMD_GROUPS 2
#define GEMM_ROWS_PER_SIMD (GEMM_TILE_M / GEMM_SIMD_GROUPS)

kernel void sgemm_q4(
    device const uint8_t *W       [[buffer(0)]],
    device const float   *X       [[buffer(1)]],
    device       float   *Y       [[buffer(2)]],
    constant     uint    &in_dim  [[buffer(3)]],
    constant     uint    &out_dim [[buffer(4)]],
    device const float   *bias    [[buffer(5)]],
    constant     uint    &use_bias [[buffer(6)]],
    constant     uint    &n_tokens [[buffer(7)]],
    uint2 tgid      [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    uint base_row = tgid.x * GEMM_TILE_M + simd_gid * GEMM_ROWS_PER_SIMD;
    uint t = tgid.y;
    if (base_row >= out_dim || t >= n_tokens) return;

    uint n_blocks = in_dim / Q4_BLOCK_SIZE;
    uint row_bytes = n_blocks * Q4_BLOCK_BYTES;
    uint rows_this = min((uint)GEMM_ROWS_PER_SIMD, out_dim - base_row);

    device const float *xt = X + uint64_t(t) * in_dim;

    float accum[GEMM_ROWS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};
    float zero_accum[GEMM_ROWS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = simd_lid; b < n_blocks; b += 32) {
        uint k_base = b * Q4_BLOCK_SIZE;

        float xv[Q4_BLOCK_SIZE];
        for (uint j = 0; j < 16; j++) {
            xv[j * 2]     = xt[k_base + j * 2];
            xv[j * 2 + 1] = xt[k_base + j * 2 + 1];
        }

        for (uint r = 0; r < rows_this; r++) {
            device const uint8_t *block =
                W + uint64_t(base_row + r) * row_bytes + uint64_t(b) * Q4_BLOCK_BYTES;

            float scale = float(*reinterpret_cast<device const half*>(block));
            float zero  = float(*reinterpret_cast<device const half*>(block + 2));
            device const uint8_t *packed = block + 4;

            float dot = 0.0f;
            float xsum = 0.0f;
            for (uint j = 0; j < 16; j++) {
                uint8_t byte = packed[j];
                dot += float(byte & 0xF) * xv[j * 2] + float(byte >> 4) * xv[j * 2 + 1];
                xsum += xv[j * 2] + xv[j * 2 + 1];
            }
            accum[r] += dot * scale;
            zero_accum[r] += xsum * zero;
        }
    }

    for (uint r = 0; r < rows_this; r++) {
        float result = simd_sum(accum[r]) + simd_sum(zero_accum[r]);
        if (simd_lid == 0) {
            if (use_bias != 0)
                result += bias[base_row + r];
            Y[uint64_t(t) * out_dim + base_row + r] = result;
        }
    }
}

// ── Q4 batched fused Gate+Up+SiLU (SGEMM variant) ───────────────────
// out[t, r] = silu(Wg[r] . X[t]) * Wu[r] . X[t]  for all t and r
// Grid: (ceil(out_dim / GEMM_FFN_TILE_M), n_tokens, 1)
#define GEMM_FFN_TILE_M 4
#define GEMM_FFN_SIMD_GROUPS 2
#define GEMM_FFN_ROWS_PER_SIMD (GEMM_FFN_TILE_M / GEMM_FFN_SIMD_GROUPS)

kernel void sgemm_q4_fused_ffn(
    device const uint8_t *W_gate  [[buffer(0)]],
    device const uint8_t *W_up    [[buffer(1)]],
    device const float   *X       [[buffer(2)]],
    device       float   *out     [[buffer(3)]],
    constant     uint    &in_dim  [[buffer(4)]],
    constant     uint    &out_dim [[buffer(5)]],
    constant     uint    &n_tokens [[buffer(6)]],
    uint2 tgid      [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    uint base_row = tgid.x * GEMM_FFN_TILE_M + simd_gid * GEMM_FFN_ROWS_PER_SIMD;
    uint t = tgid.y;
    if (base_row >= out_dim || t >= n_tokens) return;

    uint n_blocks = in_dim / Q4_BLOCK_SIZE;
    uint row_bytes = n_blocks * Q4_BLOCK_BYTES;
    uint rows_this = min((uint)GEMM_FFN_ROWS_PER_SIMD, out_dim - base_row);

    device const float *xt = X + uint64_t(t) * in_dim;

    float gate_acc[GEMM_FFN_ROWS_PER_SIMD] = {0.0f, 0.0f};
    float gate_zacc[GEMM_FFN_ROWS_PER_SIMD] = {0.0f, 0.0f};
    float up_acc[GEMM_FFN_ROWS_PER_SIMD] = {0.0f, 0.0f};
    float up_zacc[GEMM_FFN_ROWS_PER_SIMD] = {0.0f, 0.0f};

    for (uint b = simd_lid; b < n_blocks; b += 32) {
        uint k_base = b * Q4_BLOCK_SIZE;

        float xv[Q4_BLOCK_SIZE];
        for (uint j = 0; j < 16; j++) {
            xv[j * 2]     = xt[k_base + j * 2];
            xv[j * 2 + 1] = xt[k_base + j * 2 + 1];
        }

        for (uint r = 0; r < rows_this; r++) {
            uint64_t row_off = uint64_t(base_row + r) * row_bytes + uint64_t(b) * Q4_BLOCK_BYTES;

            device const uint8_t *g_block = W_gate + row_off;
            float g_scale = float(*reinterpret_cast<device const half*>(g_block));
            float g_zero  = float(*reinterpret_cast<device const half*>(g_block + 2));
            device const uint8_t *g_packed = g_block + 4;

            device const uint8_t *u_block = W_up + row_off;
            float u_scale = float(*reinterpret_cast<device const half*>(u_block));
            float u_zero  = float(*reinterpret_cast<device const half*>(u_block + 2));
            device const uint8_t *u_packed = u_block + 4;

            float g_dot = 0.0f, g_xsum = 0.0f;
            float u_dot = 0.0f, u_xsum = 0.0f;
            for (uint j = 0; j < 16; j++) {
                float x0 = xv[j * 2];
                float x1 = xv[j * 2 + 1];
                float xs = x0 + x1;

                uint8_t gb = g_packed[j];
                g_dot += float(gb & 0xF) * x0 + float(gb >> 4) * x1;
                g_xsum += xs;

                uint8_t ub = u_packed[j];
                u_dot += float(ub & 0xF) * x0 + float(ub >> 4) * x1;
                u_xsum += xs;
            }
            gate_acc[r] += g_dot * g_scale;
            gate_zacc[r] += g_xsum * g_zero;
            up_acc[r] += u_dot * u_scale;
            up_zacc[r] += u_xsum * u_zero;
        }
    }

    for (uint r = 0; r < rows_this; r++) {
        float g = simd_sum(gate_acc[r]) + simd_sum(gate_zacc[r]);
        float u = simd_sum(up_acc[r]) + simd_sum(up_zacc[r]);
        if (simd_lid == 0) {
            float s = g / (1.0f + exp(-g));
            out[uint64_t(t) * out_dim + base_row + r] = s * u;
        }
    }
}

// ── Batched RMSNorm (N tokens) ──────────────────────────────────────
// x[t*dim .. (t+1)*dim-1] → out[t*dim .. (t+1)*dim-1]
// Grid: (n_tokens, 1, 1) threadgroups, each normalizes one token.
kernel void rms_norm_batched(
    device const float *x     [[buffer(0)]],
    device const float *w     [[buffer(1)]],
    device float *out         [[buffer(2)]],
    constant uint &dim        [[buffer(3)]],
    constant float &eps       [[buffer(4)]],
    constant uint &n_tokens   [[buffer(5)]],
    uint tgid                 [[threadgroup_position_in_grid]],
    uint tid                  [[thread_index_in_threadgroup]],
    uint tpg                  [[threads_per_threadgroup]])
{
    if (tgid >= n_tokens) return;

    device const float *xi = x + uint64_t(tgid) * dim;
    device float *oi = out + uint64_t(tgid) * dim;

    threadgroup float partial[1024];

    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += tpg)
        local_sum += xi[i] * xi[i];
    partial[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(partial[0] / float(dim) + eps);

    for (uint i = tid; i < dim; i += tpg)
        oi[i] = xi[i] * rms_inv * w[i];
}

// ── Batched embedding lookup (N tokens) ─────────────────────────────
// Grid: (dim, n_tokens, 1). Each thread copies one element.
kernel void embed_lookup_batched(
    device const float *embed    [[buffer(0)]],
    device float *out            [[buffer(1)]],
    device const uint *token_ids [[buffer(2)]],
    constant uint &dim           [[buffer(3)]],
    uint2 gid                    [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint t = gid.y;
    if (i >= dim) return;
    out[uint64_t(t) * dim + i] = embed[uint64_t(token_ids[t]) * dim + i];
}

// ── Batched RoPE (N tokens) ─────────────────────────────────────────
// Applies RoPE to Q[t] and K[t] for each token t at position base_pos+t.
// Grid: total_pairs per token * n_tokens
kernel void rope_apply_batched(
    device float *q             [[buffer(0)]],
    device float *k             [[buffer(1)]],
    device const float *cos_tbl [[buffer(2)]],
    device const float *sin_tbl [[buffer(3)]],
    constant uint &n_q_heads    [[buffer(4)]],
    constant uint &n_kv_heads   [[buffer(5)]],
    constant uint &head_dim     [[buffer(6)]],
    constant uint &base_pos     [[buffer(7)]],
    constant uint &q_stride     [[buffer(8)]],
    constant uint &k_stride     [[buffer(9)]],
    uint2 gid                   [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint t = gid.y;

    uint half_dim = head_dim / 2;
    uint total_pairs = (n_q_heads + n_kv_heads) * half_dim;
    if (pair_idx >= total_pairs) return;

    uint head_pair = pair_idx / half_dim;
    uint i = pair_idx % half_dim;
    uint pos = base_pos + t;

    device float *vec;
    if (head_pair < n_q_heads)
        vec = q + uint64_t(t) * q_stride + head_pair * head_dim;
    else
        vec = k + uint64_t(t) * k_stride + (head_pair - n_q_heads) * head_dim;

    uint cos_off = pos * half_dim;
    float f = vec[i];
    float s = vec[i + half_dim];
    float c = cos_tbl[cos_off + i];
    float sv = sin_tbl[cos_off + i];
    vec[i]            = f * c - s * sv;
    vec[i + half_dim] = s * c + f * sv;
}

// ── Batched vec_add: out[i] = a[i] + b[i] for N*dim elements ────────
kernel void vec_add_batched(
    device const float *a  [[buffer(0)]],
    device const float *b  [[buffer(1)]],
    device float *out      [[buffer(2)]],
    constant uint &total_n [[buffer(3)]],
    uint gid               [[thread_position_in_grid]])
{
    if (gid >= total_n) return;
    out[gid] = a[gid] + b[gid];
}

// ── F16 matrix-vector multiply ───────────────────────────────────────
kernel void sgemv_f16(
    device const half *W      [[buffer(0)]],
    device const float *x     [[buffer(1)]],
    device float *y           [[buffer(2)]],
    constant uint &in_dim     [[buffer(3)]],
    constant uint &out_dim    [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]])
{
    if (gid >= out_dim) return;

    device const half *row = W + uint64_t(gid) * in_dim;
    float sum = 0.0f;

    uint i = 0;
    for (; i + 7 < in_dim; i += 8) {
        sum += float(row[i])     * x[i];
        sum += float(row[i + 1]) * x[i + 1];
        sum += float(row[i + 2]) * x[i + 2];
        sum += float(row[i + 3]) * x[i + 3];
        sum += float(row[i + 4]) * x[i + 4];
        sum += float(row[i + 5]) * x[i + 5];
        sum += float(row[i + 6]) * x[i + 6];
        sum += float(row[i + 7]) * x[i + 7];
    }
    for (; i < in_dim; i++)
        sum += float(row[i]) * x[i];

    y[gid] = sum;
}

// ── F32 matrix-vector multiply ───────────────────────────────────────
kernel void sgemv_f32(
    device const float *W     [[buffer(0)]],
    device const float *x     [[buffer(1)]],
    device float *y           [[buffer(2)]],
    constant uint &in_dim     [[buffer(3)]],
    constant uint &out_dim    [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]])
{
    if (gid >= out_dim) return;

    device const float *row = W + uint64_t(gid) * in_dim;
    float sum = 0.0f;

    uint i = 0;
    for (; i + 7 < in_dim; i += 8) {
        sum += row[i]     * x[i];
        sum += row[i + 1] * x[i + 1];
        sum += row[i + 2] * x[i + 2];
        sum += row[i + 3] * x[i + 3];
        sum += row[i + 4] * x[i + 4];
        sum += row[i + 5] * x[i + 5];
        sum += row[i + 6] * x[i + 6];
        sum += row[i + 7] * x[i + 7];
    }
    for (; i < in_dim; i++)
        sum += row[i] * x[i];

    y[gid] = sum;
}

// ── RMS Normalization ────────────────────────────────────────────────
// out[i] = x[i] * w[i] / sqrt(mean(x^2) + eps)
// Two-pass: first compute sum of squares (reduction), then normalize.
// Single threadgroup processes the entire vector.
kernel void rms_norm(
    device const float *x     [[buffer(0)]],
    device const float *w     [[buffer(1)]],
    device float *out         [[buffer(2)]],
    constant uint &dim        [[buffer(3)]],
    constant float &eps       [[buffer(4)]],
    uint tid                  [[thread_index_in_threadgroup]],
    uint tpg                  [[threads_per_threadgroup]])
{
    threadgroup float partial[1024];

    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += tpg)
        local_sum += x[i] * x[i];
    partial[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(partial[0] / float(dim) + eps);

    for (uint i = tid; i < dim; i += tpg)
        out[i] = x[i] * rms_inv * w[i];
}

// ── RoPE (Rotary Position Embedding) ─────────────────────────────────
// Applies RoPE to Q and K vectors in-place.
// cos_sin is precomputed: [half_dim] cos values followed by [half_dim] sin values.
kernel void rope_apply(
    device float *q            [[buffer(0)]],
    device float *k            [[buffer(1)]],
    device const float *cos_v  [[buffer(2)]],
    device const float *sin_v  [[buffer(3)]],
    constant uint &n_q_heads   [[buffer(4)]],
    constant uint &n_kv_heads  [[buffer(5)]],
    constant uint &head_dim    [[buffer(6)]],
    uint gid                   [[thread_position_in_grid]])
{
    uint half_dim = head_dim / 2;
    uint total_pairs = (n_q_heads + n_kv_heads) * half_dim;
    if (gid >= total_pairs) return;

    uint head_pair = gid / half_dim;
    uint i = gid % half_dim;

    device float *vec;
    if (head_pair < n_q_heads) {
        vec = q + head_pair * head_dim;
    } else {
        vec = k + (head_pair - n_q_heads) * head_dim;
    }

    float f = vec[i];
    float s = vec[i + half_dim];
    float c = cos_v[i];
    float sv = sin_v[i];
    vec[i]            = f * c - s * sv;
    vec[i + half_dim] = s * c + f * sv;
}

// ── SiLU activation + element-wise multiply ──────────────────────────
// gate[i] = silu(gate[i]) * up[i]
// silu(x) = x / (1 + exp(-x))
kernel void silu_mul(
    device float *gate     [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant uint &n       [[buffer(2)]],
    uint gid               [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = gate[gid];
    float s = x / (1.0f + exp(-x));
    gate[gid] = s * up[gid];
}

// ── Vector add (residual connection) ─────────────────────────────────
// out[i] = a[i] + b[i]
kernel void vec_add(
    device const float *a  [[buffer(0)]],
    device const float *b  [[buffer(1)]],
    device float *out      [[buffer(2)]],
    constant uint &n       [[buffer(3)]],
    uint gid               [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = a[gid] + b[gid];
}

// ── Bias add ─────────────────────────────────────────────────────────
// x[i] += bias[i]
kernel void bias_add(
    device float *x            [[buffer(0)]],
    device const float *bias   [[buffer(1)]],
    constant uint &n           [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] += bias[gid];
}

// ── Embedding lookup ─────────────────────────────────────────────────
// out[i] = embed[token_id * dim + i]
kernel void embed_lookup(
    device const float *embed  [[buffer(0)]],
    device float *out          [[buffer(1)]],
    constant uint &token_id    [[buffer(2)]],
    constant uint &dim         [[buffer(3)]],
    uint gid                   [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    out[gid] = embed[uint64_t(token_id) * dim + gid];
}

// ── Attention score: Q @ K^T for one head (legacy) ──────────────────
kernel void attn_score(
    device const float *qh         [[buffer(0)]],
    device const float *kv_cache_k [[buffer(1)]],
    device float *att              [[buffer(2)]],
    constant uint &head_dim        [[buffer(3)]],
    constant uint &kv_dim          [[buffer(4)]],
    constant uint &kv_head_offset  [[buffer(5)]],
    constant float &scale          [[buffer(6)]],
    constant uint &seq_len         [[buffer(7)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= seq_len) return;

    device const float *kt = kv_cache_k + uint64_t(gid) * kv_dim + kv_head_offset;
    float dot = 0.0f;
    for (uint i = 0; i < head_dim; i++)
        dot += qh[i] * kt[i];
    att[gid] = dot * scale;
}

// ── Batched attention score: all Q heads in one dispatch ─────────────
// Grid: (seq_len, n_q_heads, 1). Each thread computes one score for one head.
// GQA: maps Q head h to KV head h/gqa_factor.
kernel void attn_score_batched(
    device const float *q            [[buffer(0)]],
    device const float *kv_cache_k   [[buffer(1)]],
    device float *att                [[buffer(2)]],
    constant uint &head_dim          [[buffer(3)]],
    constant uint &kv_dim            [[buffer(4)]],
    constant uint &n_q_heads         [[buffer(5)]],
    constant uint &gqa_factor        [[buffer(6)]],
    constant float &scale            [[buffer(7)]],
    constant uint &seq_len           [[buffer(8)]],
    constant uint &max_seq           [[buffer(9)]],
    uint2 gid                        [[thread_position_in_grid]])
{
    uint t = gid.x;
    uint h = gid.y;
    if (t >= seq_len || h >= n_q_heads) return;

    uint kv_h = h / gqa_factor;
    device const float *qh = q + h * head_dim;
    device const float *kt = kv_cache_k + uint64_t(t) * kv_dim + kv_h * head_dim;

    float dot = 0.0f;
    for (uint i = 0; i < head_dim; i++)
        dot += qh[i] * kt[i];

    att[h * max_seq + t] = dot * scale;
}

// ── Batched softmax: all heads in one dispatch ───────────────────────
// One threadgroup per head. tid reduces over seq_len dimension.
kernel void softmax_batched(
    device float *att            [[buffer(0)]],
    constant uint &seq_len       [[buffer(1)]],
    constant uint &max_seq       [[buffer(2)]],
    constant uint &n_q_heads     [[buffer(3)]],
    uint tgid                    [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint tpg                     [[threads_per_threadgroup]])
{
    uint h = tgid;
    if (h >= n_q_heads) return;

    device float *head_att = att + h * max_seq;
    threadgroup float shared[1024];

    float local_max = -1e30f;
    for (uint i = tid; i < seq_len; i += tpg)
        local_max = max(local_max, head_att[i]);
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < seq_len; i += tpg) {
        float e = exp(head_att[i] - max_val);
        head_att[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];

    for (uint i = tid; i < seq_len; i += tpg)
        head_att[i] *= inv_sum;
}

// ── Batched attention weighted sum: all heads in one dispatch ────────
// Grid: (head_dim, n_q_heads, 1). Each thread computes one output dim for one head.
kernel void attn_wsum_batched(
    device const float *att          [[buffer(0)]],
    device const float *kv_cache_v   [[buffer(1)]],
    device float *out                [[buffer(2)]],
    constant uint &head_dim          [[buffer(3)]],
    constant uint &kv_dim            [[buffer(4)]],
    constant uint &n_q_heads         [[buffer(5)]],
    constant uint &gqa_factor        [[buffer(6)]],
    constant uint &seq_len           [[buffer(7)]],
    constant uint &max_seq           [[buffer(8)]],
    uint2 gid                        [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint h = gid.y;
    if (d >= head_dim || h >= n_q_heads) return;

    uint kv_h = h / gqa_factor;
    device const float *head_att = att + h * max_seq;

    float sum = 0.0f;
    for (uint t = 0; t < seq_len; t++) {
        float a = head_att[t];
        float v = kv_cache_v[uint64_t(t) * kv_dim + kv_h * head_dim + d];
        sum += a * v;
    }
    out[h * head_dim + d] = sum;
}

// ── Softmax (legacy, single head) ───────────────────────────────────
kernel void softmax_inplace(
    device float *att       [[buffer(0)]],
    constant uint &seq_len  [[buffer(1)]],
    uint tid                [[thread_index_in_threadgroup]],
    uint tpg                [[threads_per_threadgroup]])
{
    threadgroup float shared[1024];

    float local_max = -1e30f;
    for (uint i = tid; i < seq_len; i += tpg)
        local_max = max(local_max, att[i]);
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < seq_len; i += tpg) {
        float e = exp(att[i] - max_val);
        att[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];

    for (uint i = tid; i < seq_len; i += tpg)
        att[i] *= inv_sum;
}

// ── Attention weighted sum (legacy, single head) ─────────────────────
kernel void attn_weighted_sum(
    device const float *att        [[buffer(0)]],
    device const float *kv_cache_v [[buffer(1)]],
    device float *out              [[buffer(2)]],
    constant uint &head_dim        [[buffer(3)]],
    constant uint &kv_dim          [[buffer(4)]],
    constant uint &kv_head_offset  [[buffer(5)]],
    constant uint &seq_len         [[buffer(6)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= head_dim) return;

    float sum = 0.0f;
    for (uint t = 0; t < seq_len; t++) {
        float a = att[t];
        float v = kv_cache_v[uint64_t(t) * kv_dim + kv_head_offset + gid];
        sum += a * v;
    }
    out[gid] = sum;
}

// ── Argmax ───────────────────────────────────────────────────────────
// Finds argmax of logits[0..n-1], writes to result[0].
// Single threadgroup.
kernel void argmax_kernel(
    device const float *logits [[buffer(0)]],
    device int *result         [[buffer(1)]],
    constant uint &n           [[buffer(2)]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tpg                   [[threads_per_threadgroup]])
{
    threadgroup float shared_val[1024];
    threadgroup int shared_idx[1024];

    float local_max = -1e30f;
    int local_idx = 0;
    for (uint i = tid; i < n; i += tpg) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = int(i);
        }
    }
    shared_val[tid] = local_max;
    shared_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s && shared_val[tid + s] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + s];
            shared_idx[tid] = shared_idx[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) result[0] = shared_idx[0];
}

// ── Copy kernel ──────────────────────────────────────────────────────
// dst[i] = src[i] for i in [0, n)
kernel void vec_copy(
    device const float *src [[buffer(0)]],
    device float *dst       [[buffer(1)]],
    constant uint &n        [[buffer(2)]],
    uint gid                [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = src[gid];
}

// ── Zero-fill ────────────────────────────────────────────────────────
kernel void vec_zero(
    device float *dst  [[buffer(0)]],
    constant uint &n   [[buffer(1)]],
    uint gid           [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = 0.0f;
}
