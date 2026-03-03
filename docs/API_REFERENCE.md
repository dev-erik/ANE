# ANE Training -- API Reference

Complete function index for all public functions, structs, and macros organized by source file.

---

## Table of Contents

1. [stories_config.h -- Model Configuration](#stories_configh)
2. [stories_io.h -- IOSurface I/O and Compilation](#stories_ioh)
3. [stories_mil.h -- MIL Program Generators](#stories_milh)
4. [stories_cpu_ops.h -- CPU Operations](#stories_cpu_opsh)
5. [ane_runtime.h -- Generalized ANE Wrapper](#ane_runtimeh)
6. [ane_mil_gen.h -- Composable MIL Helpers](#ane_mil_genh)
7. [ane_rmsnorm_bwd.h -- RMSNorm Backward on ANE](#ane_rmsnorm_bwdh)
8. [ane_classifier.h -- Classifier and Softmax on ANE](#ane_classifierh)
9. [bridge/ane_bridge.h -- C Bridge API](#bridgeane_bridgeh)
10. [MIL Operation Reference](#mil-operation-reference)
11. [Weight Blob Format](#weight-blob-format)

---

## stories_config.h

Model constants, data structures, and memory allocation helpers.

### Macros

| Macro | Value | Description |
|-------|-------|-------------|
| `DIM` | 768 | Model hidden dimension |
| `HIDDEN` | 2048 | FFN intermediate dimension |
| `HEADS` | 12 | Number of attention heads |
| `HD` | 64 (`DIM/HEADS`) | Per-head dimension |
| `SEQ` | 256 | Sequence length |
| `NLAYERS` | 12 | Number of transformer layers |
| `VOCAB` | 32000 | Vocabulary size |
| `ACCUM_STEPS` | 10 | Gradient accumulation steps per compile batch |
| `MAX_COMPILES` | 100 | ANE compile budget before process restart |
| `KERNELS_PER_LAYER` | 5 | Weight-bearing ANE kernels per layer |
| `TOTAL_WEIGHT_KERNELS` | 60 | Total weight-bearing compiles per batch |
| `SCORE_CH` | 3072 (`HEADS*SEQ`) | Attention score channels for SDPA backward |
| `WQ_SZ` | 589824 (`DIM*DIM`) | Size of Q/K/V/O projection weight matrices |
| `WO_SZ` | 589824 (`DIM*DIM`) | Size of output projection |
| `W1_SZ` | 1572864 (`HIDDEN*DIM`) | FFN gate/value projection size |
| `W2_SZ` | 1572864 (`DIM*HIDDEN`) | FFN down-projection size |
| `W3_SZ` | 1572864 (`HIDDEN*DIM`) | FFN value projection size |
| `LAYER_PARAMS` | -- | Total floats per layer: `4*WQ_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM` |
| `TOTAL_PARAMS` | -- | Total model params: `NLAYERS * LAYER_PARAMS + DIM + VOCAB*DIM` |

### Structs

#### `LayerWeights`
Per-layer weight matrices (all `float*`).

| Field | Shape | Description |
|-------|-------|-------------|
| `Wq`, `Wk`, `Wv`, `Wo` | `[DIM, DIM]` | Attention projection weights |
| `W1`, `W3` | `[HIDDEN, DIM]` | FFN gate and value up-projections |
| `W2` | `[DIM, HIDDEN]` | FFN down-projection |
| `rms_att` | `[DIM]` | RMSNorm scale for attention sublayer |
| `rms_ffn` | `[DIM]` | RMSNorm scale for FFN sublayer |

#### `AdamState`
First/second moment buffers for a single parameter group.

| Field | Type | Description |
|-------|------|-------------|
| `m` | `float*` | First moment (mean) estimate |
| `v` | `float*` | Second moment (variance) estimate |
| `n` | `size_t` | Number of parameters |

#### `LayerAdam`
Per-layer Adam optimizer state. Contains one `AdamState` per weight matrix: `Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`, `W3`, `rms_att`, `rms_ffn`.

#### `LayerActs`
Per-layer activation tensors saved for the backward pass.

| Field | Shape | Description |
|-------|-------|-------------|
| `layer_in` | `[DIM, SEQ]` | Input to this layer (for rmsnorm1 backward) |
| `xnorm` | `[DIM, SEQ]` | RMSNorm1 output |
| `Q`, `K`, `V` | `[DIM, SEQ]` | QKV projections |
| `attn_out` | `[DIM, SEQ]` | Attention output (before Wo) |
| `o_out` | `[DIM, SEQ]` | Wo projection output |
| `x2` | `[DIM, SEQ]` | Residual after attention |
| `x2norm` | `[DIM, SEQ]` | RMSNorm2 output |
| `h1`, `h3` | `[HIDDEN, SEQ]` | FFN intermediates (W1 and W3 outputs) |
| `silu_out` | `[HIDDEN, SEQ]` | SiLU(h1) * h3 gated output |
| `ffn_out` | `[DIM, SEQ]` | FFN final output |

#### `LayerGrads`
Per-layer gradient accumulators. Same field names as `LayerWeights` (all `float*`): `Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`, `W3`, `rms_att`, `rms_ffn`.

#### `Kern`
Single ANE kernel handle (stories-specific, single I/O).

| Field | Type | Description |
|-------|------|-------------|
| `model` | `void*` | Retained `_ANEInMemoryModel` |
| `ioIn` | `IOSurfaceRef` | Input IOSurface |
| `ioOut` | `IOSurfaceRef` | Output IOSurface |
| `request` | `void*` | Retained `_ANERequest` |
| `tmpDir` | `void*` | Retained temp directory path |

#### `LayerKernels`
ANE kernels for one transformer layer.

| Field | Type | Description |
|-------|------|-------------|
| `fwdAttn` | `Kern*` | SDPA forward + taps |
| `fwdFFN` | `Kern*` | FFN forward + taps |
| `ffnBwd` | `Kern*` | FFN backward |
| `sdpaBwd1` | `Kern*` | SDPA backward part 1 (Wo^T + dV + scores) |
| `sdpaBwd2` | `Kern*` | SDPA backward part 2 (dQ + dK) |
| `qkvBwd` | `Kern*` | QKV backward (Wq^T, Wk^T, Wv^T) |

#### `CkptHdr`
Checkpoint file header (128 bytes, version 2).

| Field | Type | Description |
|-------|------|-------------|
| `magic` | `int` | `0x424C5A54` ("BLZT") |
| `version` | `int` | 2 |
| `step`, `total_steps` | `int` | Training progress |
| `n_layers`, `vocab_size`, `dim`, `hidden_dim`, `n_heads`, `seq_len` | `int` | Model shape |
| `lr`, `loss` | `float` | Learning rate, last loss |
| `cum_compile`, `cum_train`, `cum_wall` | `double` | Cumulative timing (ms) |
| `cum_steps`, `cum_batches` | `int` | Cumulative counters |
| `adam_t` | `int` | Adam timestep (for bias correction) |
| `pad[3]` | `int` | Alignment padding |

#### `Llama2Config`
Header from llama2.c model files (7 ints): `dim`, `hidden_dim`, `n_layers`, `n_heads`, `n_kv_heads`, `vocab_size`, `seq_len`.

### Global Variables

| Name | Type | Description |
|------|------|-------------|
| `g_D` | `Class` | `_ANEInMemoryModelDescriptor` ObjC class |
| `g_I` | `Class` | `_ANEInMemoryModel` ObjC class |
| `g_AR` | `Class` | `_ANERequest` ObjC class |
| `g_AIO` | `Class` | `_ANEIOSurfaceObject` ObjC class |
| `g_tb` | `mach_timebase_info_data_t` | Mach time base for timing |
| `g_compile_count` | `int` | Running count of ANE compiles |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `ane_init(void)` | `void` | Load AppleNeuralEngine.framework, resolve 4 private class references |
| `tb_ms(uint64_t t)` | `double` | Convert Mach absolute time to milliseconds |
| `adam_alloc(size_t n)` | `AdamState` | Allocate zeroed first/second moment buffers for n parameters |
| `adam_free(AdamState *s)` | `void` | Free an AdamState's buffers |
| `layer_weights_alloc(void)` | `LayerWeights` | Allocate all weight matrices for one layer |
| `layer_weights_free(LayerWeights *w)` | `void` | Free all weight matrices for one layer |
| `layer_adam_alloc(void)` | `LayerAdam` | Allocate Adam state for all weights in one layer |
| `layer_adam_free(LayerAdam *a)` | `void` | Free Adam state for one layer |
| `layer_acts_alloc(void)` | `LayerActs` | Allocate all activation buffers for one layer |
| `layer_acts_free(LayerActs *a)` | `void` | Free all activation buffers for one layer |
| `layer_grads_alloc(void)` | `LayerGrads` | Allocate zeroed gradient accumulators for one layer |
| `layer_grads_zero(LayerGrads *g)` | `void` | Zero all gradient accumulators (between accumulation steps) |
| `layer_grads_free(LayerGrads *g)` | `void` | Free gradient accumulators for one layer |

---

## stories_io.h

IOSurface creation, fp16/fp32 conversion, weight blob building, and ANE kernel compile/run.

**Depends on**: `stories_config.h`, `<arm_neon.h>`

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `make_surface(size_t bytes)` | `IOSurfaceRef` | Create a 1D IOSurface with given byte allocation |
| `build_blob(const float *w, int rows, int cols)` | `NSData*` | Build fp16 weight blob (128B header + row-major fp16 data) from fp32 weights |
| `build_blob_t(const float *w, int rows, int cols)` | `NSData*` | Build fp16 weight blob with transposed layout (col-major fp16 from row-major fp32) |
| `build_blob_fp16(_Float16 *d, int cnt)` | `NSData*` | Build weight blob from pre-existing fp16 data (no conversion) |
| `cvt_f16_f32(float *dst, const _Float16 *src, int n)` | `void` | NEON-vectorized fp16-to-fp32 conversion (8-wide SIMD) |
| `cvt_f32_f16(_Float16 *dst, const float *src, int n)` | `void` | NEON-vectorized fp32-to-fp16 conversion (8-wide SIMD) |
| `io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp)` | `void` | Write fp32 data to IOSurface as fp16 in channel-first `[C,S]` layout |
| `io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp)` | `void` | Read fp16 data from IOSurface at channel offset, convert to fp32 |
| `io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch, int channels, int sp)` | `void` | Copy fp16 data between IOSurfaces at specified channel offsets |
| `io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data, int channels, int sp)` | `void` | Write fp32 data to IOSurface at specific channel offset as fp16 |
| `compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes)` | `Kern*` | Compile MIL text + weight dictionary into a loaded ANE kernel with IOSurfaces. Increments `g_compile_count`. |
| `free_kern(Kern *k)` | `void` | Unload ANE model, release IOSurfaces, remove temp directory, free kernel |
| `ane_run(Kern *k)` | `void` | Run a compiled ANE kernel on current IOSurface contents |

---

## stories_mil.h

MIL program generators for the 6 fused ANE kernel types. Each returns an `NSString*` containing the full MIL program text.

**Depends on**: `stories_io.h`

### Macros

| Macro | Description |
|-------|-------------|
| `MIL_HDR` | Standard MIL program header (version 1.3, buildInfo with coremlc/coremltools versions) |
| `CONV_CONST` | Common conv parameter constants (pad_type, strides, pad, dilations, groups) |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `gen_sdpa_fwd_taps(void)` | `NSString*` | SDPA forward: RMSNorm + QKV + attention + Wo. Output: `concat(o_out, Q, K, V, attn_out, xnorm)` `[1, 6*DIM, 1, SEQ]` |
| `gen_ffn_fwd_taps(void)` | `NSString*` | FFN forward: RMSNorm + W1/W3 + SiLU + W2. Output: `concat(ffn_out, h1, h3, silu_out, x2norm)` `[1, 2*DIM+3*HIDDEN, 1, SEQ]` |
| `gen_ffn_bwd(void)` | `NSString*` | FFN backward: Input `concat(dffn, h1, h3)`. Output: `concat(dx, dh1, dh3)` `[1, DIM+2*HIDDEN, 1, SEQ]` |
| `gen_qkvb(void)` | `NSString*` | QKV backward: Input `concat(dQ, dK, dV)`. Output: `dx` `[1, DIM, 1, SEQ]` |
| `gen_sdpa_bwd1(void)` | `NSString*` | SDPA backward part 1: Input `concat(Q, K, V, dx2)`. Output: `concat(dV, probs, dP)` `[1, DIM+2*SCORE_CH, 1, SEQ]` |
| `gen_sdpa_bwd2(void)` | `NSString*` | SDPA backward part 2: Input `concat(probs, dP, Q, K)`. Output: `concat(dQ, dK)` `[1, 2*DIM, 1, SEQ]` |
| `get_mask_blob(void)` | `NSData*` | Lazily build and cache causal attention mask as fp16 blob. Lower-triangular 0, upper -65504. |

### Global Variables

| Name | Type | Description |
|------|------|-------------|
| `g_mask_blob` | `NSData*` | Cached causal mask blob (built on first call to `get_mask_blob`) |

---

## stories_cpu_ops.h

CPU-side operations using Accelerate framework (vDSP, vvrsqrtf, vvexpf).

**Depends on**: `stories_config.h`

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `rmsnorm(float *out, const float *x, const float *w, int d, int S)` | `void` | RMSNorm forward: `out = x * rsqrt(mean(x^2) + eps) * w`. Vectorized via vDSP. Layout: channel-first `[d, S]`. |
| `rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S)` | `void` | RMSNorm backward: computes `dx` (input gradient) and accumulates `dw` (scale gradient). |
| `adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps)` | `void` | Adam optimizer step with bias correction. Updates weights in-place. `t` is the timestep for bias correction. |
| `cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S)` | `float` | Compute mean cross-entropy loss. Writes `dlogits = (softmax(logits) - one_hot(targets)) / S`. Column-major `[V, S]` layout. Uses vDSP transpose + vvexpf for vectorized softmax. |
| `embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq)` | `void` | Embedding forward: gather rows from `embed[VOCAB, DIM]` into channel-first `x[DIM, SEQ]`. |
| `embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq)` | `void` | Embedding backward: scatter-add `dx` back into embedding table gradient `d_embed`. |

### Global Variables

| Name | Type | Description |
|------|------|-------------|
| `g_rms_tmp` | `float*` | Lazily-allocated scratch buffer for RMSNorm (size SEQ) |

---

## ane_runtime.h

Generalized ANE wrapper with multi-input/output support. Used in bridge, tests, and newer training variants.

### Structs

#### `ANEKernel`
Generalized kernel handle supporting multiple inputs and outputs.

| Field | Type | Description |
|-------|------|-------------|
| `model` | `id` | `_ANEInMemoryModel` instance |
| `ioInputs` | `IOSurfaceRef*` | Array of input IOSurfaces |
| `ioOutputs` | `IOSurfaceRef*` | Array of output IOSurfaces |
| `request` | `id` | `_ANERequest` instance |
| `tmpDir` | `NSString*` | Temp directory for MIL/weights on disk |
| `nInputs`, `nOutputs` | `int` | Number of I/O tensors |
| `inputBytes`, `outputBytes` | `size_t*` | Byte sizes for each I/O tensor |

### Global Variables

| Name | Type | Description |
|------|------|-------------|
| `g_ANEDesc` | `Class` | `_ANEInMemoryModelDescriptor` |
| `g_ANEInMem` | `Class` | `_ANEInMemoryModel` |
| `g_ANEReq` | `Class` | `_ANERequest` |
| `g_ANEIO` | `Class` | `_ANEIOSurfaceObject` |
| `g_ane_loaded` | `bool` | Guard to avoid re-loading the framework |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `ane_init(void)` | `void` | Load AppleNeuralEngine.framework (idempotent), resolve 4 private ObjC classes |
| `ane_create_surface(size_t bytes)` | `IOSurfaceRef` | Create a 1D IOSurface of given byte size |
| `ane_compile(NSData *milText, NSData *weightData, int nInputs, size_t *inputSizes, int nOutputs, size_t *outputSizes)` | `ANEKernel*` | Full compile pipeline: build descriptor, compile MIL, load model, create IOSurfaces + request. Returns NULL on failure. |
| `ane_write_input(ANEKernel *k, int idx, const void *data, size_t bytes)` | `void` | Write raw bytes to the idx-th input IOSurface (lock/memcpy/unlock) |
| `ane_read_output(ANEKernel *k, int idx, void *data, size_t bytes)` | `void` | Read raw bytes from the idx-th output IOSurface (read-lock/memcpy/unlock) |
| `ane_run_kernel(ANEKernel *k)` | `bool` | Run the compiled ANE kernel. Returns true on success. |
| `ane_free(ANEKernel *k)` | `void` | Unload model, release all IOSurfaces, remove temp dir, free struct |

---

## ane_mil_gen.h

Composable MIL generation helpers for common patterns, plus weight blob builders.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `mil_build_weight_blob(const float *w, int out_ch, int in_ch)` | `NSData*` | Build fp16 weight blob with 128B header from fp32 row-major `[out_ch, in_ch]` weights |
| `mil_gen_matmul(int in_ch, int out_ch, int spatial)` | `NSString*` | Generate MIL for matmul `y = W @ x` with both as runtime inputs. Includes fp32-to-fp16-to-fp32 casts. |
| `mil_gen_conv(int in_ch, int out_ch, int spatial)` | `NSString*` | Generate MIL for conv-based linear with baked weights from blob file (inference-only) |
| `mil_gen_qkv(int dim, int spatial)` | `NSString*` | Generate MIL for fused QKV: 3 parallel convs from single input, weights from concatenated blob |
| `mil_build_qkv_weight_blob(const float *wq, const float *wk, const float *wv, int dim)` | `NSData*` | Build concatenated weight blob for fused QKV (3 chunks, each with 64B header + fp16 data) |
| `mil_build_ffn_up_weight_blob(const float *w1, const float *w3, int hidden_dim, int dim)` | `NSData*` | Build concatenated weight blob for fused FFN up-projection (W1 + W3 chunks) |
| `mil_gen_ffn_up(int dim, int hidden_dim, int spatial)` | `NSString*` | Generate MIL for fused FFN up: W1 + W3 parallel convs, outputs h1 and h3 |

---

## ane_rmsnorm_bwd.h

MIL generator for RMSNorm backward on ANE (used by `train_large_ane.m`).

**Depends on**: `stories_mil.h`

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `gen_rmsnorm_bwd(void)` | `NSString*` | Generate MIL for RMSNorm backward. Input: `concat(dy, x)` as `[1, 2*DIM, 1, SEQ]`. Baked weight: RMSNorm scale `w[DIM]`. Output: `dx` as `[1, DIM, 1, SEQ]`. Note: `dw` (weight gradient) stays on CPU. |

---

## ane_classifier.h

MIL generators for classifier operations on ANE (used by `train_large_ane.m`).

**Depends on**: `stories_mil.h`

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `gen_classifier_fwd(void)` | `NSString*` | Classifier forward: single 32000-output-channel conv. Input: `[1, DIM, 1, SEQ]`. Baked: embedding weights `[VOCAB, DIM, 1, 1]`. Output: `[1, VOCAB, 1, SEQ]`. |
| `gen_classifier_bwd(void)` | `NSString*` | Classifier backward: `dx = embed^T @ dlogits`. Uses `matmul` op (not conv, since ANE rejects conv with 32000 input channels). Input: `[1, VOCAB, 1, SEQ]`. Baked: `embed^T [1, DIM, VOCAB]`. Output: `[1, DIM, 1, SEQ]`. |
| `gen_softmax_vocab(void)` | `NSString*` | Softmax over VOCAB dimension: `softmax(x, axis=1)`. Input: `[1, VOCAB, 1, SEQ]`. Output: `[1, VOCAB, 1, SEQ]`. |
| `gen_final_rmsnorm(void)` | `NSString*` | Final RMSNorm (standalone, not fused). Input: `[1, DIM, 1, SEQ]`. Baked: `rms_final[DIM]`. Output: `[1, DIM, 1, SEQ]`. |

---

## bridge/ane_bridge.h

C-callable bridge to ANE private APIs for Python ctypes integration.

### Types

| Type | Description |
|------|-------------|
| `ANEKernelHandle` | Opaque kernel handle (pointer to internal struct) |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `ane_bridge_init(void)` | `int` | Initialize ANE runtime (load private framework, resolve classes). Returns 0 on success, -1 on failure. |
| `ane_bridge_compile(const char *mil_text, size_t mil_len, const uint8_t *weight_data, size_t weight_len, int n_inputs, const size_t *input_sizes, int n_outputs, const size_t *output_sizes)` | `ANEKernelHandle*` | Compile MIL text + single weight blob into ANE kernel. Returns NULL on failure. |
| `ane_bridge_compile_multi_weights(const char *mil_text, size_t mil_len, const char **weight_names, const uint8_t **weight_datas, const size_t *weight_lens, int n_weights, int n_inputs, const size_t *input_sizes, int n_outputs, const size_t *output_sizes)` | `ANEKernelHandle*` | Compile MIL text + multiple named weight files. Weight names use `@model_path/` prefix convention. |
| `ane_bridge_run(ANEKernelHandle *kernel)` | `bool` | Execute a compiled kernel on ANE. Returns true on success. |
| `ane_bridge_write_input(ANEKernelHandle *kernel, int idx, const void *data, size_t bytes)` | `void` | Write data to kernel input IOSurface at index `idx` |
| `ane_bridge_read_output(ANEKernelHandle *kernel, int idx, void *data, size_t bytes)` | `void` | Read data from kernel output IOSurface at index `idx` |
| `ane_bridge_free(ANEKernelHandle *kernel)` | `void` | Unload model, release all IOSurfaces, remove temp dir, free handle |
| `ane_bridge_get_compile_count(void)` | `int` | Get current compile count (for restart budgeting) |
| `ane_bridge_reset_compile_count(void)` | `void` | Reset compile count to zero |
| `ane_bridge_build_weight_blob(const float *src, int rows, int cols, size_t *out_len)` | `uint8_t*` | Build weight blob in ANE format (128B header + fp16). Caller must free via `ane_bridge_free_blob()`. |
| `ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols, size_t *out_len)` | `uint8_t*` | Build transposed weight blob. Caller must free via `ane_bridge_free_blob()`. |
| `ane_bridge_free_blob(void *ptr)` | `void` | Free a blob allocated by `ane_bridge_build_weight_blob*` |

---

## MIL Operation Reference

All MIL programs target `ios18` and use fp16 tensors in `[1, C, 1, S]` layout (or `[1, H, S, S]` for attention scores).

| Operation | MIL Syntax | Purpose |
|-----------|-----------|---------|
| `conv` | `conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=xn)` | Linear projections (all Wq, Wk, Wv, Wo, W1, W2, W3). 1x1 conv = matmul. Weight shape: `[out_ch, in_ch, 1, 1]`. |
| `matmul` | `matmul(transpose_x=tx, transpose_y=ty, x=a, y=b)` | Attention score computation (Q at K^T, scores at V, classifier backward). |
| `softmax` | `softmax(axis=ax, x=ms)` | Attention weight normalization (`axis=-1`) and vocab softmax (`axis=1`). |
| `mul` | `mul(x=a, y=b)` | Element-wise multiply: RMSNorm scaling, SiLU gating, attention scaling, softmax Jacobian. |
| `add` | `add(x=a, y=b)` | Causal mask application, SiLU derivative `(1 + h*(1-sig))`, gradient accumulation. |
| `sub` | `sub(x=a, y=b)` | SiLU derivative: `1 - sigmoid(h1)`, softmax backward: `dp - sum(P*dP)`. |
| `sigmoid` | `sigmoid(x=h1)` | SiLU activation component (SiLU = x * sigmoid(x)). |
| `pow` | `pow(x=ss3, y=nhalf)` | RMSNorm: `x^(-0.5)` = reciprocal sqrt. |
| `reduce_sum` | `reduce_sum(x=sq, axes=rax, keep_dims=kd)` | RMSNorm: sum of squares along channel dim. Softmax backward: row-wise dot product. |
| `reshape` | `reshape(shape=sh, x=xf)` | `[1,DIM,1,SEQ]` to `[1,HEADS,HD,SEQ]` for multi-head attention. Flatten attention scores. |
| `transpose` | `transpose(perm=pm, x=q4)` | Permute `[0,1,3,2]`: swap spatial and head_dim for matmul compatibility. |
| `concat` | `concat(axis=cax, interleave=cid, values=(a,b,c))` | Pack multiple outputs into single IOSurface ("taps"). Always `axis=1`, `interleave=false`. |
| `slice_by_size` | `slice_by_size(x=x, begin=b, size=sz)` | Split concatenated inputs in backward kernels. `begin=[0,offset,0,0]`, `size=[1,channels,1,SEQ]`. |
| `cast` | `cast(dtype=to_fp16, x=x)` | fp32-to-fp16 or fp16-to-fp32 precision conversion (used in ane_mil_gen.h generators). |
| `const` | `const()[name=..., val=...]` | Declare scalar/tensor constants, conv parameters, weight blob references via `BLOBFILE`. |

---

## Weight Blob Format

### Single-weight blob (128 bytes header + data)

```
Offset  Size   Content
------  -----  -------
0       1      0x01 (format marker)
4       1      0x02 (format marker)
5-63    59     zeros (global header padding)
64      4      0xDEADBEEF (chunk magic, little-endian: EF BE AD DE)
68      1      0x01 (chunk marker)
72      4      uint32 data_size (total fp16 bytes = out_ch * in_ch * 2)
80      4      uint32 data_offset (always 128 = 64 global + 64 chunk)
84-127  44     zeros (chunk header padding)
128+    N      fp16 weight data, row-major [out_ch, in_ch]
```

### Multi-weight blob (fused QKV, FFN up)

```
Offset      Content
------      -------
0-63        Global header (same as above)
64          Chunk 0 header (64 bytes): magic, data_size, data_offset
64+64       Chunk 0 data (fp16 weights)
64+cs       Chunk 1 header (64 bytes)
64+cs+64    Chunk 1 data (fp16 weights)
...
```

Where `cs = 64 + n_elements * 2` (chunk header size + data size).

MIL references use `BLOBFILE(path="@model_path/weights/name.bin", offset=uint64(X))` where X is the chunk header offset within the file (64 for first chunk, 64+cs for second, etc.).
