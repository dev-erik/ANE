# ANE Benchmark Results: Apple M4 Max

**Date**: March 3, 2026
**Machine**: Mac16,5 (MacBook Pro, Apple M4 Max)
**macOS**: 26.2
**ANE Peak**: 15.8 TFLOPS (theoretical)

## Training Performance

### train_large (CPU classifier path)

| Metric | Value |
|--------|-------|
| Model | Stories110M (12 layers, dim=768, hidden=2048) |
| Kernels | 72 (60 weight-bearing + 12 static sdpaBwd2) |
| Avg step time | 72.4 ms/step |
| ANE TFLOPS | 1.29 sustained |
| Total TFLOPS | 2.41 (ANE+CPU) |
| ANE utilization | 8.1% of 15.8 TFLOPS |
| Compile time | 79.7% of wall time |
| Train time | 16.4% of wall time |

### train_large_ane (ANE-offloaded classifier)

| Metric | Value |
|--------|-------|
| Model | Stories110M (same as above) |
| Kernels | 99 (86 weight-bearing + 13 static) |
| Avg step time | 62.9 ms/step |
| ANE TFLOPS | 1.68 sustained |
| Total TFLOPS | 2.77 (ANE+CPU) |
| ANE utilization | 10.6% of 15.8 TFLOPS |
| Compile time | 84.5% of wall time |
| Train time | 12.5% of wall time |

**Step time breakdown (ms/step, ANE classifier path):**

| Component | Time (ms) | Description |
|-----------|-----------|-------------|
| ane | 10-12 | ANE kernel dispatch + evaluation |
| elem | 12-13 | Elementwise ops (residuals, activations) |
| cls | 5-6 | Classifier forward + backward |
| io | 3-5 | IOSurface data transfers |
| rms | 0.1 | RMSNorm |
| cblas_wait | 0.0 | BLAS sync overhead |

## Programmatic MIL Peak TFLOPS

```
Config                         W(MB)   GFLOP   ms/eval  TFLOPS
----------------------------------------------------------------------
32x conv 512ch sp64            16.0    1.07    0.408 ms   2.63
48x conv 512ch sp64            24.0    1.61    0.262 ms   6.15
64x conv 512ch sp64            32.0    2.15    0.244 ms   8.80
96x conv 512ch sp64            48.0    3.22    0.326 ms   9.89
128x conv 512ch sp64           64.0    4.29    0.385 ms  11.14
64x conv 256ch sp64             8.0    0.54    0.365 ms   1.47
128x conv 256ch sp64           16.0    1.07    0.454 ms   2.37
256x conv 256ch sp64           32.0    2.15    0.351 ms   6.11
64x conv 384ch sp64            18.0    1.21    0.429 ms   2.82
128x conv 384ch sp64           36.0    2.42    0.354 ms   6.82
```

**Peak observed: 11.14 TFLOPS** (128x conv 512ch sp64, 64 MB weights)

## In-Memory ANE Benchmark (via mlpackage)

```
Config         W (MB)    ms/eval   TFLOPS
---------------------------------------------
 256ch x64sp     0.1     0.319 ms    0.03
 512ch x64sp     0.5     0.357 ms    0.09
1024ch x64sp     2.0     0.457 ms    0.29
2048ch x64sp     8.0     0.254 ms    2.11
3072ch x64sp    18.0     0.389 ms    3.10
4096ch x64sp    32.0     1.148 ms    1.87
```

## SRAM Probe Results

### Coarse Probe (varying channels + spatial)

```
Config                      W (MB)  Act(MB)  Tot(MB)    ms/eval   TFLOPS
--------------------------------------------------------------------------
256ch x 64sp                  0.1     0.03      0.2     0.378 ms    0.02
512ch x 64sp                  0.5     0.06      0.6     0.389 ms    0.09
1024ch x 64sp                 2.0     0.12      2.2     0.392 ms    0.34
2048ch x 64sp                 8.0     0.25      8.5     0.218 ms    2.47
3072ch x 64sp                18.0     0.38     18.8     0.396 ms    3.05
4096ch x 64sp                32.0     0.50     33.0     1.116 ms    1.92
5120ch x 64sp                50.0     0.62     51.2     0.767 ms    4.38
6144ch x 64sp                72.0     0.75     73.5     0.872 ms    5.54
8192ch x 32sp               128.0     0.50    129.0     4.195 ms    1.02
```

### Fine Probe (spatial=64, weights only)

```
Channels       W (MB)    ms/eval   TFLOPS    GFLOPS/MB
--------------------------------------------------------------
   256 ch       0.1     0.378 ms    0.02       177.7
   512 ch       0.5     0.431 ms    0.08       155.6
  1024 ch       2.0     0.411 ms    0.33       163.5
  1536 ch       4.5     0.493 ms    0.61       136.1
  2048 ch       8.0     0.410 ms    1.31       163.9
  2560 ch      12.5     0.237 ms    3.53       282.6  <-- peak efficiency
  3072 ch      18.0     0.335 ms    3.60       200.1
  3584 ch      24.5     0.414 ms    3.97       162.1
  4096 ch      32.0     1.134 ms    1.89        59.2  <-- spilling
  4608 ch      40.5     0.563 ms    4.83       119.2
  5120 ch      50.0     0.659 ms    5.09       101.8
  6144 ch      72.0     0.844 ms    5.73        79.5  <-- spilling
  8192 ch     128.0     4.203 ms    1.02         8.0  <-- catastrophic spilling
```

### SRAM Analysis

The M4 Max ANE SRAM appears to be approximately **24-32 MB**:

- **Peak efficiency** at 2560ch (12.5 MB weights): 282.6 GFLOPS/MB, 3.53 TFLOPS
- **First spill** at 4096ch (32.0 MB): drops to 59.2 GFLOPS/MB (1.89 TFLOPS)
- **Catastrophic** at 8192ch (128.0 MB): 8.0 GFLOPS/MB (1.02 TFLOPS)

The 4608ch recovery (4.83 TFLOPS despite 40.5 MB weights) suggests the ANE may use tiling strategies for some weight configurations.

Training kernels (dim=768, weight matrices ~1.2 MB fp16 each) stay well within the SRAM budget.

## Known Test Results

| Test | Status | Notes |
|------|--------|-------|
| test_rmsnorm_bwd | PASS | ANE-accelerated RMSNorm backward |
| test_classifier | PASS | 4 tests passed; ANE backward 3x slower than CPU cblas for matmul |
| test_weight_reload | FAIL (expected) | ANE bakes weights at compile time; IOSurface override doesn't work |
| test_perf_stats | PASS | _ANEPerformanceStats API accessible |
| test_qos_sweep | PASS | QoS parameter has no measurable effect on latency |
| test_ane_advanced | PASS | Advanced ANE operations verified |
| inmem_basic | PASS | In-memory compilation and execution verified |
| inmem_bench | PASS | Multi-config benchmarks via mlpackage |
| inmem_peak | PASS | Peak TFLOPS measurement via programmatic MIL |
| sram_bench | PASS | SRAM capacity probing |
| sram_probe | PASS | Fine-grained SRAM spilling detection |

## Reproducing

```bash
cd scripts && bash run_benchmarks.sh
```

The benchmark script auto-generates required `.mlpackage` models (needs Python 3.11-3.13 with `coremltools`).

Override training data paths:
```bash
ANE_MODEL_PATH=/path/to/stories110M.bin ANE_DATA_PATH=/path/to/data.bin ./train_large
```
