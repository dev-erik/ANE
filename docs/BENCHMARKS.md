# ANE Training -- Benchmarks and Tests Guide

All benchmarks and tests require **macOS 15+ on Apple Silicon** (tested on M4, M5).

---

## Quick Start

```bash
# Build and run training benchmark (100 steps)
cd training
make train_large && ./train_large --steps 100

# Run the automated benchmark suite
cd ..
bash scripts/run_benchmarks.sh
```

---

## Training Benchmarks

### train_large (CPU classifier)

The main 12-layer Stories110M training loop with classifier on CPU.

| Item | Details |
|------|---------|
| **Purpose** | Full transformer training benchmark |
| **Measures** | ms/step, ANE TFLOPS, ANE utilization %, per-component timing |
| **Prerequisites** | Training data: `bash download_data.sh` (or runs on random data if absent) |
| **Build** | `cd training && make train_large` |
| **Run** | `./train_large --steps 100` |
| **CLI flags** | `--steps N` (default 10000), `--lr F` (default 3e-4), `--resume` |

**Expected output:**

```
ane=9.6 io=4.1 cls=9.1 elem=14.4 rms=0.1 cblas_wait=2.3 ms/step

=== Efficiency Report ===
Total steps:     100
Avg train:       107.0 ms/step
ANE TFLOPS:      2.45 sustained
ANE utilization: 15.5% of 15.8 TFLOPS
```

### train_large_ane (ANE classifier)

Same training with classifier, softmax, and RMSNorm backward offloaded to ANE.

| Item | Details |
|------|---------|
| **Purpose** | Measure ANE-offloaded training (16% faster) |
| **Build** | `cd training && make train_large_ane` |
| **Run** | `./train_large_ane --steps 100` |

**Compare baseline vs ANE-offloaded:**

```bash
make train_large && ./train_large --steps 100
make train_large_ane && ./train_large_ane --steps 100
```

### Dashboard (live monitoring)

```bash
pip install blessed psutil numpy
sudo python3 dashboard.py            # live mode (needs powermetrics)
sudo python3 dashboard.py --resume   # attach to resumed training
```

| Flag | Description |
|------|-------------|
| `--resume` | Resume from checkpoint |
| `--infinite` | Train indefinitely |
| `--no-powermetrics` | Disable power monitoring |
| `--no-generate` | Disable text generation preview |
| `--steps N` | Total steps (default 10000) |

---

## Root-Level Benchmark Scripts

All root-level scripts are standalone Objective-C programs. Common build pattern:

```bash
xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML \
  -framework IOSurface -ldl -o <output> <source>.m
```

### inmem_peak.m -- Peak TFLOPS (self-contained)

**No prerequisites.** Generates MIL and weight blobs programmatically.

| Item | Details |
|------|---------|
| **Purpose** | Maximum sustained TFLOPS via deep conv chains (32-256 layers deep) |
| **Measures** | ms per run, TFLOPS, % peak across 10 configurations |
| **Prerequisites** | None (self-contained MIL generation) |
| **Build** | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -ldl -o inmem_peak inmem_peak.m` |
| **Run** | `./inmem_peak` |

**Expected output:**

```
=== Programmatic MIL to In-Memory ANE Peak ===

Config                       W(MB)   GFLOP   ms/run   TFLOPS  %peak
----------------------------------------------------------------------
32x conv 512ch sp64            16.0    1.07    X.XXX ms   Y.YY   Z.Z%
64x conv 512ch sp64            32.0    2.15    X.XXX ms   Y.YY   Z.Z%
...
```

### inmem_basic.m -- In-Memory Proof-of-Concept

| Item | Details |
|------|---------|
| **Purpose** | End-to-end test: compile, load, run, benchmark using `_ANEInMemoryModel` |
| **Prerequisites** | Pre-built mlpackage at `/tmp/ane_sram_256ch_64sp.mlpackage` |
| **Build** | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -ldl -o inmem_basic inmem_basic.m` |
| **Run** | `./inmem_basic` |

### inmem_bench.m -- Dispatch Latency

| Item | Details |
|------|---------|
| **Purpose** | ANE dispatch latency across 6 model sizes (256-4096 channels) |
| **Measures** | ms per run, TFLOPS at each configuration |
| **Prerequisites** | Pre-built mlpackages for all 6 configs |
| **Build** | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -ldl -o inmem_bench inmem_bench.m` |
| **Run** | `./inmem_bench` |

### sram_bench.m -- SRAM Capacity Probe

| Item | Details |
|------|---------|
| **Purpose** | Find SRAM capacity by detecting performance cliff at increasing weight sizes |
| **Measures** | ms per run, TFLOPS, weight/activation/total memory at 9 configurations |
| **Prerequisites** | Pre-built mlpackages for 9 configs (256-8192 channels) |
| **Build** | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -ldl -o sram_bench sram_bench.m` |
| **Run** | `./sram_bench` |

### sram_probe.m -- Fine-Grained SRAM Exploration

| Item | Details |
|------|---------|
| **Purpose** | Finer-grained SRAM probe with 13 data points and GFLOPS/MB efficiency |
| **Measures** | ms per run, TFLOPS, GFLOPS/MB with spilling indicators |
| **Prerequisites** | Pre-built mlpackages for 13 configs |
| **Build** | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -ldl -o sram_probe sram_probe.m` |
| **Run** | `./sram_probe` |

### api_exploration.m -- API Discovery

| Item | Details |
|------|---------|
| **Purpose** | Explore ANE private API surface (class methods, file structures, internal objects) |
| **Prerequisites** | Pre-built mlpackage at `/tmp/ane_sram_1024ch_64sp.mlpackage` |
| **Build** | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -ldl -o api_exploration api_exploration.m` |
| **Run** | `./api_exploration` |

---

## Test Files

### Tests with Makefile targets (cd training/)

| Test | Build | What It Tests |
|------|-------|---------------|
| `test_rmsnorm_bwd` | `make test_rmsnorm_bwd` | RMSNorm backward on ANE vs CPU reference. PASS: max diff < 0.05, mean < 0.01. Benchmarks 100 runs. |
| `test_classifier` | `make test_classifier` | 4-part: final RMSNorm, classifier forward (32000-ch conv), softmax over VOCAB, classifier backward. |
| `test_weight_reload` | `make test_weight_reload` | Tests if weights can be hot-swapped by overwriting blob files + unload/reload. Key finding: NO, weights are baked. |
| `test_perf_stats` | `make test_perf_stats` | Probes `_ANEPerformanceStats` class methods, properties, and instantiation. Tests perfStats in `_ANERequest`. |
| `test_qos_sweep` | `make test_qos_sweep` | QoS parameter sweep (0-63) across compile, load, run. Finding: no measurable latency difference. |
| `test_ane_advanced` | `make test_ane_advanced` | Probes SharedEvents, weightsBuffer IOSurface, procedureIndex, ChainingRequest. Enumerates all 67 ANE classes. |

Build all probe tests at once: `make probes`

### Tests without Makefile targets (manual build)

| Test | Build Command | What It Tests |
|------|---------------|---------------|
| `test_ane_causal_attn` | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o test_ane_causal_attn test_ane_causal_attn.m` | Decomposed causal attention: Q at K^T on ANE, mask+softmax on CPU, scores at V on ANE |
| `test_ane_sdpa5` | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o test_ane_sdpa5 test_ane_sdpa5.m` | 4 approaches to causal masking with `scaled_dot_product_attention` |
| `test_conv_attn3` | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o test_conv_attn3 test_conv_attn3.m` | Grouped conv approach to attention (K,V baked as conv weights) |
| `test_full_fused` | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -ldl -o test_full_fused test_full_fused.m` | Full fused attention + FFN in single MIL dispatch at DIM=768, HEADS=12, SEQ=64 |
| `test_fused_qkv` | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o test_fused_qkv test_fused_qkv.m` | Fused QKV (3 convs + concat in one dispatch) vs separate dispatches |
| `test_fused_bwd` | `xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o test_fused_bwd test_fused_bwd.m` | Fused backward: slice_by_size + 2 convs + add in one kernel |

---

## Bridge Library

```bash
cd bridge
make                          # Build libane_bridge.dylib
make test                     # Build and link test_bridge
./test_bridge                 # Run bridge tests
```

---

## Known Results

### M4 (from README)

**Single-layer (dim=768, seq=512):**

| Optimization | ms/step | ANE utilization |
|---|---|---|
| Baseline (vDSP transpose) | 33.5 | 3.1% |
| Channel-first layout | 20.3 | 5.2% |
| vDSP vectorized RMSNorm | 14.2 | 7.4% |
| GCD async cblas overlap | 11.4 | 9.2% |
| ANE RMSNorm fusion | 11.4 | 9.2% |
| Wo^T fusion (7 to 6 kernels) | 11.4 | 9.2% |
| Deferred cblas wait | **9.3** | **11.2%** |

**Full Stories110M (12 layers):**

| Component | Time (ms/step) |
|-----------|---------------|
| ANE runs | 9.6 |
| IO (fp16 conversion) | 4.1 |
| Classifier (cblas) | 9.1 |
| Cross-entropy + residuals | 14.4 |
| RMSNorm | 0.1 |
| **Total** | **~107** |

### M5 Probe Results (from m5result.md)

**Machine**: Apple M5, macOS 26.3, ANE Family H16 (same as M4)

- **Weight reload**: FAIL -- weights baked at compile time, cannot be overwritten
- **QoS sweep**: All QoS 0-63 work, no measurable latency difference
- **Performance stats**: `_ANEPerformanceStats` class exists, `alloc/init` returns nil (needs factory methods)
- **weightsBuffer IOSurface**: Does NOT override compiled weights
- **ChainingRequest**: Exists with loopback and pipeline support -- most promising for utilization improvement

---

## Timing Metrics Key

| Metric | What it measures |
|--------|-----------------|
| `ane` | ANE kernel runs (all 6 kernels per layer x 12 layers) |
| `io` | fp16-to-fp32 IOSurface data transfer (NEON conversion) |
| `cls` | Classifier matmul (CPU cblas_sgemm) |
| `elem` | Embedding lookup, residual adds, cross-entropy |
| `rms` | RMSNorm forward/backward (CPU vDSP) |
| `cblas_wait` | Time waiting for async dW gradient sgemms to complete |
