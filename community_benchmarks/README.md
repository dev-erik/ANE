# ANE Community Benchmarks

Standardized benchmark results from different Apple Silicon machines, contributed by the community.

## How to Run

```bash
# Full benchmark (SRAM probe + peak TFLOPS + training)
bash scripts/run_community_benchmark.sh

# Quick benchmark (skip training -- useful if you don't have training data)
bash scripts/run_community_benchmark.sh --skip-training

# Custom training steps
bash scripts/run_community_benchmark.sh --steps 50
```

This produces a JSON file in `community_benchmarks/` named `<chip>_<date>.json`.

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4/M5)
- Xcode command line tools (`xcode-select --install`)
- Python 3.11-3.13 with `coremltools` (auto-installed into a temp venv)
- For training benchmarks: run `cd training && make data` first

## How to Submit

### Option 1: Pull Request

1. Fork this repo
2. Run the benchmark: `bash scripts/run_community_benchmark.sh`
3. Commit the generated JSON file from `community_benchmarks/`
4. Open a PR

### Option 2: GitHub Issue

1. Run the benchmark
2. Open a [new issue](../../issues/new) with title "Benchmark: [Your Chip]"
3. Paste the contents of your JSON file

## Viewing Aggregated Results

```bash
python3 scripts/aggregate_benchmarks.py
```

This reads all JSON files in `community_benchmarks/` and prints a markdown comparison table.

## JSON Schema (v1)

Each submission contains:

```json
{
  "schema_version": 1,
  "timestamp": "2026-03-03T12:00:00Z",
  "system": {
    "chip": "Apple M4 Max",
    "machine": "Mac16,5",
    "macos_version": "26.2",
    "memory_gb": 128,
    "neural_engine_cores": "16"
  },
  "benchmarks": {
    "sram_probe": [
      {"channels": 256, "weight_mb": 0.1, "ms_per_eval": 0.378, "tflops": 0.02, "gflops_per_mb": 177.7},
      ...
    ],
    "inmem_peak": [
      {"depth": 128, "channels": 512, "spatial": 64, "weight_mb": 64.0, "gflops": 4.29, "ms_per_eval": 0.385, "tflops": 11.14},
      ...
    ],
    "training_cpu_classifier": {
      "ms_per_step": 72.4,
      "ane_tflops_sustained": 1.29,
      "ane_util_pct": 8.1,
      "compile_pct": 79.7
    },
    "training_ane_classifier": {
      "ms_per_step": 62.9,
      "ane_tflops_sustained": 1.68,
      "ane_util_pct": 10.6,
      "compile_pct": 84.5
    }
  },
  "summary": {
    "peak_tflops": 11.14,
    "sram_spill_start_channels": 4096,
    "training_ms_per_step_cpu": 72.4,
    "training_ms_per_step_ane": 62.9,
    "training_ane_tflops": 1.68,
    "training_ane_util_pct": 10.6
  }
}
```

## What We're Measuring

| Benchmark | What it tells us |
|-----------|-----------------|
| **sram_probe** | ANE SRAM capacity -- where weight spilling starts |
| **inmem_peak** | Maximum achievable TFLOPS via programmatic MIL |
| **training (CPU cls)** | End-to-end training perf with CPU classifier |
| **training (ANE cls)** | End-to-end training perf with ANE-offloaded classifier |

Key metrics to compare across chips:
- **Peak TFLOPS**: raw ANE compute capability
- **SRAM spill point**: determines max efficient kernel size
- **Training ms/step**: real-world training performance
- **ANE utilization %**: how much of peak we actually use
