#!/bin/bash
# run_community_benchmark.sh -- Standardized ANE benchmark for community submissions
#
# Runs a focused set of benchmarks and outputs a single JSON file that can be
# submitted to the community_benchmarks/ directory via PR or GitHub issue.
#
# Usage:
#   bash scripts/run_community_benchmark.sh [--steps N] [--skip-training]
#
# Output:
#   community_benchmarks/<chip>_<date>.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

STEPS=20
SKIP_TRAINING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --skip-training) SKIP_TRAINING=true; shift ;;
        --help|-h)
            echo "Usage: bash scripts/run_community_benchmark.sh [--steps N] [--skip-training]"
            echo "  --steps N          Training steps (default: 20)"
            echo "  --skip-training    Skip training benchmarks (useful if no training data)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Collect system info ──

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
MACHINE=$(sysctl -n hw.model 2>/dev/null || echo "unknown")
MACOS_VER=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
MACOS_BUILD=$(sw_vers -buildVersion 2>/dev/null || echo "unknown")
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo "0")
MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
MEM_GB=$(echo "scale=0; $MEM_BYTES / 1073741824" | bc 2>/dev/null || echo "0")
NEURAL_CORES=$(sysctl -n hw.optional.ane.num_cores 2>/dev/null || echo "unknown")
DATE_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DATE_SHORT=$(date +"%Y%m%d")

CHIP_SLUG=$(echo "$CHIP" | tr ' ' '_' | tr -d '()' | tr '[:upper:]' '[:lower:]')

echo "=== ANE Community Benchmark ==="
echo "Chip:    $CHIP"
echo "Machine: $MACHINE"
echo "macOS:   $MACOS_VER ($MACOS_BUILD)"
echo "Memory:  ${MEM_GB} GB"
echo "CPUs:    $NCPU"
echo "ANE cores: $NEURAL_CORES"
echo ""

# ── Prerequisites ──

if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: macOS required"; exit 1
fi
if ! sysctl -n hw.optional.arm64 2>/dev/null | grep -q 1; then
    echo "ERROR: Apple Silicon required"; exit 1
fi
if ! xcrun --find clang >/dev/null 2>&1; then
    echo "ERROR: Xcode CLI tools required. Run: xcode-select --install"; exit 1
fi

CC="xcrun clang"
CFLAGS="-O2 -fobjc-arc -fstack-protector-strong -framework Foundation -framework CoreML -framework IOSurface -ldl"

# ── Ask for GitHub username (optional) ──

echo "Enter your GitHub username (optional, press Enter to skip):"
read -r GH_USERNAME
GH_USERNAME=$(echo "$GH_USERNAME" | tr -d '[:space:]' | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-39)

if [[ -n "$GH_USERNAME" ]]; then
    echo "Username: $GH_USERNAME"
else
    echo "Submitting anonymously"
fi
echo ""

# ── Temp file for collecting JSON fragments ──

TMPJSON=$(mktemp /tmp/ane_bench_XXXXXX.json)
trap "rm -f $TMPJSON" EXIT

# Start building the JSON result
USERNAME_LINE=""
if [[ -n "$GH_USERNAME" ]]; then
    USERNAME_LINE="\"username\": \"$GH_USERNAME\","
fi

cat > "$TMPJSON" << HEADER
{
  "schema_version": 1,
  $USERNAME_LINE
  "timestamp": "$DATE_ISO",
  "system": {
    "chip": "$CHIP",
    "machine": "$MACHINE",
    "macos_version": "$MACOS_VER",
    "macos_build": "$MACOS_BUILD",
    "cpu_cores": $NCPU,
    "memory_gb": $MEM_GB,
    "neural_engine_cores": "$NEURAL_CORES"
  },
HEADER

# ── 1. SRAM Probe ──

echo "--- Running sram_probe ---"
SRAM_JSON="[]"

# Find a suitable Python (3.11-3.13; coremltools doesn't support 3.14+)
find_python() {
    # 1. Existing venv from a previous run
    if [[ -x /tmp/ane_venv/bin/python3 ]]; then
        echo "/tmp/ane_venv/bin/python3"; return
    fi
    # 2. Search Homebrew (ARM + Intel), pyenv, and system python
    local SEARCH_PATHS=(
        "/opt/homebrew/opt/python@{VER}/bin/python{VER}"
        "/usr/local/opt/python@{VER}/bin/python{VER}"
        "$HOME/.pyenv/versions/{VER}.*/bin/python3"
    )
    for pyver in 3.12 3.13 3.11; do
        for tmpl in "${SEARCH_PATHS[@]}"; do
            local pattern="${tmpl//\{VER\}/$pyver}"
            for PY in $pattern; do
                if [[ -x "$PY" ]]; then
                    echo "  Found Python: $PY"
                    "$PY" -m venv /tmp/ane_venv 2>&1 && \
                        /tmp/ane_venv/bin/pip install -q coremltools numpy 2>&1
                    if [[ $? -eq 0 ]]; then
                        echo "/tmp/ane_venv/bin/python3"; return
                    else
                        echo "  WARNING: venv/pip setup failed for $PY" >&2
                        rm -rf /tmp/ane_venv
                    fi
                fi
            done
        done
    done
    # 3. System python3 (check version is 3.11-3.13)
    if command -v python3 &>/dev/null; then
        local SYS_VER
        SYS_VER=$(python3 -c "import sys; print(f'{sys.version_info.minor}')" 2>/dev/null)
        if [[ "$SYS_VER" =~ ^(11|12|13)$ ]]; then
            echo "  Found system Python 3.$SYS_VER"
            python3 -m venv /tmp/ane_venv 2>&1 && \
                /tmp/ane_venv/bin/pip install -q coremltools numpy 2>&1
            if [[ $? -eq 0 ]]; then
                echo "/tmp/ane_venv/bin/python3"; return
            else
                echo "  WARNING: venv/pip setup failed for system python3" >&2
                rm -rf /tmp/ane_venv
            fi
        fi
    fi
    return 1
}

# Generate mlpackage models if needed
if ! ls /tmp/ane_sram_*ch_*sp.mlpackage >/dev/null 2>&1; then
    echo "  Generating mlpackage models..."
    VENV_PYTHON=""
    VENV_PYTHON=$(find_python 2>&1 | tee /dev/stderr | tail -1)
    if [[ -z "$VENV_PYTHON" || ! -x "$VENV_PYTHON" ]]; then
        echo "  WARNING: No compatible Python 3.11-3.13 found for coremltools."
        echo "  SRAM probe will be skipped. Install Python 3.12 to enable it:"
        echo "    brew install python@3.12"
        VENV_PYTHON=""
    fi
    if [[ -n "$VENV_PYTHON" ]]; then
        "$VENV_PYTHON" "$SCRIPT_DIR/gen_mlpackages.py" 2>&1
        if [[ $? -eq 0 ]]; then
            echo "  mlpackage models generated"
        else
            echo "  WARNING: mlpackage generation failed. SRAM probe will be skipped."
        fi
    fi
fi

if ls /tmp/ane_sram_*ch_*sp.mlpackage >/dev/null 2>&1; then
    cd "$ROOT_DIR"
    if $CC $CFLAGS -o sram_probe sram_probe.m 2>&1; then
        SRAM_OUTPUT=$(./sram_probe 2>&1) || true
        echo "  sram_probe complete"

        SRAM_JSON=$(echo "$SRAM_OUTPUT" | python3 -c "
import sys, json, re
results = []
for line in sys.stdin:
    line = line.strip()
    m = re.match(r'\s*(\d+)\s+ch\s+([\d.]+)\s+([\d.]+)\s+ms\s+([\d.]+)\s+([\d.]+)', line)
    if m:
        results.append({
            'channels': int(m.group(1)),
            'weight_mb': float(m.group(2)),
            'ms_per_eval': float(m.group(3)),
            'tflops': float(m.group(4)),
            'gflops_per_mb': float(m.group(5))
        })
print(json.dumps(results))
" 2>/dev/null || echo "[]")
    else
        echo "  WARNING: sram_probe compilation failed"
    fi
else
    echo "  SKIPPED: no mlpackage models (need Python 3.11-3.13 + coremltools)"
fi

# ── 2. InMem Peak ──

echo "--- Running inmem_peak ---"
PEAK_JSON="[]"

cd "$ROOT_DIR"
$CC $CFLAGS -o inmem_peak inmem_peak.m 2>/dev/null

PEAK_OUTPUT=$(./inmem_peak 2>&1) || true
echo "  inmem_peak complete"

PEAK_JSON=$(echo "$PEAK_OUTPUT" | python3 -c "
import sys, json, re
results = []
for line in sys.stdin:
    line = line.strip()
    m = re.match(r'(\d+)x\s+conv\s+(\d+)ch\s+sp(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+ms\s+([\d.]+)', line)
    if m:
        results.append({
            'depth': int(m.group(1)),
            'channels': int(m.group(2)),
            'spatial': int(m.group(3)),
            'weight_mb': float(m.group(4)),
            'gflops': float(m.group(5)),
            'ms_per_eval': float(m.group(6)),
            'tflops': float(m.group(7))
        })
print(json.dumps(results))
" 2>/dev/null || echo "[]")

# ── 3. Training (optional) ──

echo "--- Running training benchmark ($STEPS steps) ---"
TRAIN_CPU_JSON="{}"
TRAIN_ANE_JSON="{}"

if ! $SKIP_TRAINING; then
    cd "$TRAINING_DIR"

    # Build training binaries
    make train_large train_large_ane 2>/dev/null || true

    if [[ -x ./train_large ]]; then
        TRAIN_OUTPUT=$(./train_large --steps "$STEPS" 2>&1) || true
        echo "  train_large complete"

        TRAIN_CPU_JSON=$(echo "$TRAIN_OUTPUT" | python3 -c "
import sys, json, re
result = {}
for line in sys.stdin:
    line = line.strip()
    if line.startswith('{\"type\":\"perf\"'):
        d = json.loads(line)
        result['ane_tflops'] = d.get('ane_tflops')
        result['ane_util_pct'] = d.get('ane_util_pct')
    m = re.match(r'Avg train:\s+([\d.]+)\s+ms/step', line)
    if m: result['ms_per_step'] = float(m.group(1))
    m = re.match(r'ANE TFLOPS:\s+([\d.]+)', line)
    if m: result['ane_tflops_sustained'] = float(m.group(1))
    m = re.match(r'Total TFLOPS:\s+([\d.]+)', line)
    if m: result['total_tflops'] = float(m.group(1))
    m = re.match(r'ANE utilization:\s+([\d.]+)%', line)
    if m: result['ane_util_pct'] = float(m.group(1))
    m = re.match(r'Compile time:\s+\d+\s+ms\s+\(([\d.]+)%\)', line)
    if m: result['compile_pct'] = float(m.group(1))
    m = re.match(r'Train time:\s+\d+\s+ms\s+\(([\d.]+)%\)', line)
    if m: result['train_pct'] = float(m.group(1))
print(json.dumps(result))
" 2>/dev/null || echo "{}")
    fi

    if [[ -x ./train_large_ane ]]; then
        TRAIN_ANE_OUTPUT=$(./train_large_ane --steps "$STEPS" 2>&1) || true
        echo "  train_large_ane complete"

        TRAIN_ANE_JSON=$(echo "$TRAIN_ANE_OUTPUT" | python3 -c "
import sys, json, re
result = {}
for line in sys.stdin:
    line = line.strip()
    m = re.match(r'Avg train:\s+([\d.]+)\s+ms/step', line)
    if m: result['ms_per_step'] = float(m.group(1))
    m = re.match(r'ANE TFLOPS:\s+([\d.]+)', line)
    if m: result['ane_tflops_sustained'] = float(m.group(1))
    m = re.match(r'Total TFLOPS:\s+([\d.]+)', line)
    if m: result['total_tflops'] = float(m.group(1))
    m = re.match(r'ANE utilization:\s+([\d.]+)%', line)
    if m: result['ane_util_pct'] = float(m.group(1))
    m = re.match(r'Compile time:\s+\d+\s+ms\s+\(([\d.]+)%\)', line)
    if m: result['compile_pct'] = float(m.group(1))
    m = re.match(r'Train time:\s+\d+\s+ms\s+\(([\d.]+)%\)', line)
    if m: result['train_pct'] = float(m.group(1))
print(json.dumps(result))
" 2>/dev/null || echo "{}")
    fi
else
    echo "  SKIPPED (--skip-training)"
fi

# ── Assemble final JSON ──

OUTDIR="$ROOT_DIR/community_benchmarks"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/${CHIP_SLUG}_${DATE_SHORT}.json"
if [[ -f "$OUTFILE" ]]; then
    i=2
    while [[ -f "${OUTFILE%.json}_${i}.json" ]]; do i=$((i+1)); done
    OUTFILE="${OUTFILE%.json}_${i}.json"
fi

python3 -c "
import json, sys

with open('$TMPJSON') as f:
    partial = f.read()

sram = json.loads('''$SRAM_JSON''')
peak = json.loads('''$PEAK_JSON''')
train_cpu = json.loads('''$TRAIN_CPU_JSON''')
train_ane = json.loads('''$TRAIN_ANE_JSON''')

peak_tflops = max((r['tflops'] for r in peak), default=0)
sram_peak_eff = max((r['gflops_per_mb'] for r in sram), default=0)
sram_spill_ch = 0
prev_tflops = 0
for r in sorted(sram, key=lambda x: x['channels']):
    if prev_tflops > 0 and r['tflops'] < prev_tflops * 0.6:
        sram_spill_ch = r['channels']
        break
    prev_tflops = max(prev_tflops, r['tflops'])

result = json.loads(partial + '\"_\": 0}')
del result['_']

result['benchmarks'] = {
    'sram_probe': sram,
    'inmem_peak': peak,
    'training_cpu_classifier': train_cpu,
    'training_ane_classifier': train_ane
}

result['summary'] = {
    'peak_tflops': round(peak_tflops, 2),
    'sram_peak_efficiency_gflops_per_mb': round(sram_peak_eff, 1),
    'sram_spill_start_channels': sram_spill_ch,
    'training_ms_per_step_cpu': train_cpu.get('ms_per_step'),
    'training_ms_per_step_ane': train_ane.get('ms_per_step'),
    'training_ane_tflops': train_ane.get('ane_tflops_sustained') or train_cpu.get('ane_tflops_sustained'),
    'training_ane_util_pct': train_ane.get('ane_util_pct') or train_cpu.get('ane_util_pct')
}

with open('$OUTFILE', 'w') as f:
    json.dump(result, f, indent=2)
    f.write('\n')

print(json.dumps(result['summary'], indent=2))
"

echo ""
echo "=== Benchmark complete ==="
echo "Results saved to: $OUTFILE"
echo ""

# ── Optional: submit to community database ──

DASHBOARD_URL="${ANE_DASHBOARD_URL:-https://web-lac-sigma-61.vercel.app}"
SUBMIT_URL="$DASHBOARD_URL/api/submit"

echo "Would you like to submit your results to the ANE community benchmark database? (y/N)"
read -r SUBMIT_ANSWER

if [[ "$SUBMIT_ANSWER" =~ ^[Yy]$ ]]; then
    echo "Submitting to $SUBMIT_URL ..."

    HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST "$SUBMIT_URL" \
        -H "Content-Type: application/json" \
        -d @"$OUTFILE" 2>/dev/null) || true

    HTTP_BODY=$(echo "$HTTP_RESPONSE" | sed '$d')
    HTTP_CODE=$(echo "$HTTP_RESPONSE" | tail -1)

    case "$HTTP_CODE" in
        201)
            SUBMIT_ID=$(echo "$HTTP_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")
            echo "Submitted successfully! (ID: $SUBMIT_ID)"
            echo "View results at: $DASHBOARD_URL"
            ;;
        409)
            echo "Already submitted (duplicate detected within the last hour)."
            echo "View results at: $DASHBOARD_URL"
            ;;
        429)
            echo "Rate limited -- too many submissions. Try again later."
            echo "You can also submit via GitHub PR instead (see below)."
            ;;
        *)
            echo "Submission failed (HTTP $HTTP_CODE). You can submit manually instead."
            ;;
    esac
    echo ""
fi

echo "Alternative submission methods:"
echo "  1. Fork https://github.com/maderix/ANE"
echo "  2. Add $OUTFILE to your fork"
echo "  3. Open a Pull Request"
echo ""
echo "Or paste the contents of $OUTFILE in a GitHub issue."
