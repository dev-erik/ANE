#!/usr/bin/env python3
"""Aggregate community benchmark JSON files into summary tables.

Usage:
    python3 scripts/aggregate_benchmarks.py [community_benchmarks/]

Reads all .json files from the given directory (default: community_benchmarks/)
and produces:
  1. A markdown summary table to stdout
  2. A combined JSON file at community_benchmarks/SUMMARY.json
"""

import json
import os
import sys
from pathlib import Path

def load_submissions(directory):
    submissions = []
    for f in sorted(Path(directory).glob("*.json")):
        if f.name == "SUMMARY.json":
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            if data.get("schema_version") != 1:
                print(f"  SKIP {f.name}: unknown schema_version", file=sys.stderr)
                continue
            data["_filename"] = f.name
            submissions.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  SKIP {f.name}: {e}", file=sys.stderr)
    return submissions

def format_table(submissions):
    lines = []
    lines.append("# ANE Community Benchmark Results\n")
    lines.append(f"Total submissions: {len(submissions)}\n")

    header = (
        "| Chip | Machine | macOS | Memory | "
        "Peak TFLOPS | SRAM Spill (ch) | "
        "Train ms/step (CPU) | Train ms/step (ANE) | "
        "ANE TFLOPS | ANE Util % | Date |"
    )
    sep = "|" + "|".join(["---"] * 11) + "|"
    lines.append(header)
    lines.append(sep)

    for s in submissions:
        sys_info = s.get("system", {})
        summary = s.get("summary", {})

        def fmt(v, suffix=""):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.2f}{suffix}"
            return str(v)

        row = "| {} | {} | {} | {} GB | {} | {} | {} | {} | {} | {} | {} |".format(
            sys_info.get("chip", "?"),
            sys_info.get("machine", "?"),
            sys_info.get("macos_version", "?"),
            sys_info.get("memory_gb", "?"),
            fmt(summary.get("peak_tflops")),
            summary.get("sram_spill_start_channels") or "-",
            fmt(summary.get("training_ms_per_step_cpu")),
            fmt(summary.get("training_ms_per_step_ane")),
            fmt(summary.get("training_ane_tflops")),
            fmt(summary.get("training_ane_util_pct"), "%"),
            s.get("timestamp", "?")[:10],
        )
        lines.append(row)

    lines.append("")

    if submissions:
        lines.append("## SRAM Probe Comparison\n")
        all_channels = set()
        for s in submissions:
            for probe in s.get("benchmarks", {}).get("sram_probe", []):
                all_channels.add(probe["channels"])
        all_channels = sorted(all_channels)

        if all_channels:
            header_cols = ["Channels (W MB)"] + [
                s.get("system", {}).get("chip", "?").replace("Apple ", "")
                for s in submissions
            ]
            lines.append("| " + " | ".join(header_cols) + " |")
            lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

            for ch in all_channels:
                row_parts = []
                weight_mb = None
                for s in submissions:
                    probe_data = {p["channels"]: p for p in s.get("benchmarks", {}).get("sram_probe", [])}
                    if ch in probe_data:
                        p = probe_data[ch]
                        if weight_mb is None:
                            weight_mb = p["weight_mb"]
                        row_parts.append(f"{p['tflops']:.2f} TFLOPS ({p['ms_per_eval']:.3f} ms)")
                    else:
                        row_parts.append("-")

                ch_label = f"{ch} ({weight_mb:.1f} MB)" if weight_mb else str(ch)
                lines.append("| " + ch_label + " | " + " | ".join(row_parts) + " |")
            lines.append("")

    return "\n".join(lines)

def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "community_benchmarks"

    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}", file=sys.stderr)
        print("Run the community benchmark first:", file=sys.stderr)
        print("  bash scripts/run_community_benchmark.sh", file=sys.stderr)
        sys.exit(1)

    submissions = load_submissions(directory)
    if not submissions:
        print("No valid benchmark submissions found.", file=sys.stderr)
        sys.exit(1)

    table = format_table(submissions)
    print(table)

    summary_path = os.path.join(directory, "SUMMARY.json")
    combined = {
        "generated": submissions[0].get("timestamp", ""),
        "count": len(submissions),
        "submissions": [
            {
                "chip": s.get("system", {}).get("chip"),
                "machine": s.get("system", {}).get("machine"),
                "macos_version": s.get("system", {}).get("macos_version"),
                "memory_gb": s.get("system", {}).get("memory_gb"),
                "summary": s.get("summary", {}),
                "timestamp": s.get("timestamp"),
                "filename": s.get("_filename"),
            }
            for s in submissions
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(combined, f, indent=2)
        f.write("\n")
    print(f"\nSummary JSON written to: {summary_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
