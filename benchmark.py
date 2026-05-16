#!/usr/bin/env python3
"""
benchmark.py
============
Fully automated benchmarking and visualization script for the
Automatic Parallelization Tool project.

Phases:
  1 - Run parallelizer.py on all test C files
  2 - Compile original and parallelized versions
  3 - Benchmark at thread counts 1, 2, 4, 8
  4 - Print formatted results tables
  5 - Generate all 10 plots into FIGs/ folder

Usage:
    python benchmark.py
"""

import os
import sys
import re
import platform
import subprocess
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
IS_WINDOWS  = platform.system() == "Windows"
EXE         = ".exe" if IS_WINDOWS else ""
THREAD_COUNTS = [1, 2, 4, 8]
FIGS_DIR    = Path("FIGs")
WORK_DIR    = Path(__file__).parent

TEST_FILES = [
    {"name": "test_safe",   "label": "Vector/Matrix",   "c": "test_safe.c"},
    {"name": "test_heat",   "label": "Heat Diffusion",  "c": "test_heat.c"},
    {"name": "test_matmul", "label": "MatMul 512x512",  "c": "test_matmul.c"},
]

TIME_RE = re.compile(r"BENCHMARK_TIME:\s*([\d.]+)\s*s", re.IGNORECASE)


def parse_time(output: str) -> float:
    """Extract the first floating-point time value (seconds) from stdout."""
    m = TIME_RE.search(output)
    return float(m.group(1)) if m else -1.0


def run(cmd, env=None, cwd=None):
    """Run a shell command, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env=env, cwd=str(cwd or WORK_DIR)
    )
    return result.returncode, result.stdout, result.stderr


def banner(title: str):
    """Print a prominent section-header bar to the terminal to visually delimit benchmark phases."""
    w = 68
    print("\n+" + "=" * w + "+")
    print(f"|  {title:<{w-1}}|")
    print("+" + "=" * w + "+")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Auto-Parallelization
# ──────────────────────────────────────────────────────────────────────────────
def phase1_parallelize():
    banner("Phase 1 - Auto-Parallelization")
    decisions = {}   # name -> {parallel, reduction, unsafe}

    # Combined report accumulators
    all_entries  = []          # list of raw "[STATUS..." lines
    total_loops  = 0
    total_par    = 0
    total_red    = 0
    total_skip   = 0
    total_unsafe = 0

    # Regex to capture decision lines from verbose stdout
    entry_re = re.compile(
        r"(\[(?:PARALLEL|REDUCTION|UNSAFE|SKIPPED)[^\]]*\][^\n]*)",
        re.IGNORECASE
    )

    for tf in TEST_FILES:
        src    = WORK_DIR / tf["c"]
        out_c  = WORK_DIR / f"{tf['name']}_auto.c"
        print(f"\n  >> Analyzing {tf['c']} ...")
        # NOTE: --report is intentionally omitted here so each run does NOT
        # overwrite report.txt.  We collect everything and write one combined
        # report after the loop.
        rc, stdout, stderr = run([
            sys.executable, str(WORK_DIR / "parallelizer.py"),
            str(src), "--output", str(out_c), "--verbose"
        ])
        print(stdout)
        if stderr.strip():
            print(f"  [STDERR] {stderr.strip()}")

        # Parse decision counts from stdout
        p  = stdout.count("[PARALLEL")
        r  = stdout.count("[REDUCTION")
        u  = stdout.count("[UNSAFE")
        s  = stdout.count("[SKIPPED")
        decisions[tf["name"]] = {"parallel": p, "reduction": r, "unsafe": u}

        # Accumulate for combined report
        for m in entry_re.finditer(stdout):
            all_entries.append(f"  {m.group(1)}")
        total_par    += p
        total_red    += r
        total_unsafe += u
        total_skip   += s
        total_loops  += p + r + u + s

    # ── Write ONE combined report.txt ──────────────────────────────────────
    rate = round((total_par + total_red) / total_loops * 100) if total_loops else 0
    sep  = "=" * 80
    dash = "-" * 80
    lines = [
        sep,
        "  AUTO-PARALLELIZER -- Analysis Report (Combined)",
        sep,
    ]
    lines.extend(all_entries)
    lines += [
        dash,
        "  SUMMARY",
        dash,
        f"  Total loops analyzed     : {total_loops}",
        f"  Parallelized (parallel)  : {total_par}",
        f"  Parallelized (reduction) : {total_red}",
        f"  Skipped (inner loop)     : {total_skip}",
        f"  Rejected (unsafe)        : {total_unsafe}",
        f"  Parallelization rate     : {rate}%",
        dash,
    ]
    report_path = WORK_DIR / "report.txt"
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  [REPORT] Combined report written to: {report_path}")

    return decisions


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Compilation
# ──────────────────────────────────────────────────────────────────────────────
def phase2_compile():
    banner("Phase 2 - Compilation")
    compile_targets = []
    for tf in TEST_FILES:
        compile_targets.append({
            "src":  WORK_DIR / tf["c"],
            "out":  WORK_DIR / f"{tf['name']}_orig{EXE}",
            "label": f"{tf['name']} (original)"
        })
        auto_c = WORK_DIR / f"{tf['name']}_auto.c"
        if auto_c.exists():
            compile_targets.append({
                "src":  auto_c,
                "out":  WORK_DIR / f"{tf['name']}_auto{EXE}",
                "label": f"{tf['name']} (parallelized)"
            })

    for t in compile_targets:
        cmd = ["gcc", "-fopenmp", "-O2", "-o", str(t["out"]), str(t["src"]), "-lm"]
        rc, stdout, stderr = run(cmd)
        if rc == 0:
            print(f"  [OK]     {t['label']}")
        else:
            print(f"  [FAILED] {t['label']}\n           {stderr.strip()[:200]}")
    return compile_targets


def _run_min(exe, env, n=3):
    """
    Run exe n times and return the minimum measured time.
    Taking the minimum eliminates Windows scheduler noise, background
    process interference, and thermal throttling spikes — the minimum
    represents the cleanest, least-interrupted run.
    """
    times = []
    for _ in range(n):
        rc, stdout, _ = run([str(exe)], env=env)
        t = parse_time(stdout)
        if t > 0:
            times.append(t)
    return min(times) if times else -1.0


# ------------------------------------------------------------------------------
# Phase 3 - Performance Measurement
# ------------------------------------------------------------------------------
def phase3_benchmark():
    banner("Phase 3 - Performance Measurement")
    print(f"  Thread counts: {THREAD_COUNTS}")
    print(f"  Each binary run 3x — reporting minimum time to reduce OS noise")
    results = {}   # name -> {threads -> {orig_time, auto_time}}

    for tf in TEST_FILES:
        name = tf["name"]
        results[name] = {}
        orig_exe = WORK_DIR / f"{name}_orig{EXE}"
        auto_exe = WORK_DIR / f"{name}_auto{EXE}"

        print(f"\n  -> {tf['label']}")
        for t in THREAD_COUNTS:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(t)

            orig_time = _run_min(orig_exe, env) if orig_exe.exists() else -1.0
            auto_time = _run_min(auto_exe, env) if auto_exe.exists() else -1.0

            results[name][t] = {
                "orig_time": orig_time,
                "auto_time": auto_time,
            }
            print(f"    threads={t:>2}  orig={orig_time:7.4f}s  auto={auto_time:7.4f}s")

    # Compute speedup, efficiency, overhead
    # NOTE: serial baseline is orig_time (unmodified binary at 1 thread).
    # Using auto_time at 1 thread would be wrong: the parallelized binary
    # incurs OpenMP thread-team initialization overhead even at 1 thread.
    for name in results:
        serial = results[name][1]["orig_time"]
        for t in THREAD_COUNTS:
            pt = results[name][t]["auto_time"]
            if pt > 0 and serial > 0:
                if t == 1:
                    sp, eff, over = 1.0, 100.0, 0.0
                else:
                    sp   = serial / pt
                    eff  = min((sp / t) * 100.0, 100.0)
                    over = (pt * t) - serial
            else:
                sp, eff, over = 0.0, 0.0, 0.0
            results[name][t]["speedup"]    = sp
            results[name][t]["efficiency"] = eff
            results[name][t]["overhead"]   = over

    return results


# ------------------------------------------------------------------------------
# Phase 4 - Results Tables
# ------------------------------------------------------------------------------
def phase4_tables(results, test_files):
    banner("Phase 4 - Results Tables")
    for tf in test_files:
        name  = tf["name"]
        label = tf["label"]
        r     = results[name]

        print(f"\n+{'='*68}+")
        print(f"|  Benchmark: {label:<55}|")
        print(f"+{'='*10}+{'='*11}+{'='*11}+{'='*14}+{'='*15}+")
        print(f"|{'Threads':^10}|{'Time(s)':^11}|{'Speedup':^11}|{'Efficiency':^14}|{'Overhead(s)':^15}|")
        print(f"+{'='*10}+{'='*11}+{'='*11}+{'='*14}+{'='*15}+")

        for t in THREAD_COUNTS:
            d    = r[t]
            # Show orig_time for the sequential row so readers see the true baseline
            time = d["orig_time"] if t == 1 else d["auto_time"]
            sp   = d.get("speedup", 0.0)
            eff  = d.get("efficiency", 0.0)
            ov   = d.get("overhead", 0.0)
            tag  = "1 (seq)" if t == 1 else str(t)
            print(f"|{tag:^10}|{time:^11.4f}|{sp:^10.2f}x|{eff:^13.1f}%|{ov:^15.3f}|")

        print(f"+{'='*10}+{'='*11}+{'='*11}+{'='*14}+{'='*15}+")


# ------------------------------------------------------------------------------
# Phase 5 - All 10 Plots
# ------------------------------------------------------------------------------

COLORS = ["royalblue", "darkorange", "green"]
LABELS = [tf["label"] for tf in TEST_FILES]
NAMES  = [tf["name"]  for tf in TEST_FILES]


def _save(fig, filename: str):
    """Save a matplotlib figure to FIGs/<filename> at 150 dpi, then close it to release memory."""
    FIGS_DIR.mkdir(exist_ok=True)
    path = FIGS_DIR / filename
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] FIGs/{filename}")


def plot1_speedup(results):
    """Plot 1 - Speedup vs Thread Count (all benchmarks + ideal)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(THREAD_COUNTS, THREAD_COUNTS, "k--", alpha=0.5, label="Ideal Linear Speedup")

    for i, name in enumerate(NAMES):
        speeds = [results[name][t].get("speedup", 0) for t in THREAD_COUNTS]
        ax.plot(THREAD_COUNTS, speeds, "o-", color=COLORS[i],
                linewidth=2, markersize=7, label=LABELS[i])
        peak = max(speeds)
        peak_t = THREAD_COUNTS[speeds.index(peak)]
        ax.annotate(f"{peak:.2f}x", xy=(peak_t, peak),
                    xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")

    ax.set_title("Speedup vs. Number of Threads", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speedup")
    ax.set_xticks(THREAD_COUNTS)
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "speedup_all.png")


def plot2_efficiency(results):
    """Plot 2 - Parallel Efficiency vs Thread Count."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(100, color="k", linestyle="--", alpha=0.5, label="Ideal 100% Efficiency")

    for i, name in enumerate(NAMES):
        effs = [results[name][t].get("efficiency", 0) for t in THREAD_COUNTS]
        ax.plot(THREAD_COUNTS, effs, "o-", color=COLORS[i],
                linewidth=2, markersize=7, label=LABELS[i])
        ax.fill_between(THREAD_COUNTS, effs, 100, alpha=0.08, color=COLORS[i])

    ax.set_title("Parallel Efficiency vs. Number of Threads", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Efficiency (%)")
    ax.set_ylim(0, 115)
    ax.set_xticks(THREAD_COUNTS)
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "efficiency_all.png")


def plot3_amdahl(results):
    """Plot 3 - Actual Speedup vs Amdahl's Law (heat benchmark)."""
    name   = "test_heat"
    # Use orig_time as the true serial baseline for speedup calculation
    serial = results[name][1]["orig_time"]
    speeds = []
    for t in THREAD_COUNTS:
        pt = results[name][t]["auto_time"]
        speeds.append(serial / pt if pt > 0 and serial > 0 else 1.0)

    # Estimate serial fraction f from 2-thread measurement using Amdahl inverse
    p2, sp2 = 2, speeds[1]
    f = ((1/sp2) - (1/p2)) / (1 - 1/p2) if sp2 > 0 else 0.1
    f = max(0.001, min(f, 0.99))  # clamp to valid range

    thread_range = np.linspace(1, 8, 100)
    amdahl_pred  = [1 / (f + (1 - f) / p) for p in thread_range]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(THREAD_COUNTS, speeds, "o-", color="darkorange",
            linewidth=2, markersize=8, label="Actual Speedup (measured)")
    ax.plot(thread_range, amdahl_pred, "b--",
            linewidth=2, label=f"Amdahl's Law (f={f:.3f})")
    ax.annotate(f"Serial fraction f approx {f:.3f}",
                xy=(4, 1 / (f + (1 - f) / 4)),
                xytext=(4.5, 1.2), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_title("Actual Speedup vs. Amdahl's Law Theoretical Prediction",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speedup")
    ax.set_xticks(THREAD_COUNTS)
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "amdahls_law.png")


def plot4_time_bars(results):
    """Plot 4 - Grouped execution time bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    n_groups  = len(NAMES)
    n_bars    = len(THREAD_COUNTS)
    width     = 0.18
    x         = np.arange(n_groups)
    bar_colors= ["#555555", "#4472C4", "#ED7D31", "#A9D18E"]

    for b_idx, t in enumerate(THREAD_COUNTS):
        times = [results[name][t].get("auto_time", 0) for name in NAMES]
        bars  = ax.bar(x + b_idx * width - width * 1.5, times,
                       width, label=f"{t} thread{'s' if t>1 else ''}",
                       color=bar_colors[b_idx], edgecolor="black", linewidth=0.5)
        for bar, tm, sp in zip(bars, times, [results[n][t].get("speedup", 1) for n in NAMES]):
            if tm > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{sp:.2f}x", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_title("Execution Time Comparison Across All Benchmarks",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "execution_time_bars.png")


def plot5_overhead(results):
    """Plot 5 - Parallel Overhead vs Thread Count."""
    fig, ax = plt.subplots(figsize=(9, 6))
    t_range = THREAD_COUNTS[1:]  # overhead only meaningful for >1 thread

    for i, name in enumerate(NAMES):
        overheads = [results[name][t].get("overhead", 0) for t in t_range]
        ax.plot(t_range, overheads, "o-", color=COLORS[i],
                linewidth=2, markersize=7, label=LABELS[i])

    ax.set_title("Parallel Synchronization and Thread Management Overhead",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Overhead (seconds)")
    ax.set_xticks(t_range)
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "parallel_overhead.png")


def plot6_tool_decisions(decisions):
    """Plot 6 - Tool decision breakdown (bar + pie)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    file_labels = [tf["label"] for tf in TEST_FILES]
    safe_counts = [decisions.get(n, {}).get("parallel", 0)  for n in NAMES]
    red_counts  = [decisions.get(n, {}).get("reduction", 0) for n in NAMES]
    uns_counts  = [decisions.get(n, {}).get("unsafe", 0)    for n in NAMES]

    y = np.arange(len(file_labels))
    h = 0.25
    ax1.barh(y + h,  safe_counts, h, label="SAFE (parallel)", color="seagreen")
    ax1.barh(y,      red_counts,  h, label="REDUCTION",       color="steelblue")
    ax1.barh(y - h,  uns_counts,  h, label="UNSAFE",          color="tomato")
    ax1.set_yticks(y); ax1.set_yticklabels(file_labels)
    ax1.set_xlabel("Loop Count")
    ax1.set_title("Loop Classifications per File", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3, axis="x")

    totals = [sum(safe_counts), sum(red_counts), sum(uns_counts)]
    pie_labels = [f"SAFE\n({totals[0]})", f"REDUCTION\n({totals[1]})", f"UNSAFE\n({totals[2]})"]
    ax2.pie(totals, labels=pie_labels, colors=["seagreen", "steelblue", "tomato"],
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    ax2.set_title("All Files Combined", fontweight="bold")

    fig.suptitle("Automatic Parallelizer - Loop Classification Results",
                 fontsize=14, fontweight="bold")
    _save(fig, "tool_decisions.png")


def plot7_heatmap():
    """Plot 7 - 2D temperature heatmap from heat_output.bin."""
    path = WORK_DIR / "heat_output.bin"
    if not path.exists():
        print("  [SKIP] heat_output.bin not found - run test_heat first")
        return
    with open(path, "rb") as f:
        n    = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(n * n * 8), dtype=np.float64).reshape((n, n))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap="inferno", interpolation="bilinear", origin="upper")
    plt.colorbar(im, ax=ax, label="Temperature (C)")
    ax.set_title("2D Heat Diffusion - Final Temperature Distribution (1024x1024)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Grid Column (j)")
    ax.set_ylabel("Grid Row (i)")
    _save(fig, "heat_map.png")


def plot8_thread_workload():
    """Plot 8 - Thread workload map from thread_map.bin."""
    path = WORK_DIR / "thread_map.bin"
    if not path.exists():
        print("  [SKIP] thread_map.bin not found - run test_heat first")
        return
    with open(path, "rb") as f:
        n    = struct.unpack("i", f.read(4))[0]
        tmap = np.frombuffer(f.read(n * 4), dtype=np.int32)

    # Expand row-thread assignment to full 2D image
    img = np.tile(tmap.reshape(-1, 1), (1, n))
    n_threads = int(tmap.max()) + 1

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(img, cmap="tab10", origin="upper",
                   vmin=0, vmax=n_threads - 1)
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_threads))
    cbar.set_label("Thread ID")
    cbar.set_ticklabels([f"Thread {i}" for i in range(n_threads)])
    ax.set_title("OpenMP Thread Workload Distribution - Heat Diffusion Benchmark",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    _save(fig, "thread_workload_map.png")


def plot9_scalability(results):
    """Plot 9 - Normalized execution time (lower = better)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Ideal (1.0)")

    best_name, best_val = None, 1.0
    for i, name in enumerate(NAMES):
        serial = results[name][1].get("orig_time", 1)
        if serial <= 0: serial = 1
        norm = [results[name][t].get("auto_time", serial) / serial for t in THREAD_COUNTS]
        ax.plot(THREAD_COUNTS, norm, "o-", color=COLORS[i],
                linewidth=2, markersize=7, label=LABELS[i])
        if norm[-1] < best_val:
            best_val, best_name = norm[-1], (i, LABELS[i])

    if best_name:
        bi, bl = best_name
        bval   = [results[NAMES[bi]][t].get("auto_time", 1) /
                  max(results[NAMES[bi]][1].get("auto_time", 1), 1e-9)
                  for t in THREAD_COUNTS]
        ax.annotate(f"Best: {bl}",
                    xy=(8, bval[-1]), xytext=(6.5, bval[-1] + 0.08),
                    arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

    ax.set_title("Normalized Execution Time - Scalability Comparison Across Benchmarks",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Normalized Time (T_parallel / T_serial)")
    ax.set_xticks(THREAD_COUNTS)
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "scalability_comparison.png")


def plot10_code_transform():
    """Plot 10 - Side-by-side code transformation panel."""
    before = (
        "/* Sequential - no pragma */\n"
        "for (int i = 0; i < N; i++) {\n"
        "    c[i] = a[i] + b[i];\n"
        "}"
    )
    after = (
        "#pragma omp parallel for\n"
        "    schedule(static)\n"
        "/* AUTO-INSERTED BY TOOL */\n"
        "for (int i = 0; i < N; i++) {\n"
        "    c[i] = a[i] + b[i];\n"
        "}"
    )

    fig, (ax1, ax_arr, ax2) = plt.subplots(1, 3, figsize=(14, 5),
                                            gridspec_kw={"width_ratios": [5, 1, 5]})

    # Left panel
    ax1.set_facecolor("#f0f0f0")
    ax1.text(0.5, 0.5, before, transform=ax1.transAxes,
             fontfamily="monospace", fontsize=11, va="center", ha="center",
             wrap=True)
    ax1.set_title("BEFORE - Sequential", fontsize=13, fontweight="bold", color="#333333")
    ax1.axis("off")

    # Arrow panel
    ax_arr.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->, head_width=0.4",
                                   color="black", lw=2))
    ax_arr.axis("off")

    # Right panel - highlight pragma line in green
    ax2.set_facecolor("#eaffea")
    lines = after.split("\n")
    line_colors = ["#006600" if "#pragma" in l or "schedule" in l or "AUTO" in l
                   else "#222222" for l in lines]
    y_start = 0.85
    for li, (line, color) in enumerate(zip(lines, line_colors)):
        weight = "bold" if color == "#006600" else "normal"
        ax2.text(0.05, y_start - li * 0.13, line, transform=ax2.transAxes,
                 fontfamily="monospace", fontsize=11, color=color, fontweight=weight)
    ax2.set_title("AFTER - Auto-Parallelized", fontsize=13, fontweight="bold", color="#006600")
    ax2.axis("off")

    fig.suptitle("Automatic Code Transformation - Tool Input vs. Output",
                 fontsize=14, fontweight="bold", y=1.02)
    _save(fig, "code_transformation.png")


def phase5_plots(results, decisions):
    banner("Phase 5 - Generating All 10 Plots")
    FIGS_DIR.mkdir(exist_ok=True)
    plot1_speedup(results)
    plot2_efficiency(results)
    plot3_amdahl(results)
    plot4_time_bars(results)
    plot5_overhead(results)
    plot6_tool_decisions(decisions)
    plot7_heatmap()
    plot8_thread_workload()
    plot9_scalability(results)
    plot10_code_transform()
    print(f"\n  All plots saved to: {FIGS_DIR.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  AUTO-PARALLELIZER - Full Benchmark Suite")
    print("=" * 70)

    decisions = phase1_parallelize()
    phase2_compile()
    results   = phase3_benchmark()
    phase4_tables(results, TEST_FILES)
    phase5_plots(results, decisions)

    banner("Complete")
    print("  All phases finished. Check FIGs/ for all plots.\n")


if __name__ == "__main__":
    main()
