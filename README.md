# Automatic Parallelization Tool for C Loops Using OpenMP

> **Course Project — Parallel Computing**
> **Author:** Hossamaldeen Elghazaly — Nile University, ITCS
> **Title:** *A Compiler-Inspired Automatic Parallelization Tool for Affine Loop Nests Using OpenMP*

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Architecture](#project-architecture)
4. [Pipeline Diagram](#pipeline-diagram)
5. [File Structure](#file-structure)
6. [Requirements](#requirements)
7. [Installation](#installation)
8. [Usage](#usage)
   - [Running the Parallelizer](#running-the-parallelizer)
   - [Running the Full Benchmark Suite](#running-the-full-benchmark-suite)
   - [Using the Makefile](#using-the-makefile)
9. [Dependency Analysis — How It Works](#dependency-analysis--how-it-works)
10. [Benchmark Test Cases](#benchmark-test-cases)
11. [Output Files Reference](#output-files-reference)
12. [Generated Plots](#generated-plots)
13. [Known Limitations](#known-limitations)
14. [Paper Alignment](#paper-alignment)
15. [A Note on AI-Assisted Documentation](#a-note-on-ai-assisted-documentation)

---

## Overview

This project implements a **compiler-inspired, source-to-source automatic parallelization tool** written in Python. The tool reads sequential C source code, performs static dependency analysis on annotated loop regions, and automatically rewrites those loops with the appropriate OpenMP parallelization pragmas — without any manual programmer intervention.

The tool is designed around the concept of a **Static Control Part (SCoP)**: a programmer-annotated region of code whose control flow and memory access patterns are statically predictable. Within each SCoP, the tool identifies loops, classifies each one as `PARALLEL`, `REDUCTION`, `SKIPPED` (inner loop), or `UNSAFE`, and generates a fully transformed C file alongside a human-readable analysis report.

The accompanying benchmarking script (`benchmark.py`) automates the full evaluation pipeline: it invokes the parallelizer, compiles both the original and transformed binaries, measures execution time across 1, 2, 4, and 8 threads, computes speedup and parallel efficiency, and generates 10 publication-quality figures.

The fundamental motivation is that a large class of scientific and numerical loops — heat diffusion stencils, matrix multiplication, vector operations — are **embarrassingly parallel**: each iteration operates on independent data, making parallelization both safe and highly beneficial. This tool detects and proves those cases automatically.

---

## Key Features

- **SCoP-Region Detection** — Scans for `#pragma scop` / `#pragma endscop` markers to identify the analysis target regions, ignoring all code outside them.
- **For-Loop and While-Loop Detection** — Extracts and normalizes both `for` and supported `while` loop forms within a SCoP.
- **Eight-Stage Dependency Analysis** — Applies a prioritized sequence of static checks to classify each loop, including:
  - Dynamic memory allocation detection (`malloc`, `calloc`, `realloc`, `free`)
  - Unknown and potentially side-effectful function call detection
  - Loop-carried dependency detection via offset array indexing (`a[i-1]`, `a[i+1]`)
  - Write-after-write and read-after-write conflict detection
  - Reduction pattern recognition (`+=`, `-=`, `*=`)
  - Safe pointer arithmetic recognition (`*(ptr + i)`)
- **OpenMP Pragma Injection** — Inserts `#pragma omp parallel for schedule(static)` for safe loops and `#pragma omp parallel for reduction(op:var)` for reduction loops.
- **Outermost-Only Parallelization Policy** — Only the outermost loop in a nest is parallelized, avoiding incorrect nested parallelism.
- **While-to-For Loop Rewriting** — Eligible `while` loops with a detectable induction variable are converted to `for` loop form, which is required for OpenMP compatibility.
- **Additive, Deduplicated Report Generation** — The `report.txt` accumulates results across multiple input files without duplicating entries, giving a single combined analysis summary.
- **Full Benchmark Automation** — `benchmark.py` orchestrates parallelization, compilation, multi-thread timing (3 runs minimum per measurement), results table printing, and 10 performance figures.

---

## Project Architecture

The project is composed of two main Python modules and three C benchmark programs:

| Component | Role |
|---|---|
| `parallelizer.py` | Core analysis engine — SCoP detection, loop extraction, dependency analysis, pragma injection, report generation |
| `benchmark.py` | End-to-end automation — parallelizes all test files, compiles, benchmarks across thread counts, and produces all figures |
| `test_safe.c` | Eight-case correctness and timing test covering the full range of loop classifications |
| `test_heat.c` | 2D heat diffusion Jacobi solver (1024×1024 grid, 5000 iterations) — primary scalability benchmark |
| `test_matmul.c` | Standard triple-nested matrix multiplication (512×512) — compute-bound parallel workload |
| `Makefile` | Build system with targets for compilation, parallelization, benchmarking, and cleanup |

---

## Pipeline Diagram

```
Input: Sequential C file (with #pragma scop markers)
         │
         ▼
 ┌─────────────────────────┐
 │ Step 1: SCoP Detection  │  Locate #pragma scop / #pragma endscop regions
 └────────────┬────────────┘
              │
              ▼
 ┌─────────────────────────┐
 │ Step 2: Loop Detection  │  Extract for-loops and supported while-loops
 └────────────┬────────────┘
              │
              ▼
 ┌──────────────────────────────┐
 │ Step 3: Dependency Analysis  │  8 prioritized checks per loop
 └─────────────┬────────────────┘
               │
       ┌───────┴────────────────────┐
       │                            │
       ▼                            ▼
  [PARALLEL / REDUCTION]        [UNSAFE / SKIPPED]
       │                            │
       ▼                            ▼
 ┌──────────────────────────────────────────────────┐
 │ Step 4: Parallelization Decision                 │
 │  - Insert #pragma omp parallel for (safe loops)  │
 │  - Insert explanatory comment (unsafe loops)     │
 └─────────────┬────────────────────────────────────┘
               │
               ▼
 ┌──────────────────────────────┐
 │ Step 5: Output Code Gen      │  Write transformed .c file; strip SCoP markers
 └─────────────┬────────────────┘
               │
               ▼
 ┌──────────────────────────────┐
 │ Step 6: Report Generation    │  Write combined report.txt
 └─────────────┬────────────────┘
               │
               ▼
 ┌──────────────────────────────┐
 │ Step 7: Benchmark & Plots    │  benchmark.py → compile → time → 10 figures
 └──────────────────────────────┘
```

---

## File Structure

```
parallel project/
├── parallelizer.py         ← Main auto-parallelization engine
├── benchmark.py            ← Full benchmark pipeline + 10-plot generator
├── test_safe.c             ← 8-case loop classifier and timing test
├── test_heat.c             ← 2D heat diffusion Jacobi solver (1024×1024)
├── test_matmul.c           ← 512×512 matrix multiplication benchmark
├── Makefile                ← Build, run, and clean targets
├── README.md               ← This file
│
│   (Generated by the tool — not committed to source control)
├── test_safe_auto.c        ← Parallelized version of test_safe.c
├── test_heat_auto.c        ← Parallelized version of test_heat.c
├── test_matmul_auto.c      ← Parallelized version of test_matmul.c
├── report.txt              ← Combined per-loop analysis report
├── heat_output.bin         ← Raw 2D temperature grid (binary, float64)
├── thread_map.bin          ← Thread-to-row workload assignment (binary, int32)
│
└── FIGs/                   ← All 10 generated benchmark figures
    ├── speedup_all.png
    ├── efficiency_all.png
    ├── amdahls_law.png
    ├── execution_time_bars.png
    ├── parallel_overhead.png
    ├── tool_decisions.png
    ├── heat_map.png
    ├── thread_workload_map.png
    ├── scalability_comparison.png
    └── code_transformation.png
```

---

## Requirements

| Requirement | Version / Notes |
|---|---|
| Python | 3.8 or higher |
| GCC | Any version with `-fopenmp` support |
| `numpy` (Python) | Any recent version |
| `matplotlib` (Python) | 3.x or higher |
| Platform | Windows 10/11 or Linux |

> **Windows users:** It is strongly recommended to use [MSYS2](https://www.msys2.org/) with the MinGW-w64 GCC toolchain for OpenMP support.

---

## Installation

**Step 1 — Clone or download the project folder:**

```bash
git clone <repository-url>
cd "parallel project"
```

**Step 2 — Install Python dependencies:**

```bash
pip install numpy matplotlib
```

**Step 3 — Verify GCC with OpenMP support is available:**

```bash
# Linux / MSYS2
gcc --version
gcc -fopenmp -E - < /dev/null

# Windows (MSYS2 / MinGW-w64 install)
pacman -S mingw-w64-x86_64-gcc
```

---

## Usage

### Running the Parallelizer

The parallelizer operates on any C file that contains `#pragma scop` / `#pragma endscop` annotation markers around the loops to be analyzed.

```bash
# Basic usage — analyzes test_safe.c and produces test_safe_auto.c
python parallelizer.py test_safe.c --verbose

# Full pipeline: custom output file + report + verbose logging
python parallelizer.py test_heat.c --output test_heat_parallel.c --report --verbose

# Quiet mode — produce output and report without terminal output
python parallelizer.py test_matmul.c --output test_matmul_parallel.c --report
```

**CLI Flags:**

| Flag | Description |
|---|---|
| `input_file` | *(positional)* Path to the sequential C source file to analyze |
| `--output`, `-o` | Output file name (default: `<input_stem>_auto.c` in the same directory) |
| `--report` | Generate or update `report.txt` with the combined per-loop analysis |
| `--verbose`, `-v` | Print detailed step-by-step analysis decisions to the terminal |

**Example terminal output (verbose mode):**

```
+========================================================+
|   AUTO-PARALLELIZER -- Analyzing: test_heat.c          |
+========================================================+

  [PARALLEL  ] Line 81 -> for (int i = 1; i < N - 1; i++)  -> #pragma omp parallel for inserted
  [SKIPPED   ] Line 82 -> inner loop (depth 2) -> outermost-only parallelization policy

  --------------------------------------------------------
  SUMMARY: 1 parallelized (1 parallel + 0 reduction)
           1 skipped (inner loops)
           0 rejected
           100.0% of loops successfully parallelized
  --------------------------------------------------------

  [DONE] Output written to: test_heat_auto.c
  [DONE] Report written to: report.txt
```

---

### Running the Full Benchmark Suite

```bash
# Runs all 5 phases automatically: parallelize → compile → benchmark → tables → 10 plots
python benchmark.py
```

The benchmark suite executes the following phases in sequence:

| Phase | Description |
|---|---|
| **Phase 1** | Runs `parallelizer.py` on all three test files and writes a combined `report.txt` |
| **Phase 2** | Compiles original and parallelized binaries with GCC (`-fopenmp -O2`) |
| **Phase 3** | Benchmarks each binary at 1, 2, 4, and 8 threads (3 runs per configuration, reporting the minimum) |
| **Phase 4** | Prints formatted results tables with time, speedup, efficiency, and overhead columns |
| **Phase 5** | Generates all 10 performance and visualization figures into the `FIGs/` directory |

> **Measurement methodology:** Each binary is executed three times per thread count, and the **minimum** measured time is recorded. This approach eliminates noise from Windows scheduler interference, background process contention, and transient thermal throttling.

> **Serial baseline:** Speedup and efficiency are computed using the **original, unmodified binary** (`_orig.exe`) at one thread — not the parallelized binary. The parallelized binary incurs OpenMP thread-team initialization overhead even when running with a single thread, which would artificially inflate the apparent baseline and distort all derived metrics.

---

### Using the Makefile

```bash
make all           # Compile all original and parallelized binaries
make parallelize   # Run parallelizer.py on all three test files
make run_original  # Run all sequential (original) versions
make run_parallel  # Run all parallelized versions with 8 threads (OMP_NUM_THREADS=8)
make benchmark     # Execute the complete benchmark.py pipeline
make clean         # Remove all generated binaries, C files, and data files
```

---

## Dependency Analysis — How It Works

The dependency analysis engine (`analyze_dependencies`) applies eight checks to each loop body in strict priority order. The first check that fails immediately classifies the loop as `UNSAFE` and returns. If all checks pass, the loop is classified as `PARALLEL`.

| Priority | Check | Classification on Failure |
|---|---|---|
| 1 | Dynamic memory call inside body (`malloc`, `calloc`, `realloc`, `free`) | `UNSAFE` |
| 2 | Possible recursive call (heuristic: function name matches source filename) | `UNSAFE` |
| 3 | Unknown (non-whitelisted) function call — cannot prove absence of side effects | `UNSAFE` |
| 4 | Unsupported pointer patterns: raw `*ptr` dereference or struct `->` access | `UNSAFE` |
| 5 | Reduction pattern detected (`sum +=`, `product *=` on a scalar variable) | `REDUCTION` ✓ |
| 6 | Offset array indexing: `a[i-1]`, `a[i+1]` — indicates loop-carried dependency | `UNSAFE` |
| 7 | Write-after-read conflict: same array accessed with different index expressions | `UNSAFE` |
| 8 | Compound array assignment `arr[expr] += ...` where `expr` does not contain loop variable | `UNSAFE` |
| — | All checks passed — iterations are fully independent | `PARALLEL` ✓ |

**Whitelisted math functions** (safe inside parallel loops, no side effects):
`sqrt`, `abs`, `fabs`, `sin`, `cos`, `tan`, `exp`, `log`, `log2`, `log10`, `pow`, `floor`, `ceil`, `round`, `fmin`, `fmax`

**Safe pointer arithmetic pattern** (the only supported form):
```c
*(ptr + i)   // where i is the loop induction variable — treated as arr[i]
```

---

## Benchmark Test Cases

### `test_safe.c` — Eight-Case Classification Suite

This file exercises all eight classification outcomes of the dependency analysis engine across a suite of representative loop patterns. Each case is timed independently to demonstrate the overhead difference between serial and parallelized execution on a workload of `N = 50,000` elements repeated `REPEAT = 100` times. A separate large-scale benchmark (`N_BENCH = 100,000,000`) provides the primary timing measurement used by `benchmark.py`, ensuring the workload is large enough to dominate thread-launch overhead.

| Case | Pattern | Expected Classification |
|---|---|---|
| 1 | `c[i] = a[i] + b[i]` — element-wise vector addition | `PARALLEL` |
| 2 | `C[i][j] = A[i][j] + B[i][j]` — 2D matrix addition | `PARALLEL` (outer loop) |
| 3 | `sum += a[i]` — scalar accumulation | `REDUCTION` |
| 4 | `a[i] = a[i-1] + 1` — recurrence relation | `UNSAFE` (loop-carried dep.) |
| 5 | `*(c+i) = *(a+i) + *(b+i)` — safe pointer arithmetic | `PARALLEL` |
| 6 | `a[i] = sqrt(b[i])` — whitelisted math function | `PARALLEL` |
| 7 | `while (i < N) { c[i] = a[i]*b[i]; i++; }` — simple while-loop | `PARALLEL` (converted to for) |
| 8 | `result[i] = userFunc(a[i])` — unknown user-defined function | `UNSAFE` (side-effect risk) |

### `test_heat.c` — 2D Heat Diffusion Jacobi Solver

Simulates 2D heat diffusion on a 1024×1024 grid using the Jacobi iterative method for up to 5000 iterations. Boundary conditions are fixed (top edge: 100°C hot, remaining edges: 0°C cold). The inner Jacobi update loop is the primary SCoP target: each row `i` is fully independent of other rows, making this an ideal parallel workload. The binary also saves a full 2D temperature grid (`heat_output.bin`) and a thread-to-row assignment map (`thread_map.bin`) for visualization in Plots 7 and 8.

### `test_matmul.c` — Matrix Multiplication (512×512)

Implements the standard triple-nested loop matrix multiplication `C[i][j] += A[i][k] * B[k][j]` on 512×512 flat-allocated matrices. The outermost `i`-loop (iterating over rows of C) is fully independent: each thread computes its assigned rows of the result matrix without accessing shared write targets. The tool classifies the `i`-loop as `PARALLEL` and inserts `#pragma omp parallel for schedule(static)`. The inner `j` and `k` loops are classified as `SKIPPED` per the outermost-only parallelization policy.

**Note on single-thread overhead:** When the parallelized binary (`test_matmul_auto.exe`) is measured at `OMP_NUM_THREADS=1`, its runtime is typically *higher* than the original sequential binary. This is expected: OpenMP still initializes the full thread team at program start even when only one thread is used, incurring measurable startup overhead. The benchmark script accounts for this by using the unmodified original binary as the serial baseline.

---

## Output Files Reference

| File | Produced By | Description |
|---|---|---|
| `test_safe_auto.c` | `parallelizer.py` | Parallelized version of `test_safe.c` with OpenMP pragmas injected |
| `test_heat_auto.c` | `parallelizer.py` | Parallelized version of `test_heat.c` |
| `test_matmul_auto.c` | `parallelizer.py` | Parallelized version of `test_matmul.c` |
| `report.txt` | `parallelizer.py --report` | Combined per-loop analysis report with classification tags, reasons, and summary statistics |
| `heat_output.bin` | `test_heat` execution | Raw binary dump of the 1024×1024 temperature grid (float64, row-major) |
| `thread_map.bin` | `test_heat` execution | Row-to-thread assignment array (int32, length 1024) |

---

## Generated Plots

All 10 figures are written to the `FIGs/` directory by `benchmark.py`:

| File | Description |
|---|---|
| `speedup_all.png` | Speedup vs. thread count for all three benchmarks, with ideal linear speedup reference |
| `efficiency_all.png` | Parallel efficiency (%) vs. thread count; shaded area shows deviation from ideal |
| `amdahls_law.png` | Measured speedup for the heat benchmark overlaid with Amdahl's Law theoretical prediction; serial fraction `f` estimated from the 2-thread measurement |
| `execution_time_bars.png` | Grouped bar chart of absolute execution times across all benchmarks and thread counts, annotated with speedup values |
| `parallel_overhead.png` | Thread synchronization and management overhead (seconds) vs. thread count |
| `tool_decisions.png` | Side-by-side bar chart and pie chart showing loop classification counts (`PARALLEL`, `REDUCTION`, `UNSAFE`) per input file |
| `heat_map.png` | 2D false-color visualization of the final temperature distribution from the heat diffusion benchmark |
| `thread_workload_map.png` | Visual map of which OpenMP thread processed each row of the heat diffusion grid, demonstrating workload balance |
| `scalability_comparison.png` | Normalized execution time (`T_parallel / T_serial`) vs. thread count; lower values indicate better scalability |
| `code_transformation.png` | Side-by-side panel showing a representative before-and-after code transformation: sequential loop vs. auto-parallelized version |

---

## Known Limitations

| Limitation | Technical Reason |
|---|---|
| **Recursion not fully detected** | Accurate recursion detection requires a complete interprocedural call graph. The tool uses a heuristic (matching function name against the source filename), which may miss some cases and cannot handle mutual recursion. |
| **Dynamic memory inside loops is always rejected** | `malloc`/`calloc` calls require heap ownership and aliasing analysis to determine safety. Static analysis cannot reliably perform this without pointer provenance information. |
| **Unknown function calls are conservatively rejected** | Without the function body or a formal specification, the tool cannot prove the absence of side effects, global state mutation, or non-local memory writes. |
| **Complex pointer aliasing is not supported** | Full alias analysis (e.g., Andersen's or Steensgaard's algorithms) is required to handle general pointer patterns beyond the simple `*(ptr + i)` form. |
| **Data-dependent while-loop bounds** | If the loop termination condition depends on runtime data values, the iteration count is unknown at compile time, preventing safe static analysis. |
| **Indirect array indexing** | Patterns such as `a[b[i]]` (scatter/gather access) create memory access sets that cannot be statically determined without value range analysis. |
| **Regex-based parsing** | The tool uses regular expressions rather than a full C AST parser. Highly unusual formatting, preprocessor macros, or complex expression nesting may cause incorrect loop detection. |

---

## Paper Alignment

This tool and its benchmark suite are designed to support a structured academic report. The following table maps each report section to the corresponding tool output or artifact:

| Paper Section | Corresponding Tool Output |
|---|---|
| Title & Abstract | Benchmark speedup numbers, parallelization rate from `report.txt` |
| Introduction | `code_transformation.png`, problem motivation |
| Related Work | Comparison context: TC Compiler, LLVM Polly, GCC auto-parallelization, OpenMP |
| Proposed Solution | Pipeline diagram, `report.txt`, verbose terminal output |
| Evaluation & Results | Plots 1–9, formatted benchmark results tables (Phase 4 output) |
| Conclusion | Known limitations section, overhead analysis (`parallel_overhead.png`, Plot 5) |

---

## A Note on AI-Assisted Documentation

The inline source code comments throughout this project — including function docstrings, section headers, inline explanations, and algorithmic notes in `parallelizer.py`, `benchmark.py`, `test_heat.c`, `test_matmul.c`, and `test_safe.c` — were refined and structured with the assistance of an AI language model. This was done to ensure that every design decision, algorithmic step, and known behavioral nuance (such as OpenMP thread-team initialization overhead at single-thread execution) is clearly explained within the codebase itself, making the project accessible and self-documenting for reviewers, collaborators, and future maintainers. This README was similarly produced with AI assistance to achieve a professional, thorough, and well-organized standard of documentation.
