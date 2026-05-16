/*
 * test_safe.c
 * ===========
 * Auto-Parallelizer Correctness and Timing Test Suite
 *
 * Exercises all eight major loop-classification outcomes of the dependency
 * analysis engine (parallelizer.py), covering the full decision space:
 *
 *   Case 1 — Element-wise vector addition     → PARALLEL
 *   Case 2 — 2D matrix addition               → PARALLEL (outer i-loop)
 *   Case 3 — Scalar accumulation (sum)        → REDUCTION
 *   Case 4 — Recurrence / loop-carried dep.   → UNSAFE
 *   Case 5 — Safe pointer arithmetic *(ptr+i) → PARALLEL
 *   Case 6 — Whitelisted math function        → PARALLEL
 *   Case 7 — Simple unit-stride while-loop    → PARALLEL (converted to for)
 *   Case 8 — Unknown user-defined function    → UNSAFE (side-effect risk)
 *
 * Cases 1–8 each run REPEAT=100 times over N=50 000 elements for lightweight
 * per-case timing.  A separate large-scale benchmark at the end of main()
 * (N_BENCH = 100 million elements, single pass) provides the primary timing
 * measurement consumed by benchmark.py — the workload is deliberately large
 * enough to dominate OpenMP thread-launch overhead and produce a stable,
 * measurable parallel speedup.
 *
 * Compile:
 *   gcc -fopenmp -O2 -o test_safe.exe test_safe.c -lm
 * Run:
 *   test_safe.exe
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 50000
#define N2D 1024
#define REPEAT 100
#define N_BENCH                                                                \
  100000000 /* 100 million — ensures >0.3s sequential, amortizes thread      \
               overhead */

/* A dummy user-defined function — unknown to the parallelizer */
double userFunc(double x) { return x * x + 1.0; }

int main(void) {
  printf("Running benchmark with N=%d, REPEAT=%d\n", N, REPEAT);
  /* ── Allocate arrays ─────────────────────────────────────────────────── */
  double *a = (double *)malloc(N * sizeof(double));
  double *b = (double *)malloc(N * sizeof(double));
  double *c = (double *)malloc(N * sizeof(double));
  double *result = (double *)malloc(N * sizeof(double));

  double **A = (double **)malloc(N2D * sizeof(double *));
  double **B = (double **)malloc(N2D * sizeof(double *));
  double **C = (double **)malloc(N2D * sizeof(double *));
  for (int i = 0; i < N2D; i++) {
    A[i] = (double *)malloc(N2D * sizeof(double));
    B[i] = (double *)malloc(N2D * sizeof(double));
    C[i] = (double *)malloc(N2D * sizeof(double));
  }

  /* Initialize 1D arrays a[], b[] with ascending/descending values;
     zero-fill result[]; populate 2D matrices A[][], B[][] and zero C[][] */
  for (int i = 0; i < N; i++) {
    a[i] = (double)(i + 1);
    b[i] = (double)(N - i);
  }
  for (int i = 0; i < N2D; i++) {
    for (int j = 0; j < N2D; j++) {
      A[i][j] = (double)(i + j + 1);
      B[i][j] = (double)(i * j + 1);
      C[i][j] = 0.0;
    }
  }

  double t_start, t_end;

  printf("=======================================================\n");
  printf("  Auto-Parallelizer Test Suite — Scaled for Timing\n");
  printf("=======================================================\n\n");

  /* Case 1: Element-wise Vector Addition — expected: PARALLEL
   * c[i] = a[i] + b[i]: every iteration writes to an independent element;
   * no cross-iteration dependency exists → safe to parallelize fully. */
  printf("Case 1: Vector Addition\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
#pragma scop
    for (int i = 0; i < N; i++) {
      c[i] = a[i] + b[i];
    }
#pragma endscop
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 2: 2D Matrix Addition — expected: PARALLEL (outer i-loop)
   * C[i][j] = A[i][j] + B[i][j]: rows are fully independent; the outer
   * i-loop is parallelized, the inner j-loop is SKIPPED per the
   * outermost-only parallelization policy. */
  printf("Case 2: Matrix Addition\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
#pragma scop
    for (int i = 0; i < N2D; i++) {
      for (int j = 0; j < N2D; j++) {
        C[i][j] = A[i][j] + B[i][j];
      }
    }
#pragma endscop
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 3: Scalar Reduction — expected: REDUCTION
   * sum += a[i]: accumulates into a scalar variable; the tool detects the
   * '+=' reduction pattern and emits:
   *   #pragma omp parallel for reduction(+:sum) */
  printf("Case 3: Reduction Sum\n");
  double sum = 0.0;
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
    sum = 0.0;
#pragma scop
    for (int i = 0; i < N; i++) {
      sum += a[i];
    }
#pragma endscop
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 4: Loop-Carried Dependency — expected: UNSAFE
   * a[i] = a[i-1] + 1: iteration i reads the value written by iteration i-1;
   * this recurrence relation cannot be executed in parallel without producing
   * incorrect results → the tool flags offset index access a[i-1] as UNSAFE. */
  printf("Case 4: Loop-Carried Dep\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
#pragma scop
    for (int i = 1; i < N; i++) {
      a[i] = a[i - 1] + 1;
    }
#pragma endscop
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 5: Safe Pointer Arithmetic — expected: PARALLEL
   * *(c+i) = *(a+i) + *(b+i): the tool recognizes the *(ptr+i) form as
   * equivalent to arr[i] indexed by the loop variable → classified as safe. */
  printf("Case 5: Pointer Arith\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
#pragma scop
    for (int i = 0; i < N; i++) {
      *(c + i) = *(a + i) + *(b + i);
    }
#pragma endscop
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 6: Whitelisted Math Function — expected: PARALLEL
   * a[i] = sqrt(b[i]): sqrt() is on the whitelist of pure, side-effect-free
   * math functions (along with sin, cos, pow, etc.) → loop is safe to parallelize. */
  printf("Case 6: Math Func\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
#pragma scop
    for (int i = 0; i < N; i++) {
      a[i] = sqrt(b[i]);
    }
#pragma endscop
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 7: Unit-Stride While-Loop — expected: PARALLEL (converted to for)
   * while (i < N) { c[i] = a[i]*b[i]; i++; }: the tool detects a unit-stride
   * induction variable with a detectable bound and rewrites this as a for-loop
   * for OpenMP compatibility before inserting the parallel pragma. */
  printf("Case 7: While-Loop\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
#pragma scop
    int i = 0;
    while (i < N) {
      c[i] = a[i] * b[i];
      i++;
    }
#pragma endscop
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 8: Unknown User-Defined Function — expected: UNSAFE
   * result[i] = userFunc(a[i]): userFunc() is not in the whitelist; without
   * its body, the tool cannot prove absence of side effects, global state
   * mutation, or non-local memory writes → conservatively marked UNSAFE. */
  printf("Case 8: Unknown Func\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
#pragma scop
    for (int i = 0; i < N; i++) {
      result[i] = userFunc(a[i]);
    }
#pragma endscop
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  free(a);
  free(b);
  free(c);
  free(result);
  for (int i = 0; i < N2D; i++) {
    free(A[i]);
    free(B[i]);
    free(C[i]);
  }
  free(A);
  free(B);
  free(C);

  /* ── Large-scale benchmark for performance measurement ── */
  double *ba = (double *)malloc(N_BENCH * sizeof(double));
  double *bb = (double *)malloc(N_BENCH * sizeof(double));
  double *bc = (double *)malloc(N_BENCH * sizeof(double));
  for (int i = 0; i < N_BENCH; i++) {
    ba[i] = i * 0.5;
    bb[i] = i * 0.3;
  }

  double t_bench = omp_get_wtime();
#pragma scop
  for (int i = 0; i < N_BENCH; i++)
    bc[i] = ba[i] + bb[i];
#pragma endscop
  t_bench = omp_get_wtime() - t_bench;
  /* Unique tag matched by benchmark.py parse_time() */
  printf("BENCHMARK_TIME: %.4f s\n", t_bench);

  free(ba);
  free(bb);
  free(bc);
  return 0;
}
