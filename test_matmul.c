/*
 * test_matmul.c
 * =============
 * Matrix Multiplication Benchmark — Auto-Parallelizer Test
 *
 * Specifications:
 *   - Matrix size: 512 x 512
 *   - Algorithm:   Standard triple-nested loop  C[i][j] += A[i][k] * B[k][j]
 *   - Timer:       omp_get_wtime() (Windows-safe)
 *   - SCoP marker: outermost loop wrapped in #pragma scop / #pragma endscop
 *   - The parallelizer will parallelize the outermost i-loop
 *
 * Compile:
 *   gcc -fopenmp -O2 -o test_matmul.exe test_matmul.c -lm
 * Run:
 *   test_matmul.exe
 *
 * NOTE — 1-thread overhead:
 *   When the auto-parallelized binary (test_matmul_auto.exe) is run with
 *   OMP_NUM_THREADS=1 its measured time will typically be HIGHER than the
 *   sequential original (test_matmul_orig.exe).  This is expected: OpenMP
 *   still initializes the full thread team at program start even with only
 *   one thread, incurring measurable startup overhead.  For this reason
 *   benchmark.py uses orig_time (the unmodified binary) as the true serial
 *   baseline for speedup and efficiency calculations — NOT auto_time at 1
 *   thread.
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define M 512

int main(void) {
  /* ── Allocate flat 2D matrices ───────────────────────────────────────── */
  double *A = (double *)malloc((size_t)M * M * sizeof(double));
  double *B = (double *)malloc((size_t)M * M * sizeof(double));
  double *C = (double *)calloc((size_t)M * M, sizeof(double));

  if (!A || !B || !C) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  /* ── Initialize A and B ─────────────────────────────────────────────── */
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      A[i * M + j] = (double)(i + j + 1) / M;
      B[i * M + j] = (double)(i * j + 1) / M;
    }
  }

  printf("=======================================================\n");
  printf("  Matrix Multiplication Benchmark\n");
  printf("  Matrix size: %d x %d\n", M, M);
  printf("=======================================================\n");

  double t_start = omp_get_wtime();

  /*
   * ── Matrix Multiplication ──────────────────────────────────────────────
   * Outermost i-loop is independent: each row of C can be computed
   * independently, making this an ideal candidate for OpenMP.
   *
   * The auto-parallelizer will detect the i-loop as SAFE and insert:
   *   #pragma omp parallel for schedule(static)
   */
#pragma scop
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < M; k++) {
        C[i * M + j] += A[i * M + k] * B[k * M + j];
      }
    }
  }
#pragma endscop

  double t_end = omp_get_wtime();

  /* ── Verify result (spot check: sum of diagonal) ─────────────────────── */
  double diag_sum = 0.0;
  for (int i = 0; i < M; i++)
    diag_sum += C[i * M + i];

  printf("BENCHMARK_TIME: %.4f s\n", t_end - t_start);
  printf("  Diag sum   : %.4f (verification)\n", diag_sum);
  printf("  C[0][0]    : %.6f\n", C[0]);
  printf("  C[M/2][M/2]: %.6f\n", C[(M / 2) * M + M / 2]);
  printf("=======================================================\n");

  free(A);
  free(B);
  free(C);
  return 0;
}
