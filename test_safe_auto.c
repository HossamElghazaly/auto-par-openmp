/*
 * test_safe.c
 * ===========
 * Scaled-up version for measurable timing.
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

  /* Initialize */
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

  /* Case 1: Vector Addition */
  printf("Case 1: Vector Addition\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
      c[i] = a[i] + b[i];
    }
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 2: Matrix Addition */
  printf("Case 2: Matrix Addition\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N2D; i++) {
      for (int j = 0; j < N2D; j++) {
        C[i][j] = A[i][j] + B[i][j];
      }
    }
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 3: Reduction Sum */
  printf("Case 3: Reduction Sum\n");
  double sum = 0.0;
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
    sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
      sum += a[i];
    }
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 4: Loop-Carried Dependency */
  printf("Case 4: Loop-Carried Dep\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
    /* AUTO-PARALLELIZER: loop NOT parallelized - loop-carried dependency: offset index access 'a[i - 1]' */
    for (int i = 1; i < N; i++) {
      a[i] = a[i - 1] + 1;
    }
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 5: Pointer Arithmetic */
  printf("Case 5: Pointer Arith\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
      *(c + i) = *(a + i) + *(b + i);
    }
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 6: Math Func */
  printf("Case 6: Math Func\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
      a[i] = sqrt(b[i]);
    }
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 7: While-Loop */
  printf("Case 7: While-Loop\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
      c[i] = a[i] * b[i];
        // i++;
    }
  }
  t_end = omp_get_wtime();
  printf("  Time: %.6f s (avg)\n\n", (t_end - t_start) / REPEAT);

  /* Case 8: Unknown Func */
  printf("Case 8: Unknown Func\n");
  t_start = omp_get_wtime();
  for (int r = 0; r < REPEAT; r++) {
    /* AUTO-PARALLELIZER: loop NOT parallelized - unknown function call 'userFunc()' — possible side effects */
    for (int i = 0; i < N; i++) {
      result[i] = userFunc(a[i]);
    }
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
  for (int i = 0; i < N_BENCH; i++)
    bc[i] = ba[i] + bb[i];
  t_bench = omp_get_wtime() - t_bench;
  /* Unique tag matched by benchmark.py parse_time() */
  printf("BENCHMARK_TIME: %.4f s\n", t_bench);

  free(ba);
  free(bb);
  free(bc);
  return 0;
}
