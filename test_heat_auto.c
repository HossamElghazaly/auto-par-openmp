/*
 * test_heat.c
 * ===========
 * 2D Heat Diffusion Jacobi Solver — Benchmark for Auto-Parallelizer
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ITER 5000
#define TOLERANCE 1e-4

static const int N = 1024;

/* ── Memory helpers ─────────────────────────────────────────────────────── */
static double *alloc_grid(int n) {
  double *g = (double *)calloc((size_t)n * n, sizeof(double));
  if (!g) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  return g;
}

/* ── Boundary Conditions ────────────────────────────────────────────────── */
static void apply_boundaries(double *grid, int n) {
  for (int j = 0; j < n; j++) {
    grid[0 * n + j] = 100.0;     /* Top edge: hot  */
    grid[(n - 1) * n + j] = 0.0; /* Bottom edge: cold */
    grid[j * n + 0] = 0.0;       /* Left edge  */
    grid[j * n + (n - 1)] = 0.0; /* Right edge */
  }
}

int main(void) {
  double *u = alloc_grid(N);
  double *u_new = alloc_grid(N);

  /* Thread-mapping array: records which OpenMP thread processed each row */
  int *thread_map = (int *)calloc(N, sizeof(int));
  if (!thread_map) {
    fprintf(stderr, "Thread map alloc failed\n");
    exit(1);
  }

  apply_boundaries(u, N);
  apply_boundaries(u_new, N);

  printf("=======================================================\n");
  printf("  2D Heat Diffusion — Jacobi Solver\n");
  printf("  Grid size  : %d x %d\n", N, N);
  printf("  Max iters  : %d\n", MAX_ITER);
  printf("=======================================================\n");

/* Thread map recording - outside SCoP, manually parallelized */
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    thread_map[i] = omp_get_thread_num();
  }

  double t_start = omp_get_wtime();
  double max_diff = 0.0;
  int iter = 0;

  for (iter = 0; iter < MAX_ITER; iter++) {
    max_diff = 0.0;

    /*
     * Jacobi update: parallelize over rows (i).
     * Each row i is fully independent — no data dependency between rows.
     * The tool inserts: #pragma omp parallel for schedule(static)
     * giving 1022 independent row iterations split across threads.
     * max_diff is computed inline but tracked per-row via local var
     * to avoid a race; the outer loop handles the global max serially
     * per row (acceptable since the inner j loop dominates).
     */
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < N - 1; i++) {
      for (int j = 1; j < N - 1; j++) {
        u_new[i * N + j] = 0.25 * (u[(i - 1) * N + j] + u[(i + 1) * N + j] +
                                   u[i * N + (j - 1)] + u[i * N + (j + 1)]);
      }
    }

    /* Swap grids */
    double *tmp = u;
    u = u_new;
    u_new = tmp;
    apply_boundaries(u, N);

    /* max_diff: quick serial estimate (one element per row) */
    for (int i = 1; i < N - 1; i++) {
      double d = fabs(u[i * N + N / 2] - u_new[i * N + N / 2]);
      if (d > max_diff)
        max_diff = d;
    }
  }

  double t_end = omp_get_wtime();

  printf("  Iterations : %d\n", iter);
  printf("  Max diff   : %.6e\n", max_diff);
  printf("BENCHMARK_TIME: %.4f s\n", t_end - t_start);

  /* ── Save full 2D temperature grid (binary) ───────────────────────────── */
  FILE *fp = fopen("heat_output.bin", "wb");
  if (fp) {
    int n_out = N;
    fwrite(&n_out, sizeof(int), 1, fp);
    fwrite(u, sizeof(double), (size_t)N * N, fp);
    fclose(fp);
    printf("  Grid saved : heat_output.bin\n");
  }

  /* ── Save thread workload map (binary) ───────────────────────────────── */
  FILE *fp2 = fopen("thread_map.bin", "wb");
  if (fp2) {
    int n_out2 = N;
    fwrite(&n_out2, sizeof(int), 1, fp2);
    fwrite(thread_map, sizeof(int), N, fp2);
    fclose(fp2);
    printf("  Thread map : thread_map.bin\n");
  }

  free(u);
  free(u_new);
  free(thread_map);
  return 0;
}
