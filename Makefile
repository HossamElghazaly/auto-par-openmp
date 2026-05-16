# ============================================================
# Makefile — Automatic Parallelization Tool Project
# ============================================================
# Targets:
#   all           - compile all original and parallelized C files
#   parallelize   - run the parallelizer tool on all test files
#   run_original  - run all sequential (original) versions
#   run_parallel  - run all parallelized versions with 8 threads
#   benchmark     - run the complete benchmark.py pipeline
#   clean         - remove all generated files
# ============================================================

CC       = gcc
CFLAGS   = -fopenmp -O2
LIBS     = -lm
PYTHON   = python
TOOL     = parallelizer.py
BENCH    = benchmark.py

ifneq (,$(findstring Windows,$(OS)))
    EXE = .exe
    RM  = del /Q /F
    RMDIR = rmdir /S /Q
    SET_ENV = set OMP_NUM_THREADS=8 &&
else
    EXE =
    RM  = rm -f
    RMDIR = rm -rf
    SET_ENV = OMP_NUM_THREADS=8
endif

# ── All: compile every original and auto-generated file ───────────────────────
all: test_safe_orig$(EXE) test_heat_orig$(EXE) test_matmul_orig$(EXE) \
     test_safe_auto$(EXE) test_heat_auto$(EXE) test_matmul_auto$(EXE)

test_safe_orig$(EXE): test_safe.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

test_heat_orig$(EXE): test_heat.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

test_matmul_orig$(EXE): test_matmul.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

test_safe_auto$(EXE): test_safe_auto.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

test_heat_auto$(EXE): test_heat_auto.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

test_matmul_auto$(EXE): test_matmul_auto.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# ── Parallelize: run tool on all test files ────────────────────────────────────
parallelize:
	$(PYTHON) $(TOOL) test_safe.c   --output test_safe_auto.c   --report --verbose
	$(PYTHON) $(TOOL) test_heat.c   --output test_heat_auto.c   --report --verbose
	$(PYTHON) $(TOOL) test_matmul.c --output test_matmul_auto.c --report --verbose

# ── Run original sequential versions ──────────────────────────────────────────
run_original:
	@echo "--- Running test_safe (original) ---"
	./test_safe_orig$(EXE)
	@echo "--- Running test_heat (original) ---"
	./test_heat_orig$(EXE)
	@echo "--- Running test_matmul (original) ---"
	./test_matmul_orig$(EXE)

# ── Run parallelized versions with 8 threads ──────────────────────────────────
run_parallel:
	@echo "--- Running test_safe (parallelized, 8 threads) ---"
	$(SET_ENV) ./test_safe_auto$(EXE)
	@echo "--- Running test_heat (parallelized, 8 threads) ---"
	$(SET_ENV) ./test_heat_auto$(EXE)
	@echo "--- Running test_matmul (parallelized, 8 threads) ---"
	$(SET_ENV) ./test_matmul_auto$(EXE)

# ── Full automated benchmark + plot pipeline ───────────────────────────────────
benchmark:
	$(PYTHON) $(BENCH)

# ── Clean all generated files ─────────────────────────────────────────────────
clean:
	$(RM) test_safe_orig$(EXE) test_heat_orig$(EXE) test_matmul_orig$(EXE) 2>nul || true
	$(RM) test_safe_auto$(EXE) test_heat_auto$(EXE) test_matmul_auto$(EXE) 2>nul || true
	$(RM) test_safe_auto.c test_heat_auto.c test_matmul_auto.c 2>nul || true
	$(RM) report.txt heat_output.bin thread_map.bin 2>nul || true
	$(RMDIR) FIGs 2>nul || true

.PHONY: all parallelize run_original run_parallel benchmark clean
