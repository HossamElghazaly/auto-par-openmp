#!/usr/bin/env python3
"""
parallelizer.py
===============
A compiler-inspired automatic parallelization tool for C code.
Performs source-to-source translation by:
  1. Detecting SCoP (Static Control Part) regions
  2. Detecting for-loops and supported while-loops
  3. Analyzing data dependencies for each loop
  4. Injecting OpenMP pragmas where safe
  5. Generating a human-readable analysis report

Usage:
    python parallelizer.py input.c --output out.c --report --verbose

Author: Hossamaldeen Elghazaly — Nile University
"""

import re
import os
import sys
import argparse
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Pure math functions with no side effects — safe to call inside parallel loops
WHITELISTED_FUNCTIONS = {
    "sqrt", "abs", "fabs", "sin", "cos", "tan", "exp", "log",
    "log2", "log10", "pow", "floor", "ceil", "round", "fmin", "fmax",
}

# Dynamic memory functions — always unsafe inside loops
DYNAMIC_MEMORY_CALLS = {"malloc", "calloc", "realloc", "free"}

# Reduction operators mapped to their OpenMP clause
REDUCTION_OPERATORS = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — SCoP Region Detection
# ──────────────────────────────────────────────────────────────────────────────

def extract_scop(source: str):
    """
    Finds and returns all SCoP (Static Control Part) regions in the source.

    A SCoP region is marked by the programmer using:
        #pragma scop
        ... code ...
        #pragma endscop

    Args:
        source (str): Full C source code as a string.

    Returns:
        list of dict: Each dict has keys:
            - 'content' (str): the code inside the region
            - 'start_line' (int): line number of #pragma scop
            - 'end_line'   (int): line number of #pragma endscop
        Returns empty list if no SCoP regions found.
    """
    regions = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"\s*#\s*pragma\s+scop\s*$", lines[i]):
            start_idx = i
            start_line = i + 1  # 1-indexed
            content_lines = []
            i += 1
            while i < len(lines):
                if re.match(r"\s*#\s*pragma\s+endscop\s*$", lines[i]):
                    regions.append({
                        "content": "\n".join(content_lines),
                        "start_line": start_line,
                        "end_line": i + 1,
                        "start_idx": start_idx,
                        "end_idx": i,
                    })
                    break
                content_lines.append(lines[i])
                i += 1
        i += 1

    return regions


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Loop Detection
# ──────────────────────────────────────────────────────────────────────────────

def _extract_loop_body(text: str, start: int) -> str:
    """
    Extracts the full body of a loop starting at the opening brace.

    Args:
        text (str): Source code text.
        start (int): Character index of the opening '{'.

    Returns:
        str: Everything between the matching braces (exclusive).
    """
    depth = 0
    i = start
    body_start = -1
    while i < len(text):
        if text[i] == '{':
            if depth == 0:
                body_start = i + 1
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[body_start:i]
        i += 1
    return ""


def detect_loops(scop: dict) -> list:
    """
    Finds all for-loops and supported while-loops inside a SCoP region.

    For while-loops, they are converted to equivalent for-loop form internally
    if a counter variable can be identified.

    Args:
        scop (dict): A SCoP region dict from extract_scop().

    Returns:
        list of dict: Each dict contains:
            - 'line'        (int):  line number within full source
            - 'loop_var'    (str):  loop iteration variable (e.g., 'i')
            - 'start_bound' (str):  initial value expression
            - 'end_bound'   (str):  upper bound expression
            - 'body'        (str):  loop body text
            - 'loop_type'   (str):  'for' or 'while'
            - 'full_text'   (str):  the complete original loop text
    """
    content = scop["content"]
    base_line = scop["start_line"]
    loops = []

    # ── Detect for-loops ──────────────────────────────────────────────────────
    # Matches: for(int i = 0; i < n; i++) or for(i=0; i<N; i++)
    for_pattern = re.compile(
        r'for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*([^;]+);\s*\w+\s*[<>]=?\s*([^;]+);\s*[^)]+\)',
        re.MULTILINE
    )
    for m in for_pattern.finditer(content):
        loop_var   = m.group(1)
        start_b    = m.group(2).strip()
        end_b      = m.group(3).strip()
        brace_pos  = content.find('{', m.end())
        if brace_pos == -1:
            continue
        body = _extract_loop_body(content, brace_pos)
        line_offset = content[:m.start()].count('\n')
        loops.append({
            "line":        base_line + line_offset,
            "loop_var":    loop_var,
            "start_bound": start_b,
            "end_bound":   end_b,
            "body":        body,
            "loop_type":   "for",
            "full_text":   content[m.start(): brace_pos + len(body) + 2],
        })

    # ── Detect supported while-loops ─────────────────────────────────────────
    # Looks for: int i = 0; while(i < n) { ... i++; }
    while_pattern = re.compile(
        r'(?:int\s+)?(\w+)\s*=\s*(\d+)\s*;\s*while\s*\(\s*\1\s*[<>]=?\s*(\w+)\s*\)',
        re.MULTILINE
    )
    for m in while_pattern.finditer(content):
        loop_var  = m.group(1)
        start_b   = m.group(2)
        end_b     = m.group(3)
        brace_pos = content.find('{', m.end())
        if brace_pos == -1:
            continue
        body = _extract_loop_body(content, brace_pos)
        # Verify i++ or i += 1 is in body (unit stride)
        if not re.search(rf'\b{loop_var}\s*\+\+|\b{loop_var}\s*\+=\s*1', body):
            continue
        line_offset = content[:m.start()].count('\n')
        loops.append({
            "line":        base_line + line_offset,
            "loop_var":    loop_var,
            "start_bound": start_b,
            "end_bound":   end_b,
            "body":        body,
            "loop_type":   "while",
            "full_text":   content[m.start(): brace_pos + len(body) + 2],
        })

    # Sort by line number
    loops.sort(key=lambda x: x["line"])
    return loops


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Dependency Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_dependencies(loop: dict) -> dict:
    """
    Performs full dependency analysis on a single loop.

    Checks are performed in this priority order:
      1. Dynamic memory (malloc/calloc/realloc) → UNSAFE
      2. Recursive function calls → UNSAFE
      3. Unknown (non-whitelisted) function calls → UNSAFE
      4. Unsupported pointer patterns → UNSAFE
      5. Reduction patterns (sum+=, product*=) → REDUCTION
      6. Offset array indexing (a[i-1], a[i+1]) → UNSAFE (loop-carried dep.)
      7. Write-after-write / read-after-write conflicts → UNSAFE
      8. Supported pointer arithmetic *(ptr+i) → treat as safe array
      9. All checks passed → SAFE (parallel)

    Args:
        loop (dict): A loop dict from detect_loops().

    Returns:
        dict with keys:
            'safe'          (bool)
            'type'          (str):  'parallel' | 'reduction' | 'unsafe'
            'reason'        (str):  human-readable explanation
            'reduction_var' (str | None)
            'reduction_op'  (str | None)
    """
    body     = loop["body"]
    var      = loop["loop_var"]
    filename = loop.get("source_file", "unknown")

    result = {
        "safe":          False,
        "type":          "unsafe",
        "reason":        "",
        "reduction_var": None,
        "reduction_op":  None,
        "line":          loop["line"],
        "loop_var":      var,
        "body_snippet":  body.strip().split('\n')[0].strip()[:60],
    }

    # Pre-processing: strip any pre-existing OpenMP pragmas from the loop body
    # before running dependency checks.  Without this step, keywords embedded
    # inside pragma strings (e.g. 'schedule', 'reduction') would be matched by
    # Check 3 as unknown function calls, causing false UNSAFE classifications.
    body = re.sub(r'#\s*pragma\s+omp\b[^\n]*\n?', '', body)

    # ── Check 1: Dynamic memory allocation ───────────────────────────────────
    for fn in DYNAMIC_MEMORY_CALLS:
        if re.search(rf'\b{fn}\s*\(', body):
            result["reason"] = f"dynamic memory allocation '{fn}()' inside loop body"
            return result

    # ── Check 2: Recursion (simplified: function calling itself by name) ──────
    # Look for any function-call pattern; cross-reference against loop context
    func_calls = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', body)
    # We detect recursion heuristically: if a non-whitelisted, non-stdlib call
    # appears that matches a user-defined function name pattern
    # (Full recursion detection would need a call graph — we mark as unsafe
    #  if we see calls that look recursive based on naming conventions)
    for fn in func_calls:
        if fn in {"for", "while", "if", "else", "return", "sizeof", "printf",
                  "fprintf", "scanf", "fopen", "fclose", "fwrite", "fread",
                  "memset", "memcpy", "omp_get_thread_num", "omp_get_wtime"}:
            continue
        if fn in WHITELISTED_FUNCTIONS or fn in DYNAMIC_MEMORY_CALLS:
            continue
        if fn.lower() == filename.replace('.c', '').lower():
            result["reason"] = f"possible recursive call to '{fn}()' detected"
            return result

    # ── Check 3: Unknown function calls ──────────────────────────────────────
    for fn in func_calls:
        if fn in {"for", "while", "if", "else", "return", "sizeof", "printf",
                  "fprintf", "scanf", "fopen", "fclose", "fwrite", "fread",
                  "memset", "memcpy", "omp_get_thread_num", "omp_get_wtime"}:
            continue
        if fn not in WHITELISTED_FUNCTIONS and fn not in DYNAMIC_MEMORY_CALLS:
            result["reason"] = f"unknown function call '{fn}()' — possible side effects"
            return result

    # ── Check 4: Unsupported pointer patterns ─────────────────────────────────
    # Allow:  *(ptr + i) or *(ptr+i)  — index must equal the loop variable
    # Reject: *ptr (standalone dereference), ptr->field (struct access),
    #         *(ptr + k) where k is not the loop variable
    # Must NOT flag: a[i] * b[i]  (multiplication, not dereference)

    arrow_deref = re.findall(r'\w+\s*->', body)
    if arrow_deref:
        result["reason"] = "struct pointer dereference '->': not supported"
        return result

    # Raw dereference: * immediately followed by an identifier, not part of *(...)
    raw_deref = re.findall(
        r'(?<![\w\]\)\+\-\/])\*\s*(?!\()([a-zA-Z_]\w*)(?!\s*[\[\(\+\-])',
        body
    )
    raw_deref = [r for r in raw_deref if r]
    if raw_deref:
        result["reason"] = (
            f"raw pointer dereference '*{raw_deref[0]}': "
            "cannot determine access pattern"
        )
        return result

    # Parenthesised pointer arithmetic: *(ptr + expr)
    # Safe only when expr is exactly the loop variable (with any surrounding spaces)
    ptr_deref_pat = re.compile(r'\*\s*\(\s*(\w+)\s*\+\s*(\w+)\s*\)')
    for pm in ptr_deref_pat.finditer(body):
        index = pm.group(2).strip()
        if index != var:
            result["reason"] = (
                f"complex pointer arithmetic: "
                f"only '*(ptr+{var})' pattern is supported"
            )
            return result

    # ── Check 5: Reduction patterns ───────────────────────────────────────────
    for op_token, omp_op in REDUCTION_OPERATORS.items():
        red_pattern = re.compile(
            rf'\b(\w+)\s*{re.escape(op_token)}\s*.+;',
            re.MULTILINE
        )
        m = red_pattern.search(body)
        if m:
            red_var = m.group(1)
            # Make sure the reduced variable is NOT an array (not followed by '[')
            if not re.search(rf'\b{re.escape(red_var)}\s*\[', body):
                result["safe"]          = True
                result["type"]          = "reduction"
                result["reason"]        = f"reduction pattern detected: '{red_var} {op_token} ...'"
                result["reduction_var"] = red_var
                result["reduction_op"]  = omp_op
                return result

    # ── Check 6: Offset array indexing (loop-carried dependency) ─────────────
    # Detect a[i-1], a[i+1], a[i+k] where k != 0
    offset_pattern = re.compile(
        rf'\w+\s*\[\s*{re.escape(var)}\s*[\+\-]\s*\d+\s*\]'
    )
    if offset_pattern.search(body):
        offending = offset_pattern.search(body).group(0)
        result["reason"] = f"loop-carried dependency: offset index access '{offending.strip()}'"
        return result

    # ── Check 7: Write-after-write / read-after-write conflicts ──────────────
    # Find all arrays written to (LHS of assignment)
    written_arrays = set(re.findall(rf'(\w+)\s*\[{re.escape(var)}\](?:\s*\[[^\]]*\])?\s*=', body))
    # Find all arrays read from (RHS of assignment or anywhere else)
    all_arrays     = set(re.findall(r'(\w+)\s*\[', body))
    # Read arrays = all arrays that appear in RHS (appear, but not only on LHS)
    # Simplified: if same array appears both as written and in RHS with potentially
    # different index — flag as unsafe
    for arr in written_arrays:
        # Find all usages of this array in the body
        usages = re.findall(rf'\b{re.escape(arr)}\s*\[([^\]]+)\]', body)
        # If more than one unique index expression and one is not strictly [var] → suspect
        unique_indices = set(idx.strip() for idx in usages)
        for idx in unique_indices:
            if idx != var and not re.match(rf'^{re.escape(var)}\s*$', idx):
                if re.search(rf'[a-zA-Z]', idx.replace(var, '')):
                    result["reason"] = (
                        f"potential write-after-read conflict: '{arr}' accessed with "
                        f"indices {unique_indices}"
                    )
                    return result

    # Refined compound array-write check: detect patterns of the form
    # array[expr] += / -= / *=.  If the index expression contains the current
    # loop variable, each iteration writes to a unique element — no race
    # condition exists at this loop level and the loop remains safe.
    # If the index does NOT contain the loop variable, multiple threads could
    # write to the same element concurrently — flag the loop as UNSAFE.
    compound_array_matches = re.finditer(r'(\w+)\s*\[\s*([^\]]+)\s*\]\s*(?:\+\=|\-\=|\*\=)', body)
    for m in compound_array_matches:
        arr_name   = m.group(1)
        index_expr = m.group(2)
        if var not in index_expr:
            result["reason"] = f"compound assignment on array element '{arr_name}[]' — potential reduction race condition"
            return result

    # ── Check 8 (implicit): Pointer arithmetic *(ptr+i) ──────────────────────
    # This pattern is already safely handled by reaching here (not rejected above)

    # ── All checks passed → SAFE ──────────────────────────────────────────────
    result["safe"]   = True
    result["type"]   = "parallel"
    result["reason"] = "all dependency checks passed — iterations are independent"
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 4+5 — Pragma Injection & Output Code Generation
# ──────────────────────────────────────────────────────────────────────────────

def inject_pragmas(source: str, analysis_results: list, scop_regions: list) -> str:
    """
    Returns the fully transformed C source code with OpenMP pragmas injected.
    Handles nested loops by only parallelizing the outermost loop.
    Converts while-loops to for-loops for OpenMP compatibility.
    """
    lines = source.splitlines(keepends=True)

    def find_keyword_line(from_line_1idx, keyword):
        """Scan forward from from_line_1idx to find the line containing keyword."""
        for offset in range(10):
            pos = from_line_1idx - 1 + offset
            if pos < len(lines):
                if keyword in lines[pos]:
                    return pos + 1
        return from_line_1idx

    def estimate_loop_end(for_line_1idx):
        """Estimate the last line of the loop body by counting braces."""
        depth = 0
        opened = False
        for i in range(for_line_1idx - 1, len(lines)):
            depth += lines[i].count('{') - lines[i].count('}')
            if '{' in lines[i]: opened = True
            if opened and depth == 0:
                return i + 1
        return for_line_1idx + 1

    # deletions: line indices (1-indexed) to remove
    deletions = set()
    # insertions: line_number (1-indexed) -> [text_to_prepend]
    insertions = {}
    # replacements: line_number (1-indexed) -> new_text
    replacements = {}

    sorted_results = sorted(analysis_results, key=lambda r: r["line"])
    annotated_ranges = []

    for res in sorted_results:
        line_no = res["line"]
        l_type  = res.get("loop_type", "for")
        l_var   = res.get("loop_var", "i")

        # Find the line where the 'for' or 'while' actually starts
        actual_line = find_keyword_line(line_no, l_type)

        # Skip if this loop falls inside an already-annotated outer loop
        if any(start <= actual_line <= end for start, end in annotated_ranges):
            continue

        if res["type"] in ("parallel", "reduction"):
            # Mark the range of this outer loop
            end_line = estimate_loop_end(actual_line)
            annotated_ranges.append((actual_line, end_line))

            # Pragma construction
            if res["type"] == "parallel":
                pragma = "    #pragma omp parallel for schedule(static)\n"
            else:
                r_var = res["reduction_var"]
                r_op  = res["reduction_op"]
                pragma = f"    #pragma omp parallel for reduction({r_op}:{r_var})\n"

            if l_type == "for":
                insertions.setdefault(actual_line, []).append(pragma)
            elif l_type == "while":
                # For while loops, we need to:
                # 1. Replace while(cond) with for(int var=start; var<end; var++)
                # 2. Delete the original initialization line (usually the 'line_no')
                # 3. Comment out the increment i++ in the body
                start = res.get("start_bound", "0")
                end   = res.get("end_bound", "N")
                
                orig_line = lines[actual_line - 1]
                # Replace the while(...) part but KEEP the rest of the line (e.g. the body and closing brace if on the same line)
                new_loop_str = f"for (int {l_var} = {start}; {l_var} < {end}; {l_var}++) {{"
                new_line = re.sub(rf'\b{l_var}\s*(\+\+|\+=\s*1);?', r'', orig_line)
                
                # If the increment is on the SAME line as the while statement, comment it out safely
                new_line = re.sub(r'while\s*\(.*?\)\s*\{?', new_loop_str, new_line)

                insertions.setdefault(actual_line, []).append(pragma)
                replacements[actual_line] = new_line

                # Locate and delete the original while-loop initializer line
                # (e.g., 'int i = 0;') by scanning upward from the while line.
                # The synthesized for-loop header already contains the initializer,
                # so the original line must be removed to avoid a duplicate
                # variable declaration that would cause a compilation error.
                init_found = False
                for i in range(actual_line - 1, -1, -1):
                    # Look for 'int i = 0;' or 'i = 0;'
                    if re.search(rf'\b(int\s+)?{l_var}\s*=\s*{re.escape(start)}\b', lines[i]):
                        deletions.add(i + 1)
                        init_found = True
                        break
                
                # Fallback to original logic if search fails
                if not init_found and line_no < actual_line:
                    deletions.add(line_no)

                # Comment out the induction-variable increment inside the body
                # (e.g., 'i++;' or 'i += 1;').  The converted for-loop header
                # already contains the increment expression; leaving it inside
                # the body would advance the counter twice per iteration.
                for i in range(actual_line, end_line):
                    # Fixed regex: removed trailing \b which fails on '++;'
                    if re.search(rf'\b{l_var}\s*(\+\+|\+=\s*1)', lines[i]):
                        replacements[i+1] = "        // " + lines[i].lstrip()

        elif res["type"] == "unsafe":
            comment = f"    /* AUTO-PARALLELIZER: loop NOT parallelized - {res['reason']} */\n"
            insertions.setdefault(actual_line, []).append(comment)

    output_lines = []
    for idx, line in enumerate(lines):
        lineno = idx + 1
        if lineno in deletions:
            continue
        if lineno in insertions:
            for ins in insertions[lineno]:
                output_lines.append(ins)
        if lineno in replacements:
            output_lines.append(replacements[lineno])
        else:
            output_lines.append(line)

    final_output = "".join(output_lines)
    
    # Brace validation check
    if final_output.count('{') != final_output.count('}'):
        print("  [WARNING] Brace mismatch in output file — possible malformed transformation.")
        
    return final_output

# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Report Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(analysis_results: list, output_path: str):
    """
    Writes a formatted analysis report to a text file.

    Each analyzed loop gets one line with its classification, line number,
    body snippet, and reason. A summary at the bottom shows totals and the
    percentage of loops successfully parallelized.

    Args:
        analysis_results (list): List of dicts from analyze_dependencies().
        output_path      (str):  Path to write the report file.
    """
    # Additive report mode: if report.txt already exists from a previous run,
    # parse and retain its entries for other input files, then merge them with
    # the new results for the current file.  This lets benchmark.py invoke the
    # parallelizer once per source file and accumulate a single combined report
    # rather than overwriting it on each call.
    report_file = Path(output_path)
    parsed_existing = []
    
    if report_file.exists():
        old_text = report_file.read_text(encoding="utf-8")
        current_fname = analysis_results[0].get("source_file", "") if analysis_results else ""
        
        for ln in re.findall(r'^\[[A-Z ]+\].+', old_text, re.MULTILINE):
            m = re.search(r'^\[(.*?)\]\s+(.*?\.c)\s+Line\s+(\d+):\s+(.*?)\s+->\s+(.*)', ln)
            if m:
                tag, fname, lineno, snippet, reason = m.groups()
                # Exclude stale lines from the current file
                if fname.strip() == current_fname:
                    continue
                parsed_existing.append({
                    "type": tag.strip().lower(),
                    "file": fname.strip(),
                    "line": int(lineno),
                    "body_snippet": snippet.strip(),
                    "reason": reason.strip(),
                    "inside_scop": True  # Assume retained old loops are valid
                })

    # Add file key for new results
    for res in analysis_results:
        res["file"] = res.get("source_file", "unknown")

    # Combine all results
    analysis_results = parsed_existing + analysis_results

    # Fix 1 & 2: Deduplication and SCoP filtering
    seen = set()
    final_lines = []
    
    for entry in analysis_results:
        # Fix 2: Skip entries outside SCoP
        if not entry.get("inside_scop", False):
            continue
            
        # Fix 1: Track (filename, line_number) pairs already logged
        key = (entry['file'], entry['line'])
        if key in seen:
            continue
        seen.add(key)
        
        # Format for report
        tag_str = f"[{entry['type'].upper():<10}]"
        ln_str = f"{tag_str} {entry['file']} Line {entry['line']:>4}: {entry['body_snippet']:<45}  -> {entry['reason']}"
        final_lines.append(ln_str)
        
    final_lines.sort()

    # Calculate combined totals
    total     = len(final_lines)
    parallel  = sum(1 for ln in final_lines if "[PARALLEL  ]" in ln)
    reduction = sum(1 for ln in final_lines if "[REDUCTION ]" in ln)
    unsafe    = sum(1 for ln in final_lines if "[UNSAFE    ]" in ln)
    skipped   = sum(1 for ln in final_lines if "[SKIPPED   ]" in ln)
    pct       = ((parallel + reduction) / total * 100) if total > 0 else 0.0

    sep = "-" * 80
    lines = [
        "=" * 80,
        "  AUTO-PARALLELIZER -- Analysis Report (Combined)",
        "=" * 80,
        "",
    ]
    lines += final_lines
    lines += [
        "",
        sep,
        "  SUMMARY",
        sep,
        f"  Total loops analyzed     : {total}",
        f"  Parallelized (parallel)  : {parallel}",
        f"  Parallelized (reduction) : {reduction}",
        f"  Skipped (inner loop)     : {skipped}",
        f"  Rejected (unsafe)        : {unsafe}",
        f"  Parallelization rate     : {pct:.1f}%",
        sep,
    ]

    report_file.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Core Pipeline Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_path: str,
                 write_report: bool, verbose: bool) -> list:
    """
    Executes the full 7-step auto-parallelization pipeline on a C source file.

    Args:
        input_path   (str): Path to the input sequential C file.
        output_path  (str): Path to write the parallelized C file.
        write_report (bool): Whether to write report.txt.
        verbose      (bool): Whether to print step-by-step decisions.

    Returns:
        list: All analysis result dicts from each loop found.
    """
    source = Path(input_path).read_text(encoding="utf-8")
    fname  = Path(input_path).name

    if verbose:
        width = 54
        print("+" + "=" * width + "+")
        print(f"|   AUTO-PARALLELIZER -- Analyzing: {fname:<{width - 36}}|")
        print("+" + "=" * width + "+")
        print()

    # Step 1 — SCoP detection
    scop_regions = extract_scop(source)
    if not scop_regions:
        if verbose:
            print("  [WARNING] No #pragma scop / #pragma endscop regions found.")
            print("            Nothing to parallelize. Output = original file.")
        Path(output_path).write_text(source, encoding="utf-8")
        return []

    # Steps 2+3 — Loop detection & dependency analysis
    all_results = []
    for scop in scop_regions:
        loops = detect_loops(scop)
        
        # Calculate nesting depth for each loop
        loop_ranges = []
        for loop in loops:
            start_line = loop["line"]
            end_line = start_line + loop["full_text"].count('\n')
            loop_ranges.append((start_line, end_line))
            
        for idx, loop in enumerate(loops):
            start_line, end_line = loop_ranges[idx]
            # It's inside another loop if the other loop starts before (or same line) and ends after (or same line)
            # but we use strict inequality for start_line to avoid self-counting, or just enumerate
            depth = 1
            for j, (s, e) in enumerate(loop_ranges):
                if idx != j and s <= start_line and end_line <= e:
                    depth += 1
            loop["depth"] = depth

        for loop in loops:
            loop["source_file"] = fname
            if loop["depth"] > 1:
                res = {
                    "safe": False,
                    "type": "skipped",
                    "reason": f"inner loop (depth {loop['depth']}) -> outermost-only parallelization policy",
                    "line": loop["line"],
                    "loop_var": loop["loop_var"],
                    "body_snippet": loop["body"].strip().split('\n')[0].strip()[:60]
                }
            else:
                res = analyze_dependencies(loop)
                
            # Merge loop metadata into analysis results for Step 4+5
            res.update(loop)
            res["inside_scop"] = True
            all_results.append(res)

            if verbose:
                tag = res["type"].upper()
                line = res["line"]
                snippet = res["body_snippet"]
                reason  = res["reason"]
                tag_str = f"[{tag:<10}]"

                if res["type"] == "parallel":
                    pragma = "#pragma omp parallel for"
                    print(f"  {tag_str} Line {line} -> {snippet[:40]:<42} -> {pragma} inserted")
                elif res["type"] == "reduction":
                    rvar  = res["reduction_var"]
                    rop   = res["reduction_op"]
                    pragma = f"#pragma omp parallel for reduction({rop}:{rvar})"
                    print(f"  {tag_str} Line {line} -> {snippet[:40]:<42} -> {pragma} inserted")
                elif res["type"] == "skipped":
                    print(f"  [SKIPPED ] Line {line} -> inner loop (depth {res['depth']}) -> outermost-only parallelization policy")
                else:
                    print(f"  {tag_str} Line {line} -> {snippet[:40]:<42} -> {reason}")

    # Step 4+5 — Inject pragmas and write output
    # Strip #pragma scop / #pragma endscop from output (analysis-only markers)
    transformed = inject_pragmas(source, all_results, scop_regions)
    clean_lines = []
    for ln in transformed.splitlines(keepends=True):
        stripped = ln.strip()
        if re.match(r'#\s*pragma\s+(end)?scop\s*$', stripped):
            continue
        clean_lines.append(ln)
    transformed = "".join(clean_lines)
    Path(output_path).write_text(transformed, encoding="utf-8")

    # Step 6 — Generate report
    if write_report:
        report_path = str(Path(output_path).parent / "report.txt")
        generate_report(all_results, report_path)

    # Verbose summary
    if verbose:
        total     = len(all_results)
        parallel  = sum(1 for r in all_results if r["type"] == "parallel")
        reduction = sum(1 for r in all_results if r["type"] == "reduction")
        skipped   = sum(1 for r in all_results if r["type"] == "skipped")
        unsafe    = sum(1 for r in all_results if r["type"] == "unsafe")
        pct       = ((parallel + reduction) / total * 100) if total > 0 else 0.0
        sep = "-" * 54
        print()
        print(f"  {sep}")
        print(f"  SUMMARY: {parallel + reduction} parallelized "
              f"({parallel} parallel + {reduction} reduction)")
        print(f"           {skipped} skipped (inner loops)")
        print(f"           {unsafe} rejected")
        print(f"           {pct:.1f}% of loops successfully parallelized")
        print(f"  {sep}")
        print()
        print(f"  [DONE] Output written to: {output_path}")
        if write_report:
            print(f"  [DONE] Report written to: {Path(output_path).parent / 'report.txt'}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Command-line interface entry point for the auto-parallelizer.

    Supported flags:
        input_file          — positional: path to input C file
        --output filename   — output file name (default: <input>_auto.c)
        --report            — generate report.txt alongside output
        --verbose           — print full analysis decisions to terminal
    """
    parser = argparse.ArgumentParser(
        description="Automatic C loop parallelizer using OpenMP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python parallelizer.py test_safe.c --verbose\n"
            "  python parallelizer.py test_heat.c --output test_heat_parallel.c --report --verbose\n"
        )
    )
    parser.add_argument("input_file",
                        help="Path to the sequential C source file to analyze")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file name (default: <input>_auto.c)")
    parser.add_argument("--report", action="store_true",
                        help="Generate a report.txt with loop analysis details")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed analysis decisions to terminal")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = str(Path(args.output))
    else:
        stem = input_path.stem
        output_path = str(input_path.parent / f"{stem}_auto.c")

    run_pipeline(
        input_path  = str(input_path),
        output_path = output_path,
        write_report= args.report,
        verbose     = args.verbose,
    )


if __name__ == "__main__":
    main()
