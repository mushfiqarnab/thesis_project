"""Lint the thesis .tex files for the specific errors that have occurred in this project.

Checks
  1. SWALLOWED TEXT: a comment line containing \\cite, \\ref, or ending with a sentence
     that continues prose (this is how a sentence was lost in Chapter 6).
  2. TRUNCATED SENTENCE: a prose line that does not end with sentence punctuation,
     followed by a comment line and then a blank line or new section.
  3. "USD" artifacts (a corrupted "$").
  4. Banned wording (see CLI_BRIEF_v2.md, section 9).
  5. Remaining TODO(authors) markers (reported, not an error).

Usage:
    python latex_lint.py chapters core appendix
Exit code 1 if any ERROR is found.
"""
import os
import re
import sys

BANNED = [r"\bprove[sd]?\b", r"\bproof\b", r"\bguarantee[sd]?\b", r"silicon-level",
          r"formally proven", r"\bzero CF gap\b", r"catastrophic", r"\bdefinitively\b",
          r"absolute ground truth", r"Counterfactual Risk Minimization",
          r"\bGWPACDNet\b", r"Equivariant Q-Attention", r"mathematically guaranteed"]
END_PUNCT = (".", ":", "}", "\\\\", "]", "?", "!", ";")
STRUCTURAL = re.compile(r"^\\(begin|end|item|section|subsection|paragraph|chapter|label|caption|centering|toprule|midrule|bottomrule|hline|includegraphics)")


def lint(path):
    errs, warns, todos = [], [], 0
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        n = i + 1
        if "TODO(authors)" in line:
            todos += 1
        if line.startswith("%"):
            body = line.lstrip("%").strip()
            if re.search(r"\\cite\{|\\ref\{|\\label\{", body):
                errs.append(f"{path}:{n}: SWALLOWED TEXT? comment contains \\cite/\\ref/\\label: {line[:90]}")
            continue
        if re.search(r"(?<!\\)USD", line):
            errs.append(f"{path}:{n}: 'USD' artifact (corrupted $): {line[:90]}")
        for pat in BANNED:
            if re.search(pat, line, flags=re.IGNORECASE):
                warns.append(f"{path}:{n}: banned wording /{pat}/: {line[:90]}")
        # truncated sentence: prose line w/o end punctuation, next is comment, then blank/structural
        if line and not STRUCTURAL.match(line) and not line.endswith(END_PUNCT) and len(line) > 40:
            j = i + 1
            saw_comment = False
            while j < len(lines) and lines[j].strip().startswith("%"):
                saw_comment = True
                j += 1
            nxt = lines[j].strip() if j < len(lines) else ""
            if saw_comment and (nxt == "" or STRUCTURAL.match(nxt)):
                errs.append(f"{path}:{n}: TRUNCATED SENTENCE? prose line followed by comment then break: {line[-90:]}")
    return errs, warns, todos


def main():
    roots = sys.argv[1:] or ["."]
    all_e, all_w, total_todo = [], [], 0
    for r in roots:
        for dp, _, fns in os.walk(r):
            for fn in fns:
                if fn.endswith(".tex") and "revisions" not in fn:
                    e, w, t = lint(os.path.join(dp, fn))
                    all_e += e; all_w += w; total_todo += t
    for m in all_e:
        print("ERROR  ", m)
    for m in all_w:
        print("WARN   ", m)
    print(f"\n{len(all_e)} errors, {len(all_w)} wording warnings, {total_todo} TODO(authors) markers remaining")
    sys.exit(1 if all_e else 0)


if __name__ == "__main__":
    main()
