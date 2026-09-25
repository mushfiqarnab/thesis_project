You are finalizing FULL_THESIS_PAPER.md for submission as a PDF draft due today. 
Read the file in full before touching anything. Then apply every edit below in 
exact order. Show a diff of each change before applying it. Do not apply any 
change I have not explicitly listed. Do not run any training, preprocessing, or 
pipeline code. This is a writing and editing task only.

VERIFIED GROUND TRUTH (from filesystem — do not deviate from these numbers):
- N = 8 video subjects confirmed (s1-s8, 3 avi each, 24 clips total)
- POS α=0.5: U=1050.0, p=0.0314 (PRIMARY pre-registered test)
- CHROM: U=1318.0, p=0.0001 (exploratory, post-hoc, NOT pre-registered)
- PBV: U=1113.0, p=0.0103 (exploratory, post-hoc, NOT pre-registered)
- Naive ERM accuracy: Seed 42 = 79.94%, Seed 100 = 80.98%, Seed 2026 = 80.98%
- EQUITAS-RCMF CF Gap = 0.0006359847 across ALL THREE regimes (ρ=0.85, 0.50, 0.15)
- EQUITAS-RCMF accuracy = 61.69% (all three regimes identical)
- Stiefel orthogonality = 1.7149×10⁻⁶ (NOT 1.76×10⁻⁶)
- Edge latency mean = 0.8987ms (rounds to 0.90ms ✓), p95 = 1.2961ms (rounds to 1.30ms ✓)
- Throughput = 1112.7 FPS ✓
- SHA-256 = 2035777F957FA4C5CC830989F4FE40FB9190589F17C567CB29A3EF0E6B2ACAE2 ✓
- XAI figure EXISTS: outputs/reports/XAI_Causal_Blindness_Proof.png ✓
- Edge compiled model EXISTS: outputs/deployment/equitas_rcmf_edge_compiled.pt ✓
- equitas_rcmf_master_best.pt is MISSING from disk (deleted/overwritten)
- sham_edit_results.json does NOT exist (probe could not run — checkpoint missing)
- ρ=0.50 and ρ=0.15 are generated at runtime by dataloader from single CSV (confirmed real)
- Split files confirmed for all 5 seeds: 42, 100, 2026, 777, 888
- Training epochs confirmed: 250 per seed (from train_naive_erm_baseline JSON report)

═══════════════════════════════════════════════════════
EDIT 1 — Pre-registration hierarchy (CRITICAL — do this first)
═══════════════════════════════════════════════════════
In Chapter 4, Section 4.1, replace the existing leakage results table with:

| rPPG Estimator | Mann-Whitney U | p-value | Pre-registration Status |
|:---|:---|:---|:---|
| **POS (α=0.5)** | **U = 1050.0** | **p = 0.0314** | **Primary pre-registered confirmatory test** |
| CHROM | U = 1318.0 | p = 0.0001 | Exploratory corroboration (post-hoc, not confirmatory) |
| PBV | U = 1113.0 | p = 0.0103 | Exploratory corroboration (post-hoc, not confirmatory) |

Immediately after the table, add this paragraph:
"The pre-registered primary test (POS α=0.5, Mann-Whitney U=1050.0, p=0.0314) 
formally rejects the null hypothesis that compression destroys all recoverable 
physiological signal (pre-registered criterion: p ≤ 0.05 = leakage detected). 
Two post-hoc estimators provide corroborating evidence: CHROM (U=1318.0, p=0.0001) 
and PBV (U=1113.0, p=0.0103), both labeled exploratory per the pre-registration 
document and not treated as independent confirmations. The CHROM result is 
particularly notable — per-clip analysis reveals r=0.849 against ground-truth wrist 
BVP for subject s2 T1, confirmed via spectral verification (cardiac frequency match 
Δf=0.000 Hz, SNR=53.5), indicating genuine physiological signal survival rather than 
a statistical artifact."

Also update the Abstract: replace "p = 0.0001, CHROM estimator" with:
"p = 0.0314 (POS, primary pre-registered test), corroborated by CHROM (p = 0.0001, 
exploratory) and PBV (p = 0.0103, exploratory)"

Also update Section 6.1 Contribution 1: replace "CHROM p = 0.0001" with:
"POS p = 0.0314 (primary, pre-registered); CHROM p = 0.0001 and PBV p = 0.0103 
(exploratory corroboration)"

═══════════════════════════════════════════════════════
EDIT 2 — Update N throughout the document
═══════════════════════════════════════════════════════
Replace every instance of "N=4" in the main text with "N=8".
Replace every instance of "4 subjects" with "8 subjects" where it refers to 
the UBFC-Phys video cohort.
In Appendix A.1, confirm it already says "N=8 video-capable subjects" — if so, 
leave it. If it says anything different, update to:
"Cohort: N=8 video-capable subjects (s1–s8), 24 clips total (3 tasks × 8 subjects)"

═══════════════════════════════════════════════════════
EDIT 3 — Fix the Stiefel orthogonality value
═══════════════════════════════════════════════════════
Replace every instance of "1.76 × 10⁻⁶" or "1.76×10⁻⁶" with "1.7149 × 10⁻⁶".
This applies in: Abstract, Section 3.4, Section 4.2 ablation table (if present), 
Appendix A.5, and anywhere else it appears.

═══════════════════════════════════════════════════════
EDIT 4 — Fix the Adebayo/sham edit section
═══════════════════════════════════════════════════════
In Section 4.5, replace the specific claimed results 
("Accuracy Delta: 0.00%", "DP Gap Delta: 0.0000") with:

"The sham-edit sanity check protocol (Adebayo et al., 2018) was implemented in 
src/evaluation/sham_edit_probe.py, applying a 30×30 pixel non-semantic perturbation 
at pixel coordinates [20:50, 20:50] of clean test images. The probe requires the 
EQUITAS-RCMF master checkpoint; due to a file system incident during final 
consolidation, the master checkpoint (equitas_rcmf_master_best.pt, SHA-256: 
4c2dcad470271ad7109ec302a2e4d31eac429cc054ba3d243ce0d39266b3bc5b) was not 
recoverable at submission time. Qualitative analysis of the architecture confirms 
selectivity by design: the Stiefel decomposition routes non-semantic perturbations 
into the confounder subspace, where they are discarded by JIT Dead-Code Elimination 
at inference. Full numerical Adebayo results will be reported in the final submission 
(October 7th) upon checkpoint recovery."

Also update Section 3 Contributions item 5: change the claimed result to:
"Adebayo (2018) Sanity Check protocol implemented (src/evaluation/sham_edit_probe.py); 
numerical results pending checkpoint recovery for final submission."

═══════════════════════════════════════════════════════
EDIT 5 — Move the limitations section to correct position
═══════════════════════════════════════════════════════
The text currently at line ~446 (after References and Appendices, starting with 
"### 4.3 Honest Limitations and Scope") is misplaced. 

Do the following:
1. Cut that entire paragraph from its current location at the end of the file.
2. Paste it inside Chapter 4, after Section 4.5 (Adebayo) and before Section 4.6 
   (Edge Benchmark). It should become the new Section 4.6, and the current 4.6 
   and 4.7 should renumber to 4.7 and 4.8.
3. Fix the broken LaTeX: change "=4$ subjects" to "N=8 subjects".
4. Update the limitations text to reflect the confirmed facts:
   - "N=8 vision subjects" (confirmed)
   - "equivalence testing was withdrawn due to N=8 being below the pre-registered 
     power threshold for strict TOST bounds at the subject-level t-interval"
   - Keep the sham condition limitation as-is (accurate)
   - Remove any reference to "N=4" — it is wrong

═══════════════════════════════════════════════════════
EDIT 6 — Add O7 sensitivity finding to Section 4.4 (EMA)
═══════════════════════════════════════════════════════
At the end of Section 4.4 (the EMA/ablation section), before the next section 
heading, add:

"**Sensitivity Analysis (O7).** A post-hoc geometry audit revealed that two clips 
(s4 T2 and s4 T3) in the α=1.0 ablation arm exhibited crop sizes approximately 37% 
smaller than the committed IOD-scale rule, due to a detection anomaly during that 
extraction pass. Excluding these two clips from the α=1.0 arm yields Mann-Whitney 
U=832.0, p=0.0857 — above the pre-registered threshold. The α=1.0 result (p=0.0437) 
must therefore not be cited as strong independent evidence of leakage at zero 
smoothing; it is fragile with respect to this geometric anomaly. The EMA verdict 
(inconclusive) is unchanged and if anything strengthened: removing the anomalous 
clips moves the no-smoothing result toward acceptance, which is the opposite of what 
an EMA-driven mechanism would predict. The primary finding rests on the α=0.5 
pre-registered result (p=0.0314) and the exploratory CHROM/PBV corroboration, 
neither of which is affected by the O7 anomaly. Full O7 analysis is persisted at 
outputs/leakage_run/o7_reanalysis.txt (SHA-256: 6a1ac8f0…61ee4d)."

═══════════════════════════════════════════════════════
EDIT 7 — Fix the demographic audit table accuracy figure
═══════════════════════════════════════════════════════
In Section 4.3 (the demographic audit table), the accuracy column currently shows 
figures like 53.69%, 50.69%, etc. per demographic group. The verified master model 
accuracy from the JSON report is 61.69% in Autonomous Mode at ρ=0.50. 

Check whether the demographic group accuracies sum/average to approximately 61.69%. 
If they are consistent (per-group breakdowns of the same overall 61.69%), leave them. 
If the overall accuracy is stated elsewhere as a different number, update it to 
61.69% (from equitas_rcmf_master_benchmark_report.json, confirmed on disk).

Also add a footnote to the table: "Source: outputs/reports/equitas_rcmf_master_benchmark_report.json, 
produced from equitas_rcmf_master_best.pt (SHA-256: 4c2dcad4…b3bc5b, checkpoint 
generated September 21, 2026)."

═══════════════════════════════════════════════════════
EDIT 8 — Remove the "Zero fabrication" claim
═══════════════════════════════════════════════════════
Find and remove the sentence "All values read directly from hardware output files. 
Zero fabrication." from Section 4.3. Replace it with:
"All values are sourced from output artifacts committed to the repository 
(see Appendix A.3); source file paths are cited inline throughout this chapter."

═══════════════════════════════════════════════════════
EDIT 9 — Update Appendix A to reflect confirmed state
═══════════════════════════════════════════════════════
In Appendix A.4 (Cross-Validation Configuration), confirm it reads:
"Seeds: [42, 100, 2026, 777, 888] | Epochs: 250 per run | Batch: 32 | LR: 1e-4"
This is confirmed correct from the JSON report. Leave as-is.

In Appendix A.1, update the cohort line to:
"Cohort: N=8 video-capable subjects (s1–s8); N=15 WESAD physiology subjects. 
24 video clips total (8 subjects × 3 tasks)."

Add a note after the SHA-256 line:
"Hash verified on 2026-09-23 via PowerShell Get-FileHash against the physical file 
at data/publishable_scar_production/multimodal_publishable.csv."

═══════════════════════════════════════════════════════
EDIT 10 — Add a defense-ready note on the missing checkpoint
═══════════════════════════════════════════════════════
In Section 6.2 (Limitations) or at the end of Appendix A.3, add:

"**Note on Master Checkpoint Recovery.** The EQUITAS-RCMF master checkpoint 
(equitas_rcmf_master_best.pt) was confirmed present during benchmark generation 
(September 21, 2026, SHA-256: 4c2dcad4…b3bc5b) but was not located on disk at 
final draft preparation. All benchmark metrics (CF Gap, accuracy, orthogonality, 
edge latency) were generated from this checkpoint and are reported from the 
corresponding JSON output files, which are committed to the repository. 
Checkpoint recovery for full reproducibility is in progress."

═══════════════════════════════════════════════════════
FINAL VERIFICATION STEP — Before saving the file
═══════════════════════════════════════════════════════
After all edits are applied, run these checks and report the results:

1. grep -n "0.0125\|0.0221\|1.76\|N=4\|=4\$\|Zero fabrication\|79.94\b" 
   docs/FULL_THESIS_PAPER.md
   (Should return ZERO matches for 0.0125, 0.0221, 1.76, "Zero fabrication". 
   79.94 should appear only in the Naive ERM accuracy section with correct context.)

2. grep -n "0.0314\|1050.0\|pre-registered\|exploratory\|1.7149\|N=8" 
   docs/FULL_THESIS_PAPER.md
   (Should return multiple matches confirming all corrections landed.)

3. Report the final line count of docs/FULL_THESIS_PAPER.md.

4. Confirm §4.3 Honest Limitations appears BEFORE §4.6 Edge Benchmark in the 
   document (not after the References).

DO NOT COMMIT ANYTHING. DO NOT RUN ANY CODE. DO NOT MODIFY ANY FILE OTHER THAN 
docs/FULL_THESIS_PAPER.md. Show me the diff of each edit and wait for my 
confirmation before applying the next one if any edit is ambiguous. Apply all 
non-ambiguous edits in sequence and report when done.