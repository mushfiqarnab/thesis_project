# EQARNB: Experiment Pre-Registration Plan

**Task:** Acute Stress Classification (Research Only)
**Date:** September 2026

## Core Hypothesis
If the FFHQ (Vision) and WESAD (Physiology) datasets are completely independent, the visual modality contains zero true signal for the Acute Stress label ($Y$). Therefore, the EQARNB architecture's ability to suppress the visual confounder ($S$, the scar) should yield performance identical to a model that never sees the visual data at all.

## Experimental Definitions & Decision Rules

**Dataset Status:** The current FFHQ+WESAD dataset relies on independent sources. These initial tests are diagnostic. Note: Previous EQARNB results are deprecated. Informative vision remains unresolved until migration to a naturally multimodal dataset (e.g., UBFC-Phys) or a controlled attribute dataset.
* **Test Set Consistency:** To ensure per-subject paired differences are isolated, all evaluation regimes ($\rho \in \{0.85, 0.5, 0.15\}$) will use the exact same test windows, varying only the synthetic scar/sham assignment.

**Primary Endpoint & Contrasts:**
* **Endpoint:** Accuracy on the designated primary regime: the Inverted Adversarial split ($\rho=0.15$). *(Amendment: This regime was designated to guarantee a well-defined per-subject difference for the t-interval, before any evaluation under this protocol; old-pipeline results were already known).* Performance on $\rho \in \{0.85, 0.5\}$ is secondary.
* **Primary Contrasts:** K1a vs EQARNB, and K2b vs EQARNB.
* **Hierarchy & Correction:** We define two separate Holm families:
  1. TOST p-values for the two contrasts.
  2. Two-sided difference p-values for the two contrasts.
* **Outcome Table (by sign):** Keep four cells (Equivalent, Different, Inconclusive, Trivially Different); report the CI as a descriptive note.
  * *K1a vs EQARNB:* K1a better -> something in the fused model hurts (attribute using K2b). EQARNB better at $\rho=0.15$ -> anomaly; run leakage and artifact audit before any interpretation.
  * *K2b vs EQARNB:* EQARNB better -> candidate case for the architecture (needs secondary confirmation). K2b equal or better -> architecture adds nothing on this endpoint.

**Manipulation Check:**
* K2a (CE, trained at $\rho=0.85$) must show an accuracy drop from $\rho=0.85$ to $\rho=0.15$ of at least X = \_\_\_ points (subject-level t-interval lower bound). If not, the $\rho=0.15$ endpoint is reported as uninformative for method comparison.

**Equivalence Rule (TOST):** 
* We use Two One-Sided Tests (TOST) at $\alpha=0.05$, checking if the 90% Confidence Interval of the paired difference lies entirely inside $[-2.0, 2.0]$.
* **Unit of Inference & Weighting:** Each subject is weighted equally.
* **Estimator:** The primary interval is a t-interval on the per-subject paired differences. A percentile bootstrap acts as a sensitivity analysis.
* **Aggregation:** For repeated grouped K-fold, a subject's paired differences are averaged across all repetitions they appear in *before* applying the t-test over subjects.

**Power Planning Fallback:**
* Before unblinding the evaluation sets, we will estimate the per-subject SD via inner cross-validation on training subjects. 
* If predicted power < 80% (using the upper 80% bound on the larger of the two contrast SDs) at the evaluation $N=15$ (physiology-only cohort), the WESAD run proceeds as estimation-only, with no equivalence claim.

### Failures and Amendments
* **Run Failures:** Any diverged or crashed run will not be silently dropped. The run will be logged, and restarted using the next pre-listed seed.
* **Amendments:** Any protocol changes made after the dry runs will be recorded as dated amendments.

### Dry-Run Validations & Freeze (Pre-Tagging)
1. **Label-Permutation Run:** Permute *training* labels only. The raw-accuracy CI upper bound must not exceed the majority-class rate, and the balanced-accuracy CI must contain 50% with half-width <= 3.0. A programmatic assertion will strictly enforce zero subject overlap across splits.
2. **Simulated-Effect Run:** Feed the analysis pipeline synthetic per-subject results (differences of 0, $\pm 2$, and $>2$). This run must exercise the exact designated primary estimator (t-interval on $\rho=0.15$), utilize unequal windows per subject, apply the actual fold structure, and output a coverage and power report over a grid of SDs.
3. **Freeze Checklist:** Before evaluation, hash and lock the analysis script, split files, seed list, config, and thresholds (X, 2.0, 3.0, 80%).

### Dataset Fix: Sham Edits & Artifact Probes
Every clean image receives a skin-texture/freckle "sham" edit. Sham and scar edits are strictly matched on size and location distributions, and both obey facial keypoint occlusion constraints. 
* **Artifact Probe:** A classifier trained exclusively on boundary-ring pixels must score near chance when predicting "scar vs. sham".

**Pre-Registration Amendment (2026-09-23): Mediapipe Face Detection, Margin Expansion, and Anatomical Confounder Placement**
1. **Detector & Margin Swap**: The pipeline is migrated to Mediapipe (`mp.solutions.face_detection`, pinned to `0.10.14`) to eliminate bbox-width jitter observed in Haar cascades. *Note: A uniform baseline failure rate of ~0.1% (exactly 1 frame per clip) is explicitly recognized as a deterministic OpenCV `CAP_PROP_FRAME_COUNT` EOF read boundary artifact, not a detection failure.* Because Mediapipe bounds the face strictly from eyes to mouth, the bounding box margin is expanded to **1.5x** (replacing 1.3x) to guarantee sufficient boundary clearance for skin-anchored confounder placement without clipping.
2. **Confounder Placement Anchor**: To prevent phase-correlated positional drift caused by 2D bounding box deformation under 3D head pitch, the scar/sham confounders will be anchored using a pure 2D anatomical interpolation: `0.5 * Eye_Midpoint + 0.5 * Nose_Tip`. This mathematically locks the confounder to the midline nose-bridge.
3. **Outlier Acknowledgement & Full-Scale Monitoring**: Face-normalized coordinate validation on the dev set ($n=4$) confirmed no consistent group-level phase-correlated drift. Subjects `s2`, `s3`, and `s4` exhibited bidirectional noise (Cohen's d < 0.5). However, `s1` exhibited a massive, anomalous phase-correlated shift (d = -2.26). Because $n=4$ is underpowered to definitively distinguish a mechanical detection flaw from a true subpopulation behavioral effect, we adopt the 2D anatomical interpolation now (explicitly rejecting full 3D pose reprojection as out of scope) but formally commit to recalculating and reporting this face-normalized anchor stability metric across the entire $N=55$ (physiology-only) cohort as a documented monitoring step. **Concrete Trigger:** If >15% of the full cohort exhibits an absolute shift of |d| > 1.0 on this face-normalized metric, the anatomical anchor choice is officially invalidated and will be revisited before model training begins.

### K1: Physiology-Only Baselines
* **K1a (Architecture Match):** Train the `phys_mlp` branch on WESAD data only.
* **K1b (Ceiling Match):** Train a gradient boosting model (e.g., XGBoost) on rich WESAD handcrafted features to establish the absolute physiological ceiling. 
* **Hypotheses:** 
  * If K1a/b $\approx$ EQARNB (via TOST), the vision branch contributes nothing.
  * If K1a > EQARNB, something in the fused model is harming performance (compare with K2b to isolate).
  * If K1a < EQARNB on the biased split, EQARNB *may* be exploiting the scar shortcut. (Confirm by verifying predictions flip when the scar is removed from the same images).

### K2: Unconstrained ERM Baselines
* **K2a (Naive ERM):** Train standard Cross-Entropy on the biased split ($\rho=0.85$).
* **K2b (Balanced ERM):** Train on balanced ($\rho=0.5$) data. The critical baseline for worst-group fairness metrics.

### K4: Conditional Independence (HSIC) & Probes
* **Estimator:** Song et al. (2012) unbiased U-statistic estimator.
* **Setup:** Compute HSIC($v_c$, $S$) within each $Y$ stratum and average. Use RBF kernel on $v_c$ (median heuristic) and delta kernel on $S$. 
* **Loss Handling:** The raw unbiased value is kept in the loss (its expectation is correct and noise averages out); $\max(0, \text{HSIC})$ is applied only for reporting.
* **Final Judgment:** A minibatch penalty only encourages independence. The final measure is the accuracy of frozen, held-out linear/MLP probes evaluated post-training.

### Pre-Registration Lock
Before execution, this document, the codebase, configs, seed lists, and split hashes will be frozen via a signed Git tag.

### K5: Vision Conditional Utilization Rate
* **Setup:** Ablate the thermodynamic gate at inference time (force $G=1$ or $G=0$) and measure the drop in accuracy.
* **Pre-Registered Prediction:** The conditional utilization rate of vision will be $\approx 0\%$, confirming that the network relies entirely on physiology for the stress classification task in this specific synthetic dataset.

---
*Note: Depending on the outcome of K1, the thesis will either pivot to a pure benchmark/analysis paper of the synthetic routing mechanism, or migrate to a dataset with informative vision (e.g., UBFC-Phys).*

### Pre-Registered Leakage Threshold (Logged 2026-09-23 02:35 AM)
For all subsequent POS extraction tests, the formal pass/fail criterion is established as a Mann-Whitney U test p-value > 0.05 (True vs. Null distributions). A p-value <= 0.05 constitutes statistically significant leakage and results in an automatic fail.


### Pre-Registered Ablation Criterion (Logged 2026-09-23 02:40 AM)
The EMA smoothing hypothesis will be tested by running the pipeline at alpha=1.0 (no smoothing) vs. alpha=0.5 (current smoothing). The hypothesis 'EMA is the leakage mechanism' is formally confirmed ONLY IF alpha=1.0 passes the leakage test (Mann-Whitney p > 0.05) AND alpha=0.5 fails the leakage test (Mann-Whitney p <= 0.05). Any other combination (both fail, or both pass) is defined as inconclusive, not confirmatory.


### Data Loss & Chain of Custody Limitation (Logged 2026-09-23 03:02 AM)
The exact unrounded constant 3.566283 (the historical mean ratio of Mediapipe bounding-box width to Inter-Ocular Distance) is taken on faith from a single computation that cannot be reproduced. The raw per-frame bbox-width/IOD data behind this scalar was permanently destroyed by an unbacked-up overwrite at 2026-09-23 02:40 AM when the alpha ablation run was launched. Anyone auditing this analysis must note that this core scaling constant has a broken chain of custody.

**Geometry audit (2026-09-23, final review):** the destroyed raw data cannot be recovered (confirmed: no manifest files exist in the original scratch directory that ran the ablation). However, the *retained* manifests of the current-generation artifacts do carry per-frame bbox width and eye keypoints, and a full audit of them (`scripts/ubfc_leakage/audit_manifest_geometry.py`, output `outputs/leakage_run/manifest_geometry_audit.txt`) found: (a) the α=0.5 pass is internally uniform across all 12 clips, with per-frame width/IOD = 5.344 ± 0.054, implying the historical constant via the committed rule (w = IOD × 3.566283 × 1.5) to within 0.1% — consistent with the constant as used, but **circular**, since these crops were cut using it; (b) the **α=1.0 pass is NOT internally uniform**: clips s4 T2 and s4 T3 were cut with per-frame width/IOD of 3.37 ± 0.15 and 3.73 ± 0.49 respectively (implying constants ≈ 2.25 and ≈ 2.48), ~37% smaller in linear size than the other ten clips, and s4 T3 is missing 15 OK frames relative to every other clip. File timestamps are identical across all twelve α=1.0 manifests (2026-09-23 02:50:57), so the anomaly was not produced by the documented 02:40 overwrite of that pass; its provenance is unresolved. The audited quantity recovers the *committed* rule, not the historical scalar, so no resolution of the constant's original derivation is claimed.

### Data Loss #2 — multimodal / multimodal-10k training inputs (Logged 2026-09-23)

The training inputs for the FFHQ-scar model battery were lost. Affected files: `data/csv/multimodal.csv`, `data/csv/multimodal_10k.csv`, `data/csv/multimodal_10k_unbiased.csv`, and the split definitions `data/csv/split_seed42_multimodal.json`, `data/csv/split_seed42_multimodal_10k.json`, `data/csv/split_seed42_multimodal_10k_unbiased.json`.

* **What they were:** the tabulated training corpora and deterministic (seed-42) split files for the counterfactual/CGF battery — Design A/B (February 2026), the multimodal-10k and unbiased variants, and the September 2026 strict/Stiefel/edge ablation battery.
* **Evidence they existed and were used:** `outputs/reports/train_counterfactual_multimodal_*.json`, `train_counterfactual_report.json`, and `repair_multimodal_10k_unbiased_mobilenet_v3_small.json` record the exact paths as run configuration; 15 checkpoints in `outputs/checkpoints/` carry `multimodal_10k` in their names with mtimes 2026-02-03 11:35 through 2026-09-21 05:08. (That the September runs re-read these files from disk rather than from a cache is UNVERIFIED.)
* **Current state:** absent from disk as of the 2026-09-23 inventory; `data/csv/` now holds only `approved_pristine_manifest.csv`, `multimodal_diffusion_worldclass.csv/.jsonl`, and `wesad_windows.csv`. **No file under `data/csv/` was ever tracked in git** (`git ls-files data/csv` is empty), so the repository contains no recoverable copy of any of them.
* **Deletion event:** Deletion was deliberate (author, 2026-09-23): the multimodal / multimodal-10k training corpora and their generating pipeline were judged low-quality and removed. Superseded, not lost to accident. The deletion remains bounded between the last consistent use (checkpoints through 2026-09-21 08:58) and verified absence (2026-09-23). The severity assessment below is unchanged: this destroyed the *inputs* to finished, checkpoint-bearing runs, which is categorically worse than the two losses above, which destroyed *derivatives* whose generating inputs and code survive.
* **Downstream now unverifiable:** all multimodal-10k-lineage checkpoints are unrerunnable from source — including `counterfactual_cgf_js_vit_b_16_multimodal_10k_unbiased_best_stiefel.pt` (344 MB, 2026-09-18), the September strict/Stiefel/concat battery, the February Design A/B runs (`multimodal.csv`-lineage, including `baseline_best.pt` and `counterfactual_fair_best.pt` at 344 MB each), and the pruned/repaired derivatives. Their surviving result reports (uncommitted) are unverifiable by re-run. **Not affected:** the 2026-09-21 `publishable` production runs, which read `data/publishable_scar_production/multimodal_publishable.csv` (present on disk, 102 MB).
* **Scope note:** these runs belong to the pre-pivot FFHQ-scar model lineage, which the Dataset Status note at the top of this document marks deprecated. How that lineage's surviving artifacts (checkpoints, uncommitted reports, edge exports) relate to the current thesis scope is an author-level question raised on 2026-09-23 and deliberately **not resolved in this document**. **Update (2026-09-23, author):** partially resolved by the O6-rescope amendment at the end of this document - the publishable corpus is in scope as the controlled-confounder benchmark; the multimodal-10k lineage itself remains superseded.

### Session Review Findings (Logged 2026-09-23, post-amendment)

A code-level review of the session's uncommitted training/dataset/model diffs was performed, plus a direct audit of `data/publishable_scar_production/multimodal_publishable.csv` (the surviving corpus) against its actual contents.

**A. Dataset audit (verified against the file, not the builder):** 3,344 rows; 418 unique faces x 8 WESAD windows each; embedded `train`/`val`/`test` split with **zero overlap across splits on both `face_id` and `physiology_subject`** (train 10 subjects / 293 faces, val 2 / 63, test 3 / 62); every face carries both scar=0 and scar=1 rows with exact `clean_path`/`scarred_path`/`counterfactual_image_path` pairing and sha256 provenance columns fully populated. Realized bias: P(scar=1|threat=1)=0.850, P(scar=1|threat=0)=0.150, corr(scar, threat)=0.70, uniform across all three splits. **However, the Core Hypothesis premise remains unearned under this corpus**: `face_id` and `physiology_subject` are unrelated by construction (0% identity match; every face is crossed with 2-8 different physiology subjects), so vision carries zero *true* signal about the label and any learnable vision-label association is the painted confounder itself. The scar renderer version in the builder is `scar-like-renderer-2.0`; no sham-edit condition exists in the current builder (the sham/artifact-probe protocol in this document is unimplemented there).

**B. Provenance finding - the λ=5.0 sweep result and the edge ONNX sit on destroyed inputs.** The session's modified `experiment_lambda_sweep.py` reads `CSV_PATH = "data/csv/multimodal.csv"` - a Data Loss #2 file verified absent from disk (2026-09-23). The existing `outputs/lambda_sweep_results.csv` row (λ=5.0, acc 0.7155, DP 0.0444; mtime 2026-09-21 00:02) and `outputs/lambda_sweep_ckpts/lambda_5.0_{best,final}.pth` (mtimes 2026-09-21 00:01-00:02) predate the loss window close, so they were plausibly produced while the input existed, but the run is **unrerunnable and unverifiable from source**, joining the rest of the multimodal-lineage artifacts above. The exported `outputs/equitas_mitl_strict_v4_edge.onnx` (mtime 2026-09-21 00:14) is derived from `lambda_5.0_final.pth` per the session's modified `src/edge/export_v4_onnx.py` and inherits the same lineage. No new claim should be built on these artifacts. The sweep script itself now implements a genuinely different estimator (learned Lagrange multiplier on the DP penalty, clamped >= 0, per-batch dual ascent) than the fixed-λ grid its results table implies; the λ column of future results must be labeled accordingly.

**C. Code fixes applied during this review (both verified):**
1. `src/train_cgf_fair.py` - the degenerate-epoch guard used the sentinel `score = -999.0` against `best_score = -1e9`; any degenerate epoch therefore *always* overwrote the best checkpoint with a degenerate model (save path: `-999.0 > -1e9`), defeating the guard's purpose. Fixed: degenerate epochs are skipped entirely (never scored, never saved); `best_score` initialized to `-inf`; if all epochs are degenerate, no checkpoint is created and the post-training test evaluation is skipped by its existing `exists()` guard.
2. `.gitignore` - the negation patterns `!data/csv/split_*.json` etc. were dead: `data/**` excludes the `data/csv` directory itself, and a negation cannot re-include a file whose parent directory is excluded (confirmed via `git check-ignore` and `git add -n`). This was the standing condition under which Data Loss #2's split files were untracked despite protective-looking patterns. Fixed: `!data/csv/` re-included before its children; `.venv*/` added (`.venv_worldclass/` was untracked-but-unignored); verified with `git add -n` that split JSONs are now trackable while bulk data remains excluded.
3. **Sentinel bug audit gap (2026-09-23, Task 2):** The two Sep 21 production benchmark reports (`thesis_production_benchmark_report.json`, `equitas_rcmf_master_benchmark_report.json`) contain no checkpoint path or hash field; the attribution of those results to `equitas_rcmf_master_best.pt` (Sep 21 08:44) rests on file timestamp and §3.4 of the session handoff document, not on any machine-readable field in the reports themselves. The five scripts audited for the sentinel bug pattern are clean; the two Sep 21 checkpoints postdate the fix. `dry_run_best.pt` (Sep 19 02:35) was produced by `train_cgf_fair.py` while the bug was active and carries no downstream citations.

**D. Verification asset created:** `data/publishable_scar_production/multimodal_publishable.csv.sha256` records SHA-256 = `2035777f957fa4c5cc830989f4fe40fb9190589f17c567cb29a3ef0e6b2acae2` for the operative corpus. Any future training run on this CSV must assert this hash at startup and refuse to proceed on mismatch. The CSV itself remains bulk-excluded from git (393 MB-class images are the real corpus; the CSV alone is not the dataset), which is why the sidecar matters: the hash is the trackable identity.

**E. Still missing before any run:** (i) the O6 pivot decision and this FFHQ-scar line were in unresolved tension at logging time - subsequently partially resolved by the O6-rescope amendment at the end of this document (the publishable corpus is in scope as the benchmark corpus); the multimodal-10k lineage remains out; (ii) the sham-edit / artifact-probe protocol has no builder implementation, so the publishable corpus cannot support the artifact-probe claim as built; (iii) a small smoke test + manual inspection of rendered scar pairs remains a prerequisite to any full run, per the freeze discipline this document already commits to.

---

## Pre-Registration Amendment (2026-09-23): Vision Source Migration FFHQ -> UBFC-Phys

**The body of this document is now partially superseded and must be read with this amendment.**

### What changed

The **Core Hypothesis** and **Dataset Status** sections describe vision as supplied by FFHQ portraits paired against WESAD physiology, with UBFC-Phys named as a *future* contingency ("Informative vision remains unresolved until migration to a naturally multimodal dataset (e.g. UBFC-Phys)"). That migration has since happened. The operative vision source is UBFC-Phys video frames.

Evidence for the migration:

* The extraction pipeline reads `data/UBFC-Phys/{sid}/vid_{sid}_T{task}.avi` directly (`scratch/preprocess_video_mediapipe.py`), i.e. per-subject `.avi` video, not static portraits.
* The Mediapipe / anatomical-anchor / POS amendments above are all defined over 12 UBFC-Phys clips (subjects `s1`-`s4`, tasks `T1`-`T3`), and the POS null distribution is built from 132 cross-subject/cross-task pairings drawn from those clips.
* FFHQ now appears only in the legacy synthetic-scar dataset builders (`src/build_publishable_scar_dataset.py`, `src/prepare_faces.py`, `src/audit_pristine_sources.py`, `src/build_approved_manifest.py`), not in the rPPG or anchor work.

### Why it matters to the hypothesis

Under FFHQ + WESAD the zero-true-signal premise in the Core Hypothesis was free: a stranger's portrait cannot carry the subject's stress label, because the two sources are independent by construction. Under UBFC-Phys that premise is **no longer free and must be earned**. Each subject's own video contains that subject's own recoverable physiological signal, and a POS rPPG extractor can in principle recover it from facial pixels alone. The Core Hypothesis therefore only continues to hold if the compression / leakage pipeline demonstrably destroys that recoverable signal.

The POS leakage test exists specifically to test that condition. It is not an unrelated diagnostic: it is the mechanism by which the zero-true-signal assumption is re-justified under the new vision source.

### Consequence for the K1-K5 definitions

The K1-K5 contrasts and the $\rho \in \{0.85, 0.5, 0.15\}$ splits were authored for the FFHQ + WESAD setup and have **not** been re-derived for UBFC-Phys. Whenever this document asserts that vision contains "zero true signal", that assertion is conditional on the leakage gate below passing. Until it does, a result in which EQARNB matches or beats K1a cannot be attributed to architecture rather than to residual vision leakage, and this document should not be cited as if the premise were established.

### Status of the zero-true-signal assumption under UBFC-Phys

**Not established.** The POS leakage test failed at both smoothing settings (detail below), so the condition required to re-justify the Core Hypothesis is currently unmet.

---

## Outstanding Handoff Items, Now Recorded (Logged 2026-09-23)

These were carried as outstanding handoff items before the crop / anchor work could be called settled. Both are recorded here as required by the amendment policy above.

### A. Alpha-smoothing ablation against the pre-registered criterion

Source: `scratch/ablation_results.txt` (both passes, complete). Pre-registered rule logged 2026-09-23 02:40: *the EMA hypothesis is confirmed ONLY IF alpha=1.0 passes the leakage test (Mann-Whitney p > 0.05) AND alpha=0.5 fails (p <= 0.05). Any other combination is inconclusive, not confirmatory.*

| Pass | True mean r (N=12) | Null 95th pct r (N=132) | Null mean | Mann-Whitney U | p (True > Null) | Formal verdict |
|---|---|---|---|---|---|---|
| alpha = 1.0 (no smoothing) | 0.0752 | 0.1172 | 0.0597 | 1029.0 | 0.0437 | **FAIL** (p <= 0.05) |
| alpha = 0.5 (smoothed) | 0.0802 | 0.1186 | 0.0599 | 1050.0 | 0.0314 | **FAIL** (p <= 0.05) |

**Geometry provenance flag (2026-09-23, final review):** the α=1.0 row above must not be cited at per-clip granularity. In the preserved α=1.0 manifests, clips s4 T2 and s4 T3 show crop geometry inconsistent with the committed crop rule (see the Geometry audit under the Data Loss entry above); the provenance of their manifest columns is unresolved. The aggregate α=1.0 statistics in this table were computed over RGB spatial means of the produced crops and are reported as they were; the geometry anomaly affects 2 of 12 clips in that arm only. The α=0.5 row is uniform across all 12 clips and unaffected. The estimator-comparison results cited in the writeup (POS/CHROM/PBV) were all computed on the α=0.5 artifacts.

**Verdict: INCONCLUSIVE. The EMA hypothesis is NOT confirmed.** Removing smoothing did not remove the leakage; the leakage remained statistically significant at alpha=1.0 (p = 0.0437) and was in fact marginally *stronger* at alpha=0.5 (p = 0.0314). The direction of the effect argues against EMA being the leakage mechanism, but per the criterion above this observation is explicitly **not** a confirmation and must not be reported as one.

**Reporting hazard to note:** the evaluation script also prints a second line, `Passed Leakage Test (True <= Null 95th)? True`, in *both* passes, which contradicts the Mann-Whitney line in the same output block. The percentile criterion (mean r below the 95th percentile) passes while the rank test fails. The criterion pre-registered at 02:35 designates the **Mann-Whitney p-value as the formal pass/fail test**, so the formal verdict is FAIL for both passes. The percentile line must not be quoted as a pass.

### B. Detection nondeterminism check (alpha=1.0 vs alpha=0.5)

This check was written (`scratch/check_nondeterminism.py`) but its output was never recorded. It has now been executed and the result is:

**PASS - all 12 clips match exactly on detection status between the two passes** (12/12 matched pairs, satisfying the script's own `matched_pairs == 12` assertion). Method: `status` column of `processed/{s1-s4}/T{1-3}/manifest_alpha1.csv` (pass 1) diffed against the corresponding `manifest.csv` (pass 2); 1801 rows per manifest (1800 target frames plus header).

**Proof of fresh execution (2026-09-23 07:43:01):** the check was re-run from a clean shell against the current artefacts using the project interpreter (`.venv/Scripts/python.exe`), not read from any earlier cached output, and printed `SUCCESS: All 12 clips match exactly on OK/FILLED/FAILED status between alpha=1.0 and alpha=0.5 passes.` The script's internal `assert matched_pairs == 12` would have raised otherwise, so the pass is not an artefact of silently skipped files.

**Scope, and why the scope is correct:** this establishes determinism of the `OK` / `FILLED` / `FAILED` status labels. Crop coordinates (`x`, `y`, `w`, `h`) and raw keypoints were deliberately **not** compared, because `alpha` is by construction the EMA coefficient on the box itself (`box = alpha * x + (1 - alpha) * prev_x`); the two arms are *supposed* to differ in box geometry, since that is the treatment, not noise. A coordinate comparison would therefore fail by design and would not be evidence of nondeterminism. Box-coordinate reproducibility *within* a single arm is a separate question and is not covered here.

### C. Reconciliation with the scar-artifact validation protocol

The scar protocol now lives at `docs/SCAR_ARTIFACT_VALIDATION_PROTOCOL.md` and is *not* duplicated here. The two documents currently use non-overlapping vocabulary for the same placement logic, so the split of authority is fixed as follows:

* **Placement geometry is authoritative in this document.** The Mediapipe 1.5x bbox margin and the `0.5 * Eye_Midpoint + 0.5 * Nose_Tip` anatomical anchor defined in the amendment above define where a confounder may be rendered. The protocol's gate on an "exclusion region around eyes, nostrils, lips" and its "frozen configuration range" are satisfied *by* these constants rather than redefining them.
* **Acceptance and rejection are authoritative in the protocol.** Its automated gates and its restriction on permitted realism claims govern what may be reported about the artifacts.
* **Zero-overlap enforcement:** this document's requirement that no source face or physiology subject appears in more than one split is enforced by the protocol's gate of the same name.
* The protocol's requirement of exact outside-mask preservation, `MAE_outside(I, I_scar) = 0`, is consistent with the exact pairing requirement in this document.

---

## Open Action Items (Logged 2026-09-23)

These are unresolved blockers. They are recorded here because the amendment policy above requires protocol-relevant gaps to be dated rather than carried implicitly.

### O1. BLOCKING - no confounder injector exists for the UBFC-Phys vision source

This is the most serious open item and it undercuts the execution, not just the documentation, of the placement geometry in the amendment above.

* The `0.5 * Eye_Midpoint + 0.5 * Nose_Tip` vessel described in the 2026-09-23 amendment as the **Confounder Placement Anchor** is, in the code that currently exists, used for two things: the **crop-box centre** in `scratch/preprocess_video_mediapipe.py`, and a **face-normalised stability metric** in `scratch/get_face_norm_anchor.py`. Neither of these is confounder placement.
* No script anywhere in the repository or the working session renders a scar or sham edit onto a UBFC-Phys crop. The UBFC crop format (`processed/{sid}/T{task}/f{k:05d}.png`) is read only by `scratch/run_full_pos_pipeline.py`, which computes POS rPPG from the spatial mean RGB of the whole clean crop and injects nothing.
* Every scar / sham renderer that does exist is FFHQ-portrait lineage (`src/data/causal_scar_synthesizer.py`, `src/build_publishable_scar_dataset.py`, `src/audit_pristine_sources.py`, `src/scarbench_*`, `src/data/commercial_diffusion_engine.py`) and operates on static portraits, not on these video-derived crops.

**Consequence:** the scar/sham edit, the sham-matched size and location distributions, the boundary-ring artifact probe, and the $\rho$-controlled scar assignment currently have **no execution path** on the migrated vision source. The claim that "placement geometry is authoritative in this document" is therefore presently aspirational rather than operational. **OUT OF SCOPE (2026-09-23, author):** descoped by the pivot decision recorded under O6. No confounder injector will be built for the UBFC-Phys vision source, and the FFHQ-lineage builders remain out of this line of work. This record is retained deliberately: it is why the K1-K5 contrasts were never run, and that has to be stated rather than left as a silent gap.

### O2. BLOCKING - the vision modality exists for only 4 of 56 subjects

The 2026-09-23 amendment commits to recalculating the face-normalised anchor stability metric "across the entire $N=55$ cohort as a documented monitoring step", with a concrete >15% invalidation trigger. That commitment is **not currently executable**.

* `data/UBFC-Phys/` contains 56 subject directories, but only `s1`-`s4` contain video: 12 `.avi` files total (56.0 GB). These four are the dev set.
* The remaining 52 subjects contain `bvp_*.csv`, `eda_*.csv` and `info_*.txt` only (12.2 MB of CSV across the whole cohort, no video). `scratch/extraction_log.txt` records exactly those 52 subjects as `OK` - the physiology was extracted, the video was not. There is no pending-download evidence on disk (`s3_listing.txt` and `data/README.md` are both 0 bytes), so the missing video cannot be assumed to be one extraction away.

**Verification (2026-09-23), performed directly against the filesystem rather than against `scratch/extraction_log.txt`:** `find data/UBFC-Phys -name '*.avi' -printf '%h\n' | sort -u` returns exactly `s1`, `s2`, `s3`, `s4`; a per-subject enumeration of all 56 directories returns `3 avi` for `s1`-`s4` and `0 avi` for the other 52; and no UBFC video exists elsewhere under the user profile. This is a current on-disk fact, not an inference from an earlier log. (The log's own `OK` labels do describe the 52 physiology-only subjects, consistent with the above.)

**Consequence:** every vision-side claim in this document - the anchor stability metric at cohort scale, the POS leakage test beyond the 4-subject dev set, the power-planning step at evaluation $N=15$ (physiology-only), and the $\rho \in \{0.85, 0.5, 0.15\}$ evaluation arms - is scoped to the vision data actually available.

**STATED CONSTRAINT (2026-09-23, author):** the video payload **cannot be expanded further** - disk space and available time preclude additional download; the author states room for "three more".

**FINAL DECISION (2026-09-23, author, confirmed after full consideration including timeline):** $N_{vision} = 4$. No additional subjects will be acquired. The earlier "three more" statement is **superseded** and $N_{vision} \le 7$ is no longer live. Every vision-side statement in this document must therefore be read at $\mathrm{df}=3$, i.e. at the $N=4$ column of O6 throughout. This closes O2 as a resourcing decision rather than an engineering task.

**Framing note:** this is not a newly introduced defect. The physiology-only extraction across all 56 subjects was a deliberate early scope choice taken to avoid the full video payload; what is new on 2026-09-23 is that the arithmetic of that choice was finally performed and it does not support the pre-registered inferential plan. It is a known tradeoff surfacing late, not a surprise regression. (The specific figure quoted for the avoided payload is not found anywhere in this repository's markdown. It has been reported as originating in the project session brief, but until the author confirms that wording it is deliberately not cited here as verified, and it should be written into the repository before being cited in the thesis.)

### O3. Unfilled threshold in the Manipulation Check

The Manipulation Check still reads "at least X = \_\_\_ points", with the threshold for the K2a accuracy drop from $\rho=0.85$ to $\rho=0.15$ left blank. This must be filled with a concrete number before any freeze; a blank threshold cannot be checked against, and the freeze checklist references it.

**OUT OF SCOPE (2026-09-23):** the Manipulation Check will not run under the pivot - there is no confounder injection and therefore no K2a manipulation to check - so this threshold is no longer required. Retained as a record of the gap.

### O4. K1-K5 and the rho-split design have not been re-derived for UBFC-Phys

The amendment above notes that the K1-K5 contrasts and the $\rho$ splits were authored for the FFHQ + WESAD setup and have not been re-derived. What is still missing is scope: no owner, no method, and no assessment of whether $\rho$-controlled bias is even constructible under a fixed subject pool rather than a resampleable portrait pool, nor what per-fold $N$ a t-interval on per-subject paired differences can support at that pool size. This is a prerequisite for the next training run and is currently unstarted.

**OUT OF SCOPE (2026-09-23):** descoped with K1-K5 under the pivot.

### O5. Cohort count discrepancy

The 2026-09-23 amendment and the Power Planning Fallback refer to $N=55$, whereas 56 subject directories exist on disk. The two figures must be reconciled and the intended denominator stated explicitly.

**Resolved for the vision side by the decision under O2:** the operative denominator for every vision-side claim is $N_{vision} = 4$ (physiology is available for all 56). All remaining $N=55$ / $N=15$ language in this document refers to the physiology cohort only, and the text should be relabelled accordingly so the two denominators are never conflated again.

**Relabelled (2026-09-23, author-approved):** every remaining $N=55$ / $N=15$ figure in this document is now explicitly marked as a physiology-cohort number via parentheticals (Power Planning Fallback, the 2026-09-23 amendment, O2, O6). Verbatim quotations retain their original wording and inherit the physiology-only qualifier through this note.

### O6. BLOCKING - the pre-registered inferential plan cannot run at the achievable cohort size

Derivation from this document's own pre-registered rules, using $N_{vision} = 4$ (final - see O2). The $N=7$ figures are retained below only to record what was considered and rejected:

* The primary estimator is a **t-interval on per-subject paired differences**, with each subject weighted equally, and a subject's repetitions averaged *before* the t-test over subjects. The effective sample size is therefore the number of subjects, not the number of windows. Window-level pooling does not rescue this, because the aggregation rule is pre-registered.
* At $N=4$ this gives $\mathrm{df}=3$, $t_{0.95,3} = 2.353$; at $N=7$, $\mathrm{df}=6$, $t_{0.95,6} = 1.943$.
* The TOST equivalence rule requires the 90% CI to lie entirely inside $[-2.0, 2.0]$, i.e. half-width $< 2.0$. Solving $t_{0.95,n-1} \cdot s_d / \sqrt{n} < 2.0$: the paired-difference SD $s_d$ would have to be **below 1.70** accuracy points at $N=4$, or **below 2.72** at $N=7$. For reference, the same bound at the abandoned $N=15$ (physiology-only cohort) is $s_d < 4.40$.
* The document already contains the governing escape hatch. The **Power Planning Fallback** states that if predicted power is under 80% at evaluation $N=15$ (physiology-only) the run proceeds as **estimation-only, with no equivalence claim**. At $N = 4$ that condition is triggered by inspection, and the per-subject SDs required to avoid it (1.70 accuracy points) are implausibly small for this task.

**Consequence - arithmetic, not preference:** the K1a/K2b-vs-EQARNB equivalence framing is **not viable** and should be withdrawn, not merely reported as underpowered. The study falls back to the estimation-only path this document already defines.

**Do not treat the pivot as settled by the above.** The pivot named at the end of this document - a pure benchmark / analysis paper of the synthetic routing mechanism, or migration to a dataset with informative vision - is a thesis-scope decision, not an arithmetic consequence. The arithmetic closes off the equivalence path; it does not choose the replacement. That choice implies a different chapter structure and a different set of claims defensible at defence, and it belongs to the author (with the advisor). It is recorded here as a live option, not a recommendation already adopted.

**Also invalidated at this size:** the >15% cohort invalidation trigger on anchor stability is meaningless against ~4-7 subjects (15% of 4 is 0.6 subjects), and the $\rho \in \{0.85, 0.5, 0.15\}$ arms cannot be constructed at all until a confounder injector exists (O1) and a pool exists to bias (O4).

**FINAL DECISION (2026-09-23, author, confirmed after full consideration including timeline):** the thesis **pivots to the benchmark / limitations framing**. The vision branch will not be run as an equivalence study, and no further vision-side training run is required on this line. The negative result already in hand - the leakage test failing at both smoothing settings, together with the power analysis above - becomes the reported finding rather than a failed prerequisite. Draft writeup: `docs/negative_result_writeup.md`.

**AMENDED 2026-09-23 (same day, author):** the benchmark arm of this pivot is now operationalized. `data/publishable_scar_production/multimodal_publishable.csv` is designated the controlled-confounder benchmark corpus, with permitted-claim scope, validity gates, and required discipline defined in the amendment immediately below. The equivalence withdrawal above is **unchanged**.

---

## Pre-Registration Amendment (2026-09-23): O6 Rescope - The Publishable Corpus as the Controlled-Confounder Benchmark

**This amendment operationalizes the benchmark arm of the pivot recorded under O6 and partially resolves the open question in the Data Loss #2 scope note. It does not reopen the equivalence study, does not amend the UBFC-Phys negative-result line (`docs/negative_result_writeup.md` stands as written), and does not rehabilitate any multimodal-10k-lineage artifact.**

### Designation

`data/publishable_scar_production/multimodal_publishable.csv` (SHA-256 `2035777f957fa4c5cc830989f4fe40fb9190589f17c567cb29a3ef0e6b2acae2`, sidecar committed as `.sha256`) is designated the **controlled-confounder benchmark (CCB) corpus**.

Its verified properties are recorded in section A of the Session Review Findings above: 3,344 rows; 418 faces x 8 WESAD windows; exact paired `scarred_path` / `counterfactual_image_path` per row with sha256 provenance; embedded disjoint train/val/test splits with zero overlap on both `face_id` and `physiology_subject`; realized bias P(scar=1|threat=1)=0.850, P(scar=1|threat=0)=0.150, uniform across splits. The embedded `split` column is the operative split; the split-honoring branch of `src/train_cgf_fair.py` is the code path that reads it.

### Why the zero-true-signal premise is earned here, and why that is the point

Under UBFC-Phys the premise had to be earned against the POS leakage test, and the test failed. Under the CCB the premise is true **by construction**: `face_id` and `physiology_subject` are unrelated by construction (0% identity match; every face is crossed with 2-8 physiology subjects; zero cross-split subject overlap). A stranger's portrait cannot carry the subject's label. This is not a defect of the benchmark - it is the benchmark's design. The rendered scar at controlled rho is the **only** vision-label association in the data, so any vision utilization the model exhibits is, by construction, confounder utilization. The CCB is therefore the correct instrument for the claim the pivot actually makes - characterizing when multimodal architectures suppress vs. exploit a synthetic shortcut - and for nothing else.

### Permitted claims (descriptive, within-corpus)

* Counterfactual consistency (`cf_gap`, already computed by `eval_metrics`).
* The **counterfactual flip rate**: over scar=1 rows, the fraction of predictions that change between `scarred_path` and its exact paired clean image. This is the exact executable form of the flip test sketched under K1's shortcut-confirmation note; the exact pairing makes it exact rather than approximate. Requires a small, committed evaluation addition before the freeze.
* Group gaps (DP/EO by scar group), HSIC(v_c, S) probes per the K4 estimator spec, and the K5-style inference-time gate ablation (forced G in {0, 1}).
* Architecture comparisons (ERM/concat vs CGF vs fair-constrained) on the above, reported descriptively.

### Non-permitted claims

* Any TOST/equivalence claim (withdrawn under O6; the df arithmetic is unchanged, and the withdrawn machinery must not be silently reused).
* Any claim that CCB results say anything about stress classification from real faces, or about the UBFC-Phys line. The two lines share no data.
* Any population-level inferential claim about subjects: all CCB metrics are descriptive statistics over this corpus's fixed splits.
* Any artifact-probe or realism claim until the sham gate below is met. Realism claim scope remains governed by `docs/SCAR_ARTIFACT_VALIDATION_PROTOCOL.md`.

### Binding validity gate: the sham condition

The CCB currently has no sham-edit condition (verified: the builder renders scars only, generator version `scar-like-renderer-2.0`). Without shams, "scar vs. clean" is trivially separable and the pre-registered boundary-ring artifact probe is meaningless - the probe requires scar vs. sham. **Any "the confounder is not a low-level boundary artifact" claim is blocked until a sham condition matched on size and location distributions is added to the builder, the corpus is regenerated and re-hashed, and the probe passes on the regenerated corpus.** Whether the benchmark chapter makes such a claim at all is an author choice; if it does, this gate is binding.

### Scope of the rho arms

The operative CSV is rho=0.85 only (`rho_target` = 0.85 for all rows). The builder renders both clean and scarred variants for every face and assigns scar labels stratified by threat, so rho in {0.5, 0.15} variants are regenerable via the builder's `--rho` argument without re-rendering images. Each rho variant is a distinct corpus with its own hash and its own freeze; any multi-rho benchmark claim requires each variant generated, hashed, and recorded in this document before any run.

### Required discipline before any CCB run

1. Startup hash assertion against the sidecar; mismatch aborts the run.
2. Smoke test (one epoch, a few hundred samples) with manual inspection of rendered pairs.
3. Commit the builder and the training/evaluation scripts used, so the run is rerunnable from source (the Data Loss #2 lesson).
4. Any threshold that converts a measurement into a claim (e.g., what counts as "exploits the confounder") must be pre-registered in a dated amendment before the freeze.

### Disposition of the descoped items

* **O1** (UBFC confounder injector): remains out of scope; the CCB does not run on UBFC crops.
* **O3** (manipulation-check threshold): remains descoped; no K2a manipulation check exists in benchmark form. If a multi-rho sweep is run, "does confounder utilization scale with rho" is a descriptive benchmark result, not the pre-registered manipulation check.
* **O4** (K1-K5 re-derivation for UBFC): remains descoped for UBFC; K-battery names are not reused for CCB measurements. The permitted-claims list above is the CCB's operative definition.
* **O6 equivalence withdrawal**: unchanged. The estimation-only / negative-result line is unaffected.

### Deliberately not decided here

* Chapter structure and the defence-facing claim hierarchy for the CCB chapter (author + advisor, per O6).
* Single-rho (0.85) vs multi-rho reporting.
* Whether an artifact-probe claim is included at all (blocked by the sham gate until met, regardless).


