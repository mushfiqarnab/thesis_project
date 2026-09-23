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
* If predicted power < 80% (using the upper 80% bound on the larger of the two contrast SDs) at the evaluation $N=15$, the WESAD run proceeds as estimation-only, with no equivalence claim.

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
3. **Outlier Acknowledgement & Full-Scale Monitoring**: Face-normalized coordinate validation on the dev set ($n=4$) confirmed no consistent group-level phase-correlated drift. Subjects `s2`, `s3`, and `s4` exhibited bidirectional noise (Cohen's d < 0.5). However, `s1` exhibited a massive, anomalous phase-correlated shift (d = -2.26). Because $n=4$ is underpowered to definitively distinguish a mechanical detection flaw from a true subpopulation behavioral effect, we adopt the 2D anatomical interpolation now (explicitly rejecting full 3D pose reprojection as out of scope) but formally commit to recalculating and reporting this face-normalized anchor stability metric across the entire $N=55$ cohort as a documented monitoring step. **Concrete Trigger:** If >15% of the full cohort exhibits an absolute shift of |d| > 1.0 on this face-normalized metric, the anatomical anchor choice is officially invalidated and will be revisited before model training begins.

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

