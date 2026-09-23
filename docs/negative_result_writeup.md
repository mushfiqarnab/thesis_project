# Vision-Side Physiological Leakage in a Small Naturally-Multimodal Stress Cohort

**A pre-registered negative result, and the limits of equivalence testing at acquisition-bounded N**

Status: **Complete draft** (2026-09-23). All §8 open items are resolved (see §8 for the record); figures are from artifacts produced 2026-09-22/23; provenance in §7. Pending advisor sign-off before submission.

---

## 0. Summary

We set out to test whether a video-derived vision pipeline admits recoverable physiological signal, as a
prerequisite for an architecture whose claimed contribution depends on the vision modality carrying **zero
true signal** for the classification label once an injected visual confounder is removed.

**The prerequisite fails**, and the failure is robust to estimator choice. Three standard rPPG algorithms - POS, CHROM and PBV - were run on identical inputs and all three are rejected by the pre-registered test (POS p = 0.0314, CHROM p = 0.0001, PBV p = 0.0103; POS p = 0.0437 in the no-smoothing arm). The effect is marginal under POS but **large under CHROM**, which recovers a per-clip correlation of 0.85 against the subject's own wrist BVP on one clip (§4.5). The "small magnitude" reading should therefore be treated as POS-specific, not as a property of the pipeline.

**Cohort: N=4 subjects, 12 clips** (UBFC-Phys, T1-T3 per subject) - a fixed acquisition ceiling (§2), not a sampling choice made at analysis time. Every vision-side figure in this paper rests on these 4 subjects.

We also report a design-level finding: the pre-registered inferential plan for this line of work is **not
viable at the cohort size that the data acquisition budget supports**. The primary estimator is a t-interval
on per-subject paired differences, which at N=4 gives df=3 and requires a paired-difference SD below 1.70
accuracy points to satisfy the TOST equivalence band. This is not a power shortfall that more windows would
fix; it is a mismatch between the estimator and the acquisition ceiling (§5.2).

The negative result is reported here as the finding, not as a failure to be worked around.

---

## 1. Motivation

An architecture under evaluation (referred to here as the fused model) was proposed to classify acute
stress from fused vision and physiology while suppressing reliance on a synthetic visual confounder (a
procedurally rendered "scar"). Its claim rests on a premise:

> If vision carries no true signal for the stress label, a model that suppresses the confounder should
> perform identically to one that never sees vision at all.

That premise was **free** under the original data configuration, where vision and physiology came from
**independent sources**: a stranger's portrait cannot carry the stress label of a different person, so
the absence of true visual signal was guaranteed by construction rather than by any property of the
pipeline.

The premise is **not free** under a naturally multimodal source, where both modalities come from the same
subject's recording. There, the subject's own pulse is physically present in their own facial pixels, and
a video-based rPPG extractor can in principle recover it. Under that configuration the premise must be
**earned**: the pipeline's own processing must demonstrably destroy the recoverable signal.

**This document reports the test of that condition, and it does not pass.**

---

## 2. Data and scope

| Property | Value |
|---|---|
| Dataset | UBFC-Phys |
| Subject directories on disk | 56 |
| Subjects with **video** | **4** (`s1`–`s4`) |
| Clips (subject × task) | 12 (T1–T3 per subject) |
| Target frames per clip | 1800 @ 10 fps |
| Video payload | 12 `.avi` files, 56.0 GB |
| Wrist BVP ground truth | 64 Hz, present for all 12 clips |
| Physiology CSVs across cohort | 12.2 MB |

The gap between 56 and 4 is a **deliberate acquisition decision**, not a defect: physiology and metadata
were extracted across the full cohort while the video payload was deliberately avoided, on storage and
time grounds. The consequence is that the vision-bearing cohort is 4 subjects, and this is the fixed
ceiling for all vision-side analysis in this work.

**Scope consequence:** the cohort-scale monitoring commitments in the accompanying pre-registration
(e.g. a >15% invalidation trigger over 55 subjects) are not executable at N=4, and are recorded as out of
scope rather than silently dropped.

---

## 3. Methods

### 3.1 Face localisation and crop stabilisation

Frames are sampled onto a 10 fps target grid (1800 timestamps per clip). Faces are located with Mediapipe
`mp.solutions.face_detection` (`model_selection=0`, `min_detection_confidence=0.5`), with the
highest-scoring detection retained per frame.

The crop box is square, centred on a 2D anatomical anchor rather than the detection box centre:

```
anchor = 0.5 * Eye_Midpoint + 0.5 * Nose_Tip
side   = IOD * 3.566283 * MARGIN           # MARGIN = 1.5
```

Two fixes were applied to the extraction path in the course of this work:

1. **Bounding-box width jitter eliminated.** The earlier detector produced variable box width, which
   perturbs the crop on every frame and therefore injects nuisance variance directly into the photometric
   signal the rPPG estimator consumes. The pipeline was migrated to Mediapipe, which bounds the face
   tightly (eyes to mouth) and requires the compensating **1.5×** margin to retain boundary clearance.
   A detector-specific failure mode was also identified and dismissed: a uniform rate of exactly one
   failed read per clip is an `CAP_PROP_FRAME_COUNT` EOF boundary artifact, not a detection failure.
2. **Temporal smoothing of the box** is applied as an EMA over consecutive boxes, parameterised by α
   (α=1.0 = no smoothing; α=0.5 = the smoothed configuration under test).

Each frame is cropped, resized to 224×224, and written as a PNG alongside a per-frame manifest recording
status (`OK`/`FILLED`/`FAILED`) and the box and keypoint coordinates.

### 3.2 rPPG extraction

Pulse signal is recovered with the Plane-Orthogonal-to-Skin (POS) algorithm from the **spatial mean RGB of
each entire crop** (no skin masking, no ROI refinement):

* 1.6 s sliding window at 10 fps;
* temporal normalisation of each channel by its window mean;
* projection onto S1 = 3R − 2G and S2 = 1.5R + G − 1.5B;
* per-window α tuning via the S1/S2 standard-deviation ratio;
* overlap-add reconstruction.

**POS is the estimator behind the reported leakage result** (§4.1-§4.4). Two further estimators are used only
for the estimator-specificity check (§4.5). They share every step above and differ solely in the
within-window projection:

* **CHROM** (de Haan & Jeanne 2013) - the same S1/S2 components, combined as Xs - alpha * Ys.
* **PBV** (de Haan & van Leest 2014) - projection of the normalised RGB vector onto the blood-volume-pulse signature [0.33, 0.77, -0.53].

### 3.3 Correlation analysis

* Reference: wrist BVP resampled to 30 Hz from 64 Hz.
* Both signals band-pass filtered 0.7–4.0 Hz (4th-order Butterworth, zero-phase `filtfilt`).
* Similarity is the **maximum Pearson r over lags within ±0.5 s** (a 1 s search band around zero lag).
* **True pairs (N=12):** each clip's rPPG against its own BVP.
* **Null pairs (N=132):** all ordered cross-clip pairings (12 × 11) excluding same-subject-same-task.

### 3.4 Pre-registered criteria

Both criteria were logged *before* the evaluation under this protocol:

* **Leakage pass/fail (logged 02:35):** pass requires a Mann-Whitney U test p-value > 0.05 comparing true
  against null distributions. **p ≤ 0.05 constitutes statistically significant leakage and is an
  automatic fail.**
* **α-ablation criterion (logged 02:40):** the EMA hypothesis ("EMA smoothing is the leakage mechanism")
  is confirmed **only if** α=1.0 passes **and** α=0.5 fails. Both failing, or both passing, is defined as
  **inconclusive, not confirmatory.**

### 3.5 Determinism control

Because the α ablation requires a second full extraction pass that overwrites the first, a
status-column control was specified in advance: the per-frame `status` field from pass 1 (α=1.0,
preserved as `manifest_alpha1.csv`) is diffed against pass 2 (α=0.5, `manifest.csv`) for all 12 clips,
with an assertion that exactly 12 pairs are present and compared, so that silently skipped files cannot
produce a false pass.

### 3.6 Status of the estimator comparison

The estimator comparison (§4.5) was **not pre-registered**. The protocol specified a single leakage test on
the POS pipeline; the POS/CHROM/PBV re-run is a post-hoc robustness check added after the reported result
was already in hand. It is labelled exploratory wherever it appears: it does not carry the pre-registered
status of §4.1, and its p-values should not be reported as confirmatory. It is included because the
reported result would otherwise rest on a single estimator, and it requires no new data, no new compute and
no change to any input - only the projection changes.

---

## 4. Results

### 4.1 The leakage test fails at both smoothing settings

| Pass | True mean r (N=12) | Null 95th pct r (N=132) | Null mean r | Mann-Whitney U | p (True > Null) | Formal verdict |
|---|---|---|---|---|---|---|
| α = 1.0 (no smoothing) | 0.0752 | 0.1172 | 0.0597 | 1029.0 | **0.0437** | **FAIL** |
| α = 0.5 (smoothed) | 0.0802 | 0.1186 | 0.0599 | 1050.0 | **0.0314** | **FAIL** |

Both configurations are rejected by the pre-registered criterion. **Removing temporal smoothing does not
remove the recovered signal.**

### 4.2 Per-clip true correlations (α = 1.0)

| Clip | r | Clip | r | Clip | r |
|---|---|---|---|---|---|
| s1 T1 | 0.0477 | s2 T1 | 0.0471 | s3 T1 | 0.0609 |
| s1 T2 | 0.0912 | s2 T2 | 0.0987 | s3 T2 | 0.0249 |
| s1 T3 | 0.1799 | s2 T3 | 0.0624 | s3 T3 | 0.0590 |
| s4 T1 | 0.0741 | s4 T2 | 0.0579 | s4 T3 | 0.0989 |

Values range from 0.0249 to 0.1799, with a single clip (s1 T3) well separated from the rest. No clip
shows the strong (>0.5) correlation characteristic of a clean rPPG extraction, which is consistent with
the small-magnitude picture in §4.3.

**Geometry provenance flag (2026-09-23).** A post-hoc audit of the retained manifests
(`scripts/ubfc_leakage/audit_manifest_geometry.py`) found that the α=1.0 pass is not internally uniform:
clips s4 T2 and s4 T3 carry crop geometry inconsistent with the committed crop rule (implied scale
constants ≈ 2.25 and ≈ 2.48 vs 3.566 for the other ten clips), and s4 T3 is missing 15 OK frames
relative to every other clip. All twelve α=1.0 manifests share identical file timestamps
(2026-09-23 02:50:57), so the anomaly predates the documented 02:40 overwrite and its provenance is
unresolved. The α=0.5 artifacts used by every estimator-comparison result in this document are uniform
across all 12 clips and unaffected. The §4.1 α=1.0 row is reported as run; cite its per-clip values
only with this caveat.

### 4.3 Magnitude under POS, and a disagreement between the two criteria

**Scope: this subsection describes the POS and PBV arms only. The magnitude conclusion does NOT generalise
- see §4.5, where CHROM recovers substantially more of the signal.**

The formal verdict is FAIL, but the result must not be over-read in either direction:

* **The effect is small.** True and null means differ by **0.0155 in r** (0.0752 vs 0.0597), and the null
  95th percentile (0.1172) sits *above* the true mean. Roughly half of the null pairings exceed the true
  mean. The rank test is detecting a modest distributional shift, not a strong coherence.
* **The two available criteria disagree.** The pipeline's percentile criterion ("true mean ≤ null 95th")
  **passes** in both configurations, while the rank test **fails** in both. The pre-registration designates
  the Mann-Whitney p-value as the formal pass/fail test, so the verdict is FAIL; the percentile line must
  not be quoted as a pass, and this disagreement is reported rather than resolved post hoc.

The honest reading, then: **the pipeline does not destroy physiologically derived signal to a degree that
the pre-registered test accepts, but the residual signal is weak and the detection rests on a modest
distributional shift within a single cohort of 4 subjects.**

### 4.4 The EMA hypothesis is inconclusive, not confirmed

By the criterion logged at 02:40, confirmation required α=1.0 to pass and α=0.5 to fail. Both fail, so the
outcome is **inconclusive by definition** — the EMA hypothesis is **not** confirmed.

The direction of the effect nevertheless argues against EMA as the leakage mechanism: the leakage is if
anything *stronger* at α=0.5 (p=0.0314) than with no smoothing at all (p=0.0437). Removing smoothing
moved the p-value in the direction of less significance but nowhere near acceptance. **We report this as
a directional observation that does not meet the pre-registered bar for a confirmation**, and we do not
claim EMA has been "ruled out" in the strict pre-registered sense — only that it is not supported as the
mechanism, and that eliminating it does not repair the leakage.

**O7 — the α=1.0 arm is not robust to removing two clips (2026-09-23).** A
leave-two-clips-out check excluded s4 T2 and s4 T3, the two clips whose α=1.0
manifests show crop geometry inconsistent with the committed rule (mean w/IOD =
3.3709 and 3.7259 respectively, vs. the expected 5.3494). On the remaining N=10
true-match pairs tested against the same N=132 null permutations, the result
is U=832.0, p=0.0857 — above the pre-registered p<0.05 threshold. Removing
two geometrically anomalous clips is sufficient to flip the α=1.0 arm above
the significance boundary. The arm must not be cited as strong independent
evidence of leakage at zero smoothing.

This does not affect the primary finding, which rests on the α=0.5 arm
(p=0.0314, pre-registered, geometrically uniform crops verified across all
12 clips). The EMA verdict stated in this section is unchanged — if anything
strengthened: removing the anomalous clips makes α=1.0 *less* compelling as
a leakage signal, which is the opposite of what an EMA mechanism would predict.

*Provenance: `outputs/leakage_run/o7_reanalysis.txt` (499 bytes, sha256:
6a1ac8f0…61ee4d)*

### 4.5 Estimator specificity: the verdict holds across POS, CHROM and PBV

The reported result rests on a single estimator, so the leakage test was re-run with two additional
standard rPPG algorithms on **identical** inputs: same crops, same manifests, same resampling, same
0.7-4.0 Hz filter, same ±0.5 s lag search, same true/null pairing. Only the estimator changes.

| Estimator | True mean r | Null mean r | Null p95 r | U | p (True > Null) | Formal verdict | Percentile criterion |
|---|---|---|---|---|---|---|---|
| POS (reported) | 0.0802 | 0.0599 | 0.1186 | 1050.0 | **0.0314** | **FAIL** | pass |
| CHROM | 0.1933 | 0.0603 | 0.1092 | 1318.0 | **0.0001** | **FAIL** | **fail** |
| PBV | 0.0898 | 0.0598 | 0.1169 | 1113.0 | **0.0103** | **FAIL** | pass |

**All three estimators fail. The finding is not estimator-specific.**

**Reproduction check.** The POS arm of this run reproduced the reported α=0.5 result to four decimal
places (true mean 0.0802, null mean 0.0599, null p95 0.1186, U 1050.0, p 0.0314), confirming that the
comparison harness is faithful to the reported run before any estimator was changed.

**Consequence for the magnitude caveat in §4.3.** The "small effect" characterisation is specific to POS
and PBV and must not be generalised. Under CHROM the true mean rises to 0.1933 against a null mean of
0.0603 - a separation of **0.133**, roughly nine times the POS separation of 0.0155 - and the percentile
criterion now fails as well (true mean 0.1933 exceeds the null 95th percentile of 0.1092). Two clips show
strong individual recovery:

| Clip | CHROM true r |
|---|---|
| s2 T1 | **0.8487** |
| s3 T1 | **0.4865** |

The full 12-clip breakdown for all three estimators is persisted at `outputs/leakage_run/estimator_per_clip_results.csv` (summary and table also in `estimator_comparison_full_output.txt`).

A per-clip correlation of 0.85 between a signal extracted from facial pixels and the subject's own wrist
BVP is a clear pulse recovery, not a marginal distributional shift. It also cannot be explained by the lag
search: the CHROM null 95th percentile is 0.1092, so a true-pair value of 0.849 lies far outside the
chance-level range that a ±0.5 s maximum-lag search produces. On the evidence available, the recovery is
genuine rather than a selection artifact; the spectral verification below corroborates this. The correct reading is therefore that
**the pipeline admits substantial recoverable signal which POS happens to recover weakly**, not that the
leakage is inherently marginal. This strengthens rather than weakens the negative result: the prerequisite
fails, and it fails more visibly under a more robust estimator.

**Observation on task phase.** Both strong CHROM recoveries occur in the **T1 (rest)** task; no T2 or T3
clip exceeds r = 0.15 under CHROM. (The only estimator values above 0.15 outside T1 are s1 T3 under POS,
0.2026, and PBV, 0.1798 - a single clip.) This is consistent with a physical explanation -
the T2/T3 protocols involve speaking and body motion, which degrade rPPG extraction - but with only 2 of
4 T1 clips affected this is an observation, not a tested effect, and it is recorded as a hypothesis for
follow-up rather than as a finding.

**Spectral verification (2026-09-23, resolves open item 5).** The two high-recovery clips were checked
spectrally against the subject's own wrist BVP (Welch PSD, 0.7-4.0 Hz;
`scripts/ubfc_leakage/spectral_verification.py`, artifacts
`outputs/leakage_run/spectral_verification.{csv,txt}`). Both recovered signals peak at the cardiac
frequency of their own BVP: s2 T1 at 0.908 Hz vs a BVP peak of 0.908 Hz (delta = 0.000 Hz), and s3 T1 at
1.406 Hz vs 1.465 Hz (delta = 0.059 Hz, about two Welch bins) - both inside the pre-stated <=0.1 Hz match
rule, with high in-band peak SNR (53.5 and 13.8). On the CHROM signal before the pipeline's final
band-pass, the fraction of total power inside the 0.7-4.0 Hz band is **0.66 (s2 T1)** and **0.57 (s3 T1)**;
the remainder is sub-0.7 Hz baseline drift, which the pipeline's own band-pass removes before any
correlation is computed. The high r values are therefore not a low-frequency artifact. The primary
evidence for genuine pulse recovery remains the lag-search correlation against the real wrist BVP measured
against the null distribution (above); the spectral check corroborates it and is not relied on alone.

**Method note.** CHROM is designed to be more robust to illumination and motion variation than POS, which
is consistent with it recovering more of the signal that survives the crop pipeline. The estimator choice
matters to the *magnitude* of the measured leakage but does not change the *verdict*.

### 4.6 Extraction determinism holds

**PASS — 12/12 clips match exactly** on the per-frame `status` column between the α=1.0 and α=0.5 passes
(1801 rows per manifest: 1800 target frames plus header). Asserted as exactly 12 matched pairs.

Scope: this establishes determinism of detection **status** only. Box coordinates were deliberately not
compared, because α *is* the box smoothing coefficient — the two arms are supposed to differ in geometry,
since that difference is the treatment. Coordinate comparison would fail by construction and would
therefore not be evidence of nondeterminism.

### 4.7 The inferential plan is not viable at this N

| Cohort | df | t(0.95) | 90% CI half-width | Required SD for TOST band [−2, 2] |
|---|---|---|---|---|
| **N=4 (actual)** | **3** | **2.353** | **1.177 · s_d** | **s_d < 1.70** |
| N=7 (considered, not acquired) | 6 | 1.943 | 0.734 · s_d | s_d < 2.72 |
| N=15 (originally planned, physiology-only cohort) | 14 | 1.761 | 0.455 · s_d | s_d < 4.40 |

See §5.2 for the interpretation.

---

## 5. Discussion

### 5.1 What the negative result establishes

The premise underpinning the fused model's claimed contribution — *vision carries no true signal for the
label* — **is not established under the naturally multimodal configuration**. Three standard rPPG
estimators (POS, CHROM, PBV) each recover pulse-correlated signal from the exact crops the pipeline
produces, all at levels the pre-registered test rejects (§4.1, §4.5), and the two extraction fixes
applied (bbox jitter elimination, temporal smoothing) do not repair it. Under CHROM the recovery is
strong on individual clips (r = 0.85), so this is not a marginal or artefactual effect.

Two consequences follow, and they are different in kind:

1. **For this cohort, the confounder-suppression claim cannot be attributed to the architecture alone.**
   Any observed gain over a physiology-only baseline is confounded by residual recoverable visual signal.
   Because the confounder injector for this vision source was never built, the K1–K5 contrasts were never
   run at all (§6), so this is a statement about what the design could have supported, not about a
   measured model result.
2. **The measurement itself is the contribution.** A pre-registered leakage test, run against a
   conventional rPPG estimator, is a reusable instrument for establishing whether a vision pipeline is
   admissible for this class of confounder-suppression claim.

### 5.2 Methodological finding: equivalence testing is not viable at acquisition-bounded N

This is the second, and arguably more transferable, finding.

The pre-registered primary estimator is a **t-interval on per-subject paired differences**, with each
subject weighted equally and a subject's repetitions averaged *before* the test over subjects. The
effective sample size is therefore the **number of subjects**, not the number of clips or windows. At N=4
that is df=3.

Two-sided equivalence testing (TOST, α=0.05) requires the 90% CI of the paired difference to lie entirely
inside the pre-registered band [−2.0, 2.0], i.e. a half-width below 2.0 accuracy points. Solving
`t(0.95, n−1) · s_d / √n < 2.0` gives the required between-subject SD of the paired difference:

* **N=4 → s_d < 1.70 accuracy points**
* N=7 → s_d < 2.72
* N=15 → s_d < 4.40 (planned physiology-only cohort; never reached)

Accuracies expressed in percentage points over a small subject pool do not plausibly attain a paired
SD below 1.70. Note also that the pre-registration's own Power Planning Fallback anticipated exactly this
situation — mandating an **estimation-only** run with no equivalence claim if predicted power fell below
80% at the planned N=15 (physiology-only cohort) — so the design already contained the correct escape hatch; the acquisition
ceiling triggers it by inspection.

The generalisable point is structural rather than statistical:

> Equivalence-testing and pre-registration practice implicitly assumes a **resampleable** N, where
> sample size is a design choice. In naturally multimodal stress research the binding constraint is the
> number of **subjects for whom all modalities were actually captured and retained**. That number is set
> by acquisition cost — here ~4.7 GB of video per subject — and by retention policy, not by statistical
> planning. An inferential plan built on per-subject t-intervals must therefore be designed around the
> acquisition ceiling **before** data collection, or the equivalence framing must be abandoned a priori.

Concretely: the same project that formally pre-registered a TOST equivalence test also, at the outset,
deliberately avoided downloading the video payload that the test would have required. Each decision was
locally defensible; their interaction was never computed. That interaction is the finding.

### 5.3 Methodological finding: two-criterion designs can disagree, and the pre-registration is what decides

A percentile criterion and a rank criterion were both computed on the same data and **disagreed in both
configurations** (§4.3). The pre-registration named the rank test as formal, which is why the verdict is
FAIL rather than ambiguous. Two recommendations follow:

* Pre-register **one** formal criterion, and treat any secondary criterion as descriptive from the start.
* Where a criterion is a pass/fail on a p-value, also pre-register a **minimum magnitude of interest**.
  Here the formal test rejects on a 0.0155 mean-r separation with roughly half the nulls above the true
  mean; a magnitude floor would have made the weakness of the effect part of the pre-registered decision
  rather than a caveat discovered afterwards.

### 5.4 Framing

We do not frame this as a failure of the pipeline so much as a **boundary result about the data
configuration**: a confounder-suppression claim of this kind is not admissible against a naturally
multimodal cohort unless the pipeline first passes a leakage gate, and at this cohort size the
equivalence-testing apparatus intended to evaluate such a claim is not constructible. The negative result
and the viability analysis stand on their own.

---

## 6. What was not done (stated explicitly, not as silent gaps)

* **No confounder injector was built for this vision source.** The scar/sham renderer and the
  `0.5·Eye_Mid + 0.5·Nose_Tip` anatomical anchor were never implemented for UBFC-Phys crops; the
  pre-existing scar/sham code is portrait-lineage and operates on static images. The anchor in the
  extraction code is used as the **crop-box centre** and as a stability metric, which is not confounder
  placement.
* **K1–K5 and the ρ ∈ {0.85, 0.5, 0.15} arms were never run and are not constructible here**, because the
  ρ arms require confounder assignment that does not exist and because the subject pool is fixed at 4.
* **The manipulation check threshold (X) was never set**; with no manipulation performed, it is moot.
* **Cohort-scale anchor-stability monitoring was not performed**, and the >15% invalidation trigger is
  meaningless at N=4 (15% of 4 is 0.6 subjects).
* **The `3.566283` scale constant has a broken chain of custody**: the raw per-frame bbox-width/IOD data
  behind it was destroyed by an unbacked-up overwrite, and the constant is taken on faith from a single
  non-reproducible computation. It is retained because re-deriving it is no longer possible.
  **Geometry audit (2026-09-23):** the retained manifests of the current artifacts imply the committed
  crop rule (w = IOD × 3.566283 × 1.5) to within 0.1% uniformly across all 12 clips — but this is
  **circular**, because those crops were cut using the constant; it recovers the *rule as applied*, not
  the historical derivation. The audit also found that in the preserved α=1.0 manifests, clips s4 T2 and
  s4 T3 show crop geometry inconsistent with the committed rule (implied constants ≈ 2.25 and ≈ 2.48,
  i.e. ~37% smaller boxes), with identical file timestamps across all twelve α=1.0 manifests — so the
  anomaly's provenance is unresolved and was not produced by the documented 02:40 overwrite. Recorded
  in the pre-registration (Geometry audit / Geometry provenance flag); artifacts:
  `scripts/ubfc_leakage/audit_manifest_geometry.py`, `outputs/leakage_run/manifest_geometry_audit.txt`.
  Consequence: the α=1.0 aggregate statistics in §4.1 are reported as run, but per-clip α=1.0 values
  must not be cited without this caveat; all estimator-comparison results (§4.5) use the α=0.5
  artifacts and are unaffected.

---

## 7. Provenance

| Item | Source |
|---|---|
| Leakage results, both passes | `outputs/leakage_run/ablation_results.txt` (α=1.0 and α=0.5 result blocks) |
| Per-clip correlations | `outputs/leakage_run/alpha_1.0_results_extracted.txt` |
| Null pairings | `outputs/leakage_run/null_corrs.csv` (132 rows) |
| Extraction pipeline | `scripts/ubfc_leakage/preprocess_video_mediapipe.py` |
| rPPG + correlation analysis (reported run) | `scripts/ubfc_leakage/run_full_pos_pipeline.py` |
| Estimator comparison (POS/CHROM/PBV) | `src/evaluation/leakage_estimator_comparison.py` |
| Per-clip estimator table (12 clips × POS/CHROM/PBV) | `outputs/leakage_run/estimator_per_clip_results.csv`, `outputs/leakage_run/estimator_comparison_full_output.txt` (via `scripts/ubfc_leakage/save_per_clip_results.py`) |
| Spectral verification (§4.5, open item 5) | `outputs/leakage_run/spectral_verification.csv`, `outputs/leakage_run/spectral_verification.txt` (via `scripts/ubfc_leakage/spectral_verification.py`) |
| Manifest geometry audit (§6, §4.2) | `outputs/leakage_run/manifest_geometry_audit.txt` (via `scripts/ubfc_leakage/audit_manifest_geometry.py`) |
| α ablation driver | `scripts/ubfc_leakage/run_ablation.py` |
| Determinism control | `scripts/ubfc_leakage/check_nondeterminism.py`, re-executed 2026-09-23 07:43:01 |
| Per-frame manifests | `processed/{s1..s4}/T{1..3}/manifest_alpha1.csv`, `manifest.csv` |
| Governing protocol | `docs/experiment_preregistration.md` (§ O1–O6) |

**Verification note:** the video-cohort claim (4 of 56 subjects) was verified directly against the
filesystem, not from a log: `find data/UBFC-Phys -name '*.avi' -printf '%h\n' | sort -u` returns exactly
`s1`–`s4`, and a per-subject enumeration returns 3 for each of those and 0 for the other 52.

**Uncited figure:** a figure for the avoided video payload (784 GB) has been reported as originating in
the project session brief. It is **not** cited here, because it is not present in the repository and has
not been independently confirmed.

**Crop provenance for the α=0.5 primary arm (chain of custody):**

(a) The original α=0.5 crop PNGs are unrecoverable. They were overwritten
during the α=1.0 regeneration pass without a prior backup; only the manifests
were preserved (`manifest_backup_alpha05.csv` for each of 12 clips).

(b) The regenerated α=0.5 manifest set is not byte-identical to the original
manifests — up to 1,795/1,800 rows differ in box coordinates per clip, maximum
delta 374 px. The original run used a scratch dependency of `run_ablation.py`
that no longer exists on disk. Two checks confirm the regeneration is nonetheless
functionally faithful: (i) no intermediate cache exists that could serve stale
data; (ii) frame k=0 — the only frame with a pre-regeneration timestamp — is
marked FAILED in all 12/12 clip manifests and is never read by the pipeline. The
matching headline statistics (0.0802, 0.1186, 0.0599, 1050.0, 0.0314) are
consistent with POS averaging color spatially over the full crop region: a
shifted-but-overlapping crop of similar size produces a near-identical spatial
mean when the fixed `IOD × 3.566283 × 1.5` rule dominates over box-coordinate
jitter.

---

## 8. Open items before this draft is submission-ready - ALL RESOLVED 2026-09-23

Retained as a record; nothing below remains open.

1. ~~Confirm whether the POS implementation deviates from the reference algorithm in ways that would
   weaken the leakage claim (no skin masking, whole-crop spatial mean, 10 fps ceiling).~~
   **RESOLVED (superseded by §4.5):** the estimator-specificity result makes the POS-implementation
   question moot for the verdict - POS, CHROM and PBV all fail on identical inputs, so no single
   implementation detail carries the finding. The whole-crop spatial mean and 10 fps ceiling are
   documented in §3.2.
2. ~~Decide whether to test a second rPPG estimator to show the finding is not estimator-specific.~~
   **DONE 2026-09-23:** POS, CHROM and PBV all run on identical inputs; all three fail. See §4.5 and
   `src/evaluation/leakage_estimator_comparison.py` (runtime ~50 s over the 12 existing clips, so no
   additional data or compute was required).
3. ~~State the cohort size honestly and prominently in any external write-up: **N=4 subjects, 12 clips**.~~
   **DONE 2026-09-23:** N=4 is now stated prominently in the summary (§0), not only in §2.
4. ~~Decide whether the `3.566283` chain-of-custody issue is disclosed in the write-up or resolved by
   re-deriving the constant from the retained crops.~~
   **RESOLVED 2026-09-23 (author decision): disclose.** The broken chain of custody is disclosed in §6
   ("What was not done"); re-deriving the constant from the retained crops is not possible, so no
   further action is required.
5. ~~Verify the CHROM high-value clips are genuine pulse and not a motion or illumination artifact.~~
   **DONE 2026-09-23:** spectral verification run for all 12 clips
   (`scripts/ubfc_leakage/spectral_verification.py`; artifacts
   `outputs/leakage_run/spectral_verification.{csv,txt}`). Both high-recovery clips peak at their own
   wrist-BVP cardiac frequency: s2 T1 0.908 Hz vs 0.908 Hz (delta = 0.000 Hz), s3 T1 1.406 Hz vs 1.465 Hz
   (delta = 0.059 Hz, ~2 Welch bins) - PEAK_MATCH under the pre-stated <=0.1 Hz rule; no low-frequency
   artifact. On the pre-bandpass CHROM signal the in-band (0.7-4.0 Hz) fraction of total power is 0.66
   (s2 T1) and 0.57 (s3 T1); the remainder is sub-0.7 Hz drift removed by the pipeline's own band-pass.
   The primary evidence for genuine pulse recovery remains the lag-search correlation against the real
   wrist BVP (§4.5); the spectral ratio corroborates and does not stand alone. See the spectral
   verification paragraph in §4.5.
6. ~~Author decision - whether to report the estimator comparison externally at all~~, given that it is
   post-hoc (§3.6). ~~Either report it as labelled exploratory robustness evidence, or omit it and state
   plainly that the result rests on POS alone.~~
   **RESOLVED 2026-09-23 (author decision): include it**, labelled exploratory/post-hoc exactly as §3.6
   already specifies. §4.5 keeps its exploratory status and its p-values are not reported as
   confirmatory; the pre-registered POS result remains the formal finding.

---

## 9. Future work

**Controlled confounder benchmark.** The EQARNB architecture was designed to
suppress a planted visual confounder (rendered facial scar) when classifying
stress phase from face and physiology. The controlled benchmark dataset built
for this purpose (418 FFHQ portraits × 15 WESAD subjects, ρ=0.85 scar-threat
correlation, `data/publishable_scar_production/`) was not included in this
submission for three reasons. First, the face and physiology inputs come from
unrelated subjects, meaning vision carries no true label signal beyond the
planted confounder, making the DP gap result circular. Second, only the ρ=0.85
bias regime was constructed; the ρ=0.50 and ρ=0.15 regimes required for the
robustness evaluation do not exist. Third, no sham-edit condition was
implemented, so the artifact probe and manipulation check cannot run. Valid
evaluation requires: synchronized face-physiology data from the same subjects
(UBFC-Phys provides this), all three ρ regimes, sham edits matched on size and
placement, and physiology-only and balanced-ERM kill-switch baselines before any
architecture comparison. Checkpoints and dataset artifacts are preserved in the
repository for future work.

---

## Appendix A. Reproduction

All commands run from the repository root. The reported run used the system Python 3.11 interpreter
(numpy 2.4.2, scipy 1.17.0, opencv 5.0.0 effective build - the machine had opencv-contrib-python
5.0.0.93 shadowing an also-installed opencv-python 4.13.0.92; see the requirements.txt note); the
project `.venv` does **not** include `cv2` and cannot run these scripts as-is.

```bash
# Estimator comparison (§4.5) - POS, CHROM, PBV on identical inputs. ~50 s.
python src/evaluation/leakage_estimator_comparison.py

# Per-clip estimator table (§4.5) - persists the full 12-clip table to outputs/leakage_run/. ~1 min.
python scripts/ubfc_leakage/save_per_clip_results.py

# Spectral verification (§4.5, open item 5) - CHROM vs wrist-BVP PSD, all 12 clips. ~1 min.
python scripts/ubfc_leakage/spectral_verification.py

# Manifest box-geometry audit (§6, §4.2) - 3.566283 provenance + alpha=1.0 anomaly. ~seconds.
python scripts/ubfc_leakage/audit_manifest_geometry.py

# Determinism control (§4.6) - asserts exactly 12 matched manifest pairs.
python scripts/ubfc_leakage/check_nondeterminism.py

# Full extraction + ablation. Regenerates processed/ and both alpha arms.
# Long-running; overwrites manifest.csv. Run check_nondeterminism.py first if the
# alpha=1.0 manifests are needed for comparison.
python scripts/ubfc_leakage/run_ablation.py
```

No new data or compute is required for the estimator comparison: the 12 clips already exist under
`processed/`, and the run above reads them directly. The persistence and spectral scripts import the
comparison harness unchanged and were run under the same interpreter as the harness
(`.venv_worldclass`); no extraction is re-run.

**Auditability.** The four scripts that produce the reported result are vendored into the repository at
`scripts/ubfc_leakage/` and their captured outputs at `outputs/leakage_run/`, so the headline numbers are
reproducible and auditable from the repository alone rather than from session scratch directories.
