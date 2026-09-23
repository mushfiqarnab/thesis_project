# Synthetic Scar-Like Artifact Validation Protocol

## Scope

This protocol evaluates a controlled visual intervention for fairness and shortcut-learning experiments. It does not treat generated pixels as clinical evidence and does not claim that procedural artifacts are equivalent to real scars.

## Evidence Basis

The clinical descriptors are adapted from validated scar-assessment literature:

- [Vancouver Scar Scale](https://pubmed.ncbi.nlm.nih.gov/20101227/): vascularity, pigmentation, pliability, and height.
- [Patient and Observer Scar Assessment Scale](https://pubmed.ncbi.nlm.nih.gov/16327618/): observer and patient dimensions including vascularity, pigmentation, thickness, relief, pliability, pain, and itch.
- [Review of scar scales and measuring devices](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2903966/): limitations of single-score scar assessment and the need for complementary measurements.
- [International clinical recommendations on scar management](https://pubmed.ncbi.nlm.nih.gov/12140478/): scar maturation and clinical management context.

Image-quality metrics are treated as complementary measurements:

- [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html) measures learned perceptual distance and is not a clinical realism test.
- [FID](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html) compares feature distributions and is not sufficient for individual-sample acceptance.
- [Improved precision and recall](https://arxiv.org/abs/1904.06991) separates sample fidelity from distribution coverage and requires a relevant reference distribution.

## Artifact Model

For source image $I$, soft edit mask $M$, and renderer $R_\theta$:

$$
I_{scar} = (1-M) \odot I + M \odot R_\theta(I).
$$

The final image must satisfy exact outside-mask preservation:

$$
\operatorname{MAE}_{outside}(I, I_{scar}) = 0.
$$

Each artifact records the full parameter vector $\theta$, including morphology, geometry, color, texture, maturation preset, seed, source hash, mask hash, and generator version.

## Morphology Presets

The renderer uses visual presets rather than medical diagnoses:

1. `linear_mature`: narrow, tapered, low-vascularity mark with residual pigmentation.
2. `linear_immature`: wider transition, increased red tone, stronger local contrast.
3. `irregular_atrophic_like`: irregular boundary and reduced local luminance without a raised-edge claim.
4. `short_clustered`: multiple short connected segments with bounded spacing.

These labels describe rendered appearance only. They must not be reported as confirmed keloid, hypertrophic, or atrophic scars without clinical review.

## Automated Gates

Reject an artifact when any gate fails:

- not exactly one adult source face with landmarks;
- mask intersects an exclusion region around eyes, nostrils, lips, or image boundary;
- mask area or aspect ratio is outside the frozen configuration range;
- output dimensions or color mode differ from the source contract;
- any pixel outside the final mask changes;
- saturated or near-white artifact pixels exceed the frozen limit;
- clean and scarred hashes are identical;
- source, output, or mask hashes are missing;
- source face or physiology subject appears in more than one split.

## Human Visual Study

The realism study must be blinded and randomized. Include clean, synthetic, and licensed real-scar references when a real reference set is available. Record rater expertise, repeated-image checks, rating time, and missing ratings.

Experts rate anatomical placement, boundary naturalness, pigmentation, vascular appearance, texture, maturation consistency, and unintended changes. Non-experts rate visibility and perceptual plausibility separately.

Report weighted agreement statistics for ordinal ratings and confidence intervals. A mean plausibility score without agreement and subgroup analysis is insufficient evidence.

## External Utility Test

The strongest validation is transfer to held-out real-scar images. Train on the synthetic condition and evaluate on real references without changing the threshold or selecting a checkpoint using the real test set. Report task performance, fairness metrics, and calibration separately.

## Reporting Boundary

Until the human and external-utility studies pass, the permitted claim is:

> controlled synthetic scar-like visual confounder with auditable counterfactual pairing.

The terms `clinically realistic scar`, `medical scar simulator`, and named clinical scar diagnoses require additional evidence and are not implied by this pipeline.