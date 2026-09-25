# CLI Brief v3: Final Plan (supersedes v1 and v2)

Read this file in full before acting. It replaces `CLI_BRIEF.md` and `CLI_BRIEF_v2.md`. Rules from v2 §3 (hard rules), §7 (metric definitions), §8 ("pending" wording), and §9 (wording and LaTeX rules) **still apply** and are summarized in §2 below.

## 0. What changed and why

Your last report established that the project has **two lines of work**, and the thesis files have been rewritten to match:

| | **Line A** | **Line B (EQUITAS-RCMF)** |
|---|---|---|
| Question | Does facial video contain physiological information? | Can a model ignore a spurious visual attribute? |
| Data | UBFC-Phys video, s1–s4, 12 clips, E4 wrist BVP as reference | WESAD chest ECG/EDA (15 participants) + 418 FFHQ faces with synthetic scars, paired at random |
| Result | POS p = 0.0314 (pre-registered); CHROM, PBV exploratory | accuracy 0.617 in all three regimes (autonomous); CF gap 0.00064 |

Two consequences drive this plan:
1. **In Line B, the faces carry no stress information.** A model that relies only on physiology is behaving *correctly*. The "modality collapse" claim is withdrawn, because its evidence compared validation accuracy on a different dataset.
2. **The decisive missing number** is a physiology-only baseline on the *same* split. It is the accuracy ceiling for this benchmark.

## 1. New files (in this package)

- `thesis_latex_v3/`: the rewritten abstract and Chapters 1, 3, 4, and 6, plus the updated Appendix A. Chapter 5, the title page, the approval page, `main.tex`, and the bibliography additions carry over from v2.
- `thesis_latex_v3/chapters/chapter_2_revisions_v3.tex`: replacement passages for `chapter_2.tex`. It supersedes every earlier revisions file, which must **not** be used. Never `\input` it.
- `tools/physio_baseline.py`: a new, pre-declared physiology-only baseline.
- `tools/extract_metrics.py`: prints every metric block in a benchmark JSON with its full key path and raw value. Use it instead of copying numbers by hand.
- The other tools are unchanged except `latex_lint.py`, which now skips revision files.

## 2. Rules (summary; the v2 wording governs)

1. No synthetic, dummy, or hand-typed data behind any reported number. No reverse-engineering.
2. Every number in a `.tex` file must be traceable (file plus field) and recorded in `docs/evidence_ledger.csv`. A string match is **not** confirmation: open the file and check that it is the same quantity.
3. **No training before submission, with one gated exception (P0.8)** that runs only if the user replies "yes" to it explicitly.
4. Never delete files or overwrite outputs; use `git mv` and timestamped new files. Never modify `src/models/`, the dataset, or the pre-registration.
5. No post-hoc changes to thresholds, splits, seeds, or metric definitions after seeing a result.
6. The pre-registered hierarchy is fixed: POS is primary; everything else in Line A is exploratory.
7. Single writer: hold `docs/EDITING.lock` while editing `.tex` files. Make one commit per task, and log every command in `docs/cli_session_log.md`.
8. Put `%` comments on their own lines, between complete sentences, and never inside a sentence. A comment must not contain `\cite`, `\ref`, or `\label` (the linter flags this).
9. **Do not reintroduce** any of these: "modality collapse" as an established result; the 72.15% camera-off figure as a comparison with EQUITAS-RCMF; "eight participants"; UBFC-Phys as the source of Line B data; rPPG or pulse language about FFHQ still images; the Bangladesh data-protection law; the phrase "WESAD is a legacy pre-training artifact".

## 3. Stop and ask when

- a verification in P0.3 **disagrees** with the v3 text;
- a TODO's answer would change a number in the abstract, Chapter 4's tables, or Chapter 6;
- Model A's reports were trained on a CSV other than `multimodal_publishable.csv` and no alternative Model A results exist (report it; do not substitute);
- the lock is held by someone else, or a `.tex` file changes while you are working.

---

## 4. P0: before submission (≈ 5 hours, in order)

**P0.1 Install v3 (15 min).** Take the lock, commit the current state, then copy the files from `thesis_latex_v3/` into the thesis LaTeX folder, replacing `core/abstract.tex`, `chapters/chapter_1.tex`, `chapter_3.tex`, `chapter_4.tex`, `chapter_6.tex`, and `appendix/appendix_a.tex`. Copy `chapter_2_revisions_v3.tex` alongside them. Keep your existing `chapter_2.tex`. Commit as `P0.1: install v3 thesis files`.

**P0.2 Apply the Chapter 2 revisions (45 min).** Replace the eleven marked passages in `chapter_2.tex` with the text in `chapter_2_revisions_v3.tex`: the relevance paragraphs for Schmidt, Costantini, Toneva, Vapnik, Anthis, Arjovsky, Cohen, JTT, and FairGRAPE, the 2.3 dataset paragraph, and the final sentence of the 2.3 IRM paragraph. Then apply its sentence-level fixes. If an earlier edit changed "HRV and GSR" to "BVP and EDA" anywhere, change it back: HRV/GSR is correct for Line B. If a passage cannot be located exactly, stop and report it rather than guessing where it goes.

**P0.3 Verify the v3 text against the sources (1 h).** First run `python tools\extract_metrics.py outputs\reports\equitas_rcmf_master_benchmark_report.json outputs\reports\thesis_production_benchmark_report.json > docs\metrics_extract.txt` and use its output for items 1, 2, and 11. For each item, confirm it matches the source, or stop and ask. Record each one in the ledger with `human_confirmed = yes`.

| # | Claim in v3 | Source to open |
|---|---|---|
| 1 | Regimes table: autonomous and privileged rows for acc, DP, EO, CF (Ch4, Table `tab:regimes`) | `outputs/reports/equitas_rcmf_master_benchmark_report.json` |
| 2 | Face-attribute table values, and face counts 44 / 18 / 28 / 21 / 13 | same JSON; `multimodal_publishable.csv` (test split, grouped by `face_id`) |
| 3 | Split: train S2, S3, S4, S7, S8, S9, S13, S14, S16, S17; val S6, S15; test S5, S10, S11; 2,344 / 504 / 496 rows; faces disjoint | `tools\check_split.py` outputs |
| 4 | Training ρ = 0.85 | `rho_target` column (train split) |
| 5 | 418 faces; FFHQ; InsightFace screen "exactly one adult face with keypoints"; random pairing within split and stress class | `src/build_publishable_scar_dataset.py` |
| 6 | WESAD chest ECG and EDA, 700 Hz; 30 s windows, 15 s stride; RMSSD and mean EDA as the two features in `X_P` | `src/prepare_wesad.py`; confirm the CSV's `hrv` = RMSSD and `gsr` = mean EDA |
| 7 | Gate scalar; energy = \|A\|; nearest-neighbour mask resizing; `Focus_auto` trained by BCE to predict the scar; gate uses the privileged focus in training; focus clamped to [0, 10] | `src/models/equitas_rcmf.py`, `src/train_equitas_rcmf.py` |
| 8 | `L_causal-inv` and `L_latent-inv` = MSE between factual and counterfactual representations; `L_conf` = BCE | `src/train_equitas_rcmf.py` |
| 9 | The CF gap equals the mean absolute difference in predicted stress probability (Ch3 definition) | the metric code used for the JSON |
| 10 | 1,140,069 parameters, all trainable | parameter count from P0.3i of your v2 report |
| 11 | Model C: 50.20% accuracy, CF gap 0.0019 | `outputs/reports/thesis_production_benchmark_report.json` |
| 12 | Leakage statistics, the s2 T1 r = 0.849 and SNR 53.5, the O7 figures | `outputs/leakage_run/*` |
| 13 | Latency 0.90 / 1.30 ms, 1112.7 fps; CPU AMD Ryzen 5 7500F | edge benchmark JSON; your v2 report |

If item 9 disagrees (for example, the code uses JS divergence), change the Chapter 3 definition to match the code. Do not change the numbers.

**P0.4 Resolve the remaining 13 TODOs (1.5 h).** For each marker, apply the verified answer and delete the marker, or replace it with the "pending" wording and keep the marker.

| TODO location | What to find |
|---|---|
| Ch3, orthogonality | Whether `stiefel_orthogonality` in the JSON was computed on the trained checkpoint (read the code that writes it). If it was measured at initialization, say so. |
| Ch3, `L_DP`/`L_EO` | Their exact surrogate forms. |
| Ch3, WESAD conditions | Which conditions map to stress and non-stress (for example, stress vs. baseline only, or amusement included), and whether `X_P` is standardized with training statistics. |
| Ch3, FFHQ counts | The number screened and rejected (the builder prints both), and the scar morphologies used (the `morphology` column). |
| Ch3, regime construction | How the ρ = 0.85 / 0.50 / 0.15 test regimes are built from the 496 samples. |
| Ch3, models' CSVs and seeds | The `csv_path` and seed of each model's report (A, B, C, D). |
| Ch4, Model A | If `thesis_production_benchmark_report.json` has "baseline" (ERM) results per regime **and** that model was trained on `multimodal_publishable.csv`, add a table mirroring `tab:regimes`. Otherwise keep the pending sentence. |
| Ch4, Model C | The regime and mode of the 50.20% figure; reconcile its DP and EO values (the report shows 0.0040 and 0.0346) before adding them. |
| Ch4, attribution figure | Which images `XAI_Causal_Blindness_Proof.png` uses. If they are FFHQ, check that the text contains no pulse-related claim. |
| Ch4, latency | Batch size, thread count, and whether preprocessing is included. |
| Ch4, seeds | Whether other EQUITAS-RCMF seeds have evaluation results on this benchmark. |
| Ch5, ViT-B/16 | Either add `dosovitskiy2021` to the bibliography and cite it, or delete the sentence comparing ViT-B/16 figures. |
| approval.tex | Stop and ask the user for the semester and acceptance month. |

**P0.5 Quarantine the invalid scripts (10 min).** `git mv` the "magic constant" script, the dummy-data bootstrap, and the synthetic bfloat16 Newton–Schulz script into `scripts/_quarantine/`, and add a README stating they are not evidence. Search the thesis for `3.566283`, `NSMR`, `Newton-Schulz`, `bfloat16`, and `TRL-3`, and remove any claims based on them.

**P0.6 Bibliography (15 min).** Merge `references_additions.bib` and any missing entries: `geirhos2020`, `pearl2009`, `sabour2023`, `sagawa2020`, `arjovsky2019`, `liu2021jtt`, `vapnik2015`, `zhu2024vim`, `wang2024repvit`. Run the duplicate and missing-key check from v2 P0.7. **Accept when:** `dup: []` and `missing: []`.

**P0.7 Build and lint (30 min).**
```powershell
python tools\latex_lint.py chapters core appendix
latexmk -pdf -interaction=nonstopmode main.tex
Select-String -Path main.log -Pattern "undefined|multiply defined|Missing character"
```
**Accept when:** 0 lint errors, and no undefined citations or references. Negated wording ("does not guarantee") is correct as written.

**P0.7b Pre-registration check (20 min, before P0.8).** Open `docs/experiment_preregistration.md` and quote, with line numbers, into `docs/facts.md`:
1. The definitions of **K1a, K1b, K2a, K2b** (whichever exist) and **EQARNB**: their features, models, data, and split.
2. Which analyses the **power-planning fallback** ("estimation-only path") applies to, and what "N_vision = 4" and "evaluation N = 15" refer to.
3. Whether any K1/K2 **results already exist** (check `scratch/k1_baseline.py` and its outputs).

Then:
- If the fallback and the withdrawn equivalence tests concern the **model-versus-baseline comparison** (Line B) rather than the leakage test, edit the Line A paragraph in Chapter 3: remove "and the planned equivalence tests are withdrawn" from it, and add to the Line B protocol section: "The pre-registered equivalence comparison between the model and the physiology-only baselines was withdrawn under the pre-registered power fallback; the comparison is reported as an estimate only." Stop and ask if the pre-registration is ambiguous.
- If K1 is pre-registered, **P0.8 must run K1 exactly as pre-registered** (its features and model), on the `multimodal_publishable.csv` participant split. `tools/physio_baseline.py` then becomes a secondary, exploratory check that matches EQUITAS-RCMF's two inputs; label it that way in the text.

**P0.8 (gated) Physiology-only baseline (≈ 10 min, run only if the user replies "yes").** This is the only training allowed before submission. It fits a logistic regression and a one-layer MLP on two features in seconds, under a protocol declared in advance.
```powershell
python tools\physio_baseline.py --csv data\publishable_scar_production\multimodal_publishable.csv --out outputs\analysis_v3\physio_baseline.json
```
Run it **once**. Do not change features, models, or settings afterwards. Report test accuracy for both models, per-participant accuracy, and the majority-class accuracy. If the user approves, replace the Chapter 4 "Model B (camera-off)" paragraph with:
> "A physiology-only model trained and evaluated on this benchmark's split reached a test accuracy of [X] with logistic regression and [Y] with a one-layer MLP (majority-class accuracy [Z]). Because the face images carry no stress information, this is the reference against which EQUITAS-RCMF's accuracy of 0.617 should be judged."

and replace the second paragraph of Chapter 6's "Invariance and Accuracy" subsection with one sentence saying whether 0.617 is at, above, or below that reference. Report the numbers; do not interpret them beyond that sentence.

**P0.9 Report** using §6, then release the lock. The user submits the draft.

## 5. P1: after submission (each item inference-only and under 1 h, unless noted)

Using the recovered checkpoint `outputs\quarantine_checkpoints_20260924\equitas_rcmf_master_best.pt`:
1. **Focus-scale test:** on validation data, compute the privileged focus and `Focus_auto` for the same inputs. Report their means, ranges, and correlation, and the gate values each produces.
2. **Gate statistics:** mean and standard deviation of \(G\) for scarred and unscarred inputs, per regime. Values near 0 would confirm that the model relies on physiology, as intended on this benchmark.
3. **Vision ablation:** accuracy with a constant image on the same 496 test samples.
4. **Naive ERM regimes** (if not already in the JSON): evaluate the ERM checkpoint trained on this benchmark in the three regimes. This is the Colored-MNIST-style comparison: ERM should degrade in the inverted regime.
5. `python tools\ortho_precision.py --ckpt <checkpoint> --param <W_raw name> --k 192` → Chapter 6's quantization paragraph.
6. **Sanity checks:** `sham_edit_probe.py`, then the Adebayo et al. cascading randomization test for Integrated Gradients (50 fixed test images; signed and absolute Spearman correlations).
7. **Line A exploratory check:** repeat the leakage statistics excluding s1 T2, s3 T1, and s4 T2. Report it next to the unchanged primary result.
8. **Other seeds**, if checkpoints exist: repeat the regime table.

**Do not run `fairness_bootstrap.py` clustered by participant on Line B**: with three test participants the interval is meaningless. Report per-participant results instead.

## 6. Report template

```
1. Install: v3 files copied=<yes/no>; commit=<hash>; lock=<held/released>
2. Chapter 2: passages replaced=<n/11>; not located=<list>; checklist items done=<list>
3. Verification (P0.3): items 1-13 each MATCH / MISMATCH (+ evidence for mismatches)
4. TODOs: resolved=<list>; pending=<list with reason>; remaining count=<n>
5. Quarantine: <paths>; claims removed=<list>
6. Bibliography: dup=[]; missing=[]
7. Build: success=<>; lint errors=0; undefined refs/cites=0
8. P0.7b pre-registration: K1 defined=<yes/no, features>; fallback applies to=<Line A | Line B | both>
   P0.8 physiology baseline: <not run (no approval) | K1 as pre-registered=<>, exploratory 2-feature=<> | LR test=<>, MLP test=<>, majority=<>, per-participant=<>>
9. Stop-and-ask items: <list>
10. (P1) results, each with its output path
```
