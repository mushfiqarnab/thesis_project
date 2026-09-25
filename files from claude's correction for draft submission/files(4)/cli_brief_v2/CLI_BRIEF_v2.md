# CLI Brief v2: Final 48-Hour Plan for the EQUITAS-RCMF Thesis

**This file supersedes `CLI_BRIEF.md` (v1).** Where they differ, follow this one. Read it in full before running anything, then work through it in order.

---

## 0. Situation

- **Deadlines:** initial draft PDF **Sept 26** · slides Sept 29 · defense Oct 3 · final report Oct 7.
- **This 48-hour window:** Block P0 must finish before the draft is submitted. Blocks P1 and P2 follow submission and feed the slides and final report.
- **Final pipeline:** EQUITAS-RCMF on UBFC-Phys: facial video plus wrist BVP and EDA from an Empatica E4, a synthetic brow-line scar, a Stiefel decomposition, a LUPI scar-mask branch, and a focus-driven gate.
- **Thesis files:** `main.tex`, `core/{titlepage,approval,abstract}.tex`, `chapters/chapter_1.tex`, `chapter_2.tex` (existing literature review), `chapter_2_revisions.tex` (replacement passages; **never `\input` it**), `chapter_3.tex` to `chapter_6.tex`, `appendix/appendix_a.tex`, `bibliography/references_additions.bib`.
- **Tools** (`tools/`): `find_checkpoint.py`, `evidence_ledger.py`, `check_split.py`, `fairness_bootstrap.py`, `ortho_precision.py`, `latex_lint.py`. All have been tested except `ortho_precision.py`, whose logic was checked but which must be run once on this machine before its output is used.

## 1. Settled decisions (do not re-ask)

1. **Pipeline:** EQUITAS-RCMF on UBFC-Phys. The CGF model on WESAD + FFHQ is obsolete.
2. **Cohort:** N = 8 video-capable participants (s1–s8). Verify the count (P0.2) and report any mismatch.
3. **WESAD:** a legacy artifact, removed from Appendix A. If the final physiology encoder is pre-trained on WESAD, report it (P0.3); do not re-add it yourself.
4. **Name:** EQUITAS-RCMF only (formerly GWPACDNet; checkpoint prefix `gw_cd_`). There is no acronym expansion in the abstract.
5. **Modality collapse:** EQUITAS-RCMF is less accurate than the camera-off physiology-only model (≈72.15%). This limitation stays.
6. **Three-regime results:** no tables exist. No claim of consistency across regimes, modes, or seeds may appear unless P1 produces the table.
7. **Bangladesh data-protection law:** removed for this draft. Do not re-add it.

## 2. Verified external facts you may use

These come from the dataset's documentation and papers that use it. Cite `\cite{sabour2023}` wherever they appear in the thesis.

- UBFC-Phys has **56 participants** (46 women, 10 men), each recorded in three tasks: rest (T1), speech (T2), and arithmetic (T3), inspired by the Trier Social Stress Test.
- Video: **1024 × 1024 pixels at 35 frames per second** (MJPEG compression).
- **BVP and EDA** were recorded with an **Empatica E4 wristband**. This resolves the Costantini TODO in `chapter_2_revisions.tex`; delete that TODO's first sentence.
- Each participant was randomly assigned one of **two difficulty scenarios, "test" or "ctrl"**.
- Per-participant info files record **gender, scenario, date, and start time**. No source lists age.
- The dataset authors publish **exclusion lists for rPPG evaluation** (as reproduced in the rPPG-Toolbox and in Paruchuri et al.'s *Motion Matters*). Within s1–s8: **s8 is excluded for T1, T2, and T3**; s3 for T1; s1, s4, and s6 for T2; and s5 for T3.

## 3. Hard rules (never break these)

1. **No synthetic, dummy, simulated, or hand-typed data** may produce any number in the thesis, the slides, or a report. Every number comes from running the real model or analysis on the real data, or from an existing output file generated that way.
2. **No reverse-engineering** of a value to match a target.
3. **Every number in a `.tex` file must be traceable** to a file and a field, recorded in `docs/evidence_ledger.csv`.
4. **No training before the draft is submitted.** Inference and analysis only. After submission, no job over 1 hour without asking.
5. **Never delete files.** Use `git mv`. Never overwrite outputs; write new timestamped files.
6. **Never modify** the pre-registered dataset, the pre-registration document, or `src/models/`.
7. **No post-hoc changes** to thresholds, seeds, subsets, splits, or metric definitions after seeing a result. Bad results are reported as they are.
8. **The pre-registered hierarchy is fixed:** POS (I = 0.5), U = 1050.0, p = 0.0314 is primary; CHROM and PBV are exploratory. Any new analysis (such as the s8 check) is labelled **exploratory**.
9. **Unverifiable means "pending"** (§8). Never fill a gap with a plausible value.
10. **Single writer.** Before editing thesis files, create `docs/EDITING.lock` containing your session name and the time; delete it when you finish. If a lock already exists, or if any `.tex` file's modification time changes while you are working, **stop and ask**. (A previous uncoordinated edit silently deleted a sentence from Chapter 6.)
11. **One commit per task,** with a message naming the task ID (for example, `P0.5: resolve Chapter 3 TODOs`).
12. **Log every command** and its output path in `docs/cli_session_log.md`.

## 4. Stop and ask when

- a hash does not match, or two checkpoint candidates match;
- **the age labels have no documented source** (P0.2d);
- the split is window-level (P0.2f);
- the code shows the privileged and autonomous focus signals on **different scales** without rescaling (P0.3a); report it, then apply §6 branch C;
- a metric definition in the code differs from §7;
- any result would weaken a claim in the abstract or Chapter 6;
- a task would exceed 1 hour or requires training;
- an editing-lock conflict occurs (rule 10).

When you stop, give what you found, the evidence (file:line or command output), and two or three options. Then wait.

---

## 5. Work plan

### P0 — before submission (≈ 8 hours, in this order)

**P0.1 Freeze (15 min).** Create the lock (rule 10), then:
```powershell
git status; git add -A; git commit -m "P0.1: freeze before initial draft"; git tag draft-freeze-v2
Copy-Item outputs outputs_snapshot_v2 -Recurse
Get-FileHash data\publishable_scar_production\multimodal_publishable.csv -Algorithm SHA256
```
**Accept when:** the hash equals `2035777f957fa4c5cc830989f4fe40fb9190589f17c567cb29a3ef0e6b2acae2`.

**P0.1b Start the checkpoint search in a separate terminal** (it runs in the background):
```powershell
python tools\find_checkpoint.py --roots C:\ D:\ "$env:USERPROFILE\OneDrive" --out docs\checkpoint_search.csv
```
The target is `4c2dcad470271ad7109ec302a2e4d31eac429cc054ba3d243ce0d39266b3bc5b`. Also list every other `.pt` or `.pth` file (other seeds, Models A–C, `gw_cd_*`) in `docs/checkpoint_inventory.csv`. Ask the user to check the Recycle Bin and any cloud storage.

**P0.2 Integrity checks (1.5 h).** Record each answer with its evidence in `docs/facts.md`.
- **a. N.** Count unique participant IDs in the pre-registered CSV and in `outputs/leakage_run/`. Report both counts. If they differ, stop and ask.
- **b. Demographic table source.** Open `outputs/reports/equitas_rcmf_master_benchmark_report.json` and confirm that every value in Chapter 4's demographic table appears there.
- **c. Gender source.** Confirm that the gender labels come from UBFC-Phys `info_s*.txt` files, and count participants per gender within s1–s8.
- **d. Age source (critical).** Find where the age groups (18–30, 30–45, 45–65) come from: search the code and data for `age` and inspect the loader. UBFC-Phys files are not documented to contain age. Possible outcomes: a documented source exists → record it; ages were **estimated** by a model → record the model; no source → **stop and ask** (then §6 branch F).
- **e. Participants per subgroup.** For each row of the demographic table, count the participants (not windows) it contains. Tables based on one or two participants must say so (§6 branch F).
- **f. Split unit.**
  ```powershell
  python tools\check_split.py --csv data\publishable_scar_production\multimodal_publishable.csv --subject-col <COL> --split-col <COL>
  ```
  (or `--split-json <file>`). Record "participant-level" or "window-level" exactly as printed. Window-level → stop and ask, then §6 branch A.
- **g. rPPG exclusion list.** For the leakage test, list which s1–s8 clips were analysed. Mark each clip that appears on the dataset authors' exclusion list (§2). Do **not** re-run or alter the pre-registered test (see P1.8).
- **h. "Test" and "ctrl" scenarios.** Record each participant's scenario. If all eight share one scenario, or the split separates them, note it.

**P0.3 Extract facts from the code for Chapter 3 (1.5 h).** For every item, quote the relevant code (file:line) in `docs/facts.md`.
- **a. Focus scale (critical).** Find (i) the loss that trains `Focus_auto`, (ii) its target and whether that target is rescaled, and (iii) which focus value feeds the gate during training. The privileged focus is `log(1 + r)` (unbounded); `Focus_auto` is a sigmoid in (0, 1). If they are on different scales without rescaling → stop and ask, then §6 branch C.
- **b. Gate shape:** is \(G\) a scalar or a 192-dimensional vector? (§6 branch D)
- **c. Counterfactual construction:** is the counterfactual the clean pre-scar frame, or a scar-removed version of the scarred frame (blur or inpainting)? (§6 branch B)
- **d. Scar–stress association in training:** the ρ of the training set. (§6 branch E)
- **e. Windows:** length, stride, and which frame represents each window.
- **f. Physiological features** in \(X_P\) (for example, mean IBI, SDNN, RMSSD, EDA statistics).
- **g. Activation energy** in the focus formula (for example, the squared L2 norm over channels) and how the mask is resized.
- **h. Loss definitions:** \(\mathcal{L}_{\text{causal-inv}}\), \(\mathcal{L}_{\text{latent-inv}}\), \(\mathcal{L}_{\text{DP}}\), \(\mathcal{L}_{\text{EO}}\) (distance measure and surrogate form).
- **i. Parameter counts:** instantiate the model class without a checkpoint and print total, trainable, and backbone-only counts.
- **j. Model C's name:** CGF or CGP.
- **k. Camera-off 72.15%:** its source file, split, regime, and whether it is validation or test accuracy.
- **l. Latency setup:** CPU (`Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors`), threads, batch size, and whether preprocessing is included.
- **m. WESAD:** is it used anywhere in the final training path?

**P0.4 Evidence ledger (1 h).**
```powershell
python tools\evidence_ledger.py --tex-dir chapters core appendix --search-dirs outputs docs logs --out docs\evidence_ledger.csv
```
Confirm each row by opening the hit file and checking that it is the same quantity, not just the same digits. For each `NOT_FOUND` row, find the true source or plan a "pending" replacement.

**P0.5 Edit the thesis (2 h), one commit per chapter.**
1. Resolve every `% TODO(authors)` marker with a P0.2 or P0.3 fact, using the branch text in §6 where one applies. Otherwise use the pending wording (§8) and keep the marker.
2. In `chapter_3.tex`, add the verified UBFC-Phys facts (§2): 56 participants, 35 fps at 1024 × 1024, E4 wristband, and the test/ctrl scenarios.
3. Apply `chapter_2_revisions.tex` to `chapter_2.tex` (five passages), then work through its checklist. Apply only the corrections your P0.3 facts confirm; mark the rest with a TODO.
4. Search for leftover old names and replace them:
   ```powershell
   Select-String -Path chapters\*.tex,core\*.tex,appendix\*.tex,docs\*.md -Pattern "GWPACDNet|GWPACD|Equivariant Q-Attention|CGP"
   ```

**P0.6 Quarantine invalid scripts (10 min).** Move the "magic constant" derivation, the dummy-data bootstrap, and the synthetic bfloat16 Newton–Schulz script into `scripts/_quarantine/` with `git mv`. Add a README: "circular, hardcoded, or synthetic inputs; not evidence; must not be cited." Then search the thesis for `3.566283`, `NSMR`, `Newton-Schulz`, `bfloat16`, and `TRL-3`, and remove any claims based on them. The crop rule may appear only as it is already worded in Chapter 3.

**P0.7 Bibliography (20 min).** Append `references_additions.bib` and any missing entries (`geirhos2020`, `pearl2009`, `sabour2023`, `sagawa2020`, `arjovsky2019`, `liu2021jtt`). Then check for duplicate or missing keys:
```powershell
python -c "import re,glob,collections;b=open('bibliography/references.bib',encoding='utf-8').read();k=re.findall(r'@\w+\{([^,]+),',b);print('dup:',[x for x,c in collections.Counter(k).items() if c>1]);c=set();[c.update(x.strip() for m in re.findall(r'\\cite\{([^}]*)\}',open(f,encoding='utf-8').read()) for x in m.split(',')) for f in glob.glob('chapters/*.tex')+glob.glob('core/*.tex')+glob.glob('appendix/*.tex')];print('missing:',sorted(c-set(k)))"
```
**Accept when:** `dup: []` and `missing: []`.

**P0.8 Build and lint (30 min).**
```powershell
python tools\latex_lint.py chapters core appendix
latexmk -pdf -interaction=nonstopmode main.tex
Select-String -Path main.log -Pattern "undefined|multiply defined|Missing character"
```
**Accept when:** the linter reports 0 errors and the log has no undefined citations or references. Review each wording warning: negated uses such as "does not guarantee" are correct and may stay.

**P0.9 Report** using §10. Delete `docs/EDITING.lock`. The user then submits.

### P1 — after submission: real analyses (inference only; each under 1 h)

Save predictions under `outputs/analysis_v2_<date>/` with the columns `id, subject, scar, y, p, p_cf, model, regime, mode, seed`.

If the master checkpoint (or other checkpoints) is available:
1. **Focus-scale test (highest value).** On validation data, compute both the privileged focus and `Focus_auto` for the same inputs. Report their means, ranges, and correlation, and the gate values \(G\) produced by each. This settles §6 branch C empirically.
2. **Gate statistics:** mean and standard deviation of \(G\) for scarred and unscarred inputs.
3. **Vision ablation:** accuracy with a constant (mean) image on the same test set.
4. **Like-for-like physiology baseline:** Model B on exactly the same test windows as Model D.
5. **Regime × mode × seed table** (ρ ∈ {0.85, 0.50, 0.15}; privileged and autonomous), for every seed with a checkpoint.
6. `python tools\ortho_precision.py --ckpt <ckpt> --param <W_raw name> --k 192`; use the output in Chapter 6's quantization paragraph.
7. **Sanity checks:** run `sham_edit_probe.py`, then the real Adebayo et al. cascading randomization test for Integrated Gradients (50 fixed test images; randomize layers from the head downward; report signed and absolute Spearman correlations).
8. **Exploratory s8 sensitivity analysis:** repeat the leakage statistics excluding the clips on the dataset authors' exclusion list. Report it as exploratory next to the unchanged primary result, as was done for the O7 analysis.

With per-window predictions (checkpoint or not):
```powershell
python tools\fairness_bootstrap.py --preds <preds_D.csv> --B 10000 --seed 0
python tools\fairness_bootstrap.py --preds <preds_D.csv> --compare <preds_B.csv> --B 10000 --seed 0
```
Report these as participant-clustered bootstrap intervals and repeat the script's small-N warning. The `--compare` difference is the correct way to state whether Model D differs from Model B (or A).

If neither checkpoints nor per-window predictions exist, keep the pending wording and never reconstruct predictions from aggregate JSON files.

### P2 — slides and consistency (after P1)
- `docs/slide_numbers.md`: only numbers confirmed in the ledger, each with its source.
- Update the thesis with any P1 results, then repeat P0.8 and P0.9.

---

## 6. Decision branches (use this text exactly)

**A. Split unit** (P0.2f)
- *Participant-level:* in Chapter 3, replace the split TODO with "The split is by participant, so no participant appears in more than one split." In Chapter 6, delete "to evaluate with participant-level splits;" and its TODO.
- *Window-level:* in Chapter 3, write "The split is by window, so every participant contributes windows to the training, validation, and test sets." In Chapter 4, add to the limitations list:
  `\item \textbf{Window-level split.} Windows from the same participant appear in the training and test sets, so test accuracy partly reflects familiarity with individual participants and is likely to overestimate performance on new individuals.`

**B. Counterfactual** (P0.3c)
- *Clean pre-scar frame:* Chapter 3 → "Each scarred frame is paired with the original frame before the scar was added, so the two inputs differ only inside the scar region." Delete the Chapter 6 sentence beginning "Using the original, pre-scar frame…".
- *Scar removed from the scarred frame:* Chapter 3 → "The counterfactual is produced by [method] inside the scar mask. Because this alters pixels in the scar region rather than restoring the original frame, the counterfactual is approximate."

**C. Focus scale** (P0.3a, or measured in P1.1)
- *Same scale, or rescaled:* Chapter 3 → describe the loss and the rescaling, and delete the Chapter 3 and Chapter 6 focus-scale TODOs.
- *Different scales without rescaling:* Chapter 3 → "During training, the gate uses the privileged focus, which is unbounded; at inference it uses the autonomous estimate, which lies between 0 and 1." Add to the Chapter 4 limitations:
  `\item \textbf{Focus-scale mismatch.} The privileged focus used in training and the autonomous focus used at inference lie on different scales, so the deployed gate may penalize scar-focused visual features less strongly than the trained gate. The reported autonomous-mode results reflect the deployed behaviour, but the gate's training signal and its deployed input are not matched.`
  In Chapter 6, replace the focus-scale TODO with a short subsection stating the same point as future work.

**D. Gate shape** (P0.3b): state "\(G\) is a scalar that weights all 192 dimensions equally" or "\(G \in (0,1)^{192}\), with the focus penalty applied to every dimension."

**E. Training association** (P0.3d)
- *ρ = 0.85:* keep the regime wording as it is.
- *Balanced (ρ = 0.5):* Regime 1 → "the scar occurs mostly with stress"; Regime 3 → "the scar occurs mostly without stress"; delete "matching the association in the training data" and "reversing that association."
- *Other:* state the value and adjust the wording to match.

**F. Demographics** (P0.2d, e)
- *Ages have a documented source:* cite it in Chapter 3.
- *Ages were estimated by a model:* Chapter 3 → "Age groups were estimated from the facial video with [model], because UBFC-Phys does not record age; they are approximate."
- *No source:* delete the three age rows from Table 4.x and every mention of age groups (Chapter 4, Chapter 6, abstract).
- *In all cases:* add the number of participants in each subgroup to the table caption, for example "Gender: Female (n = 6 participants)".

**G. rPPG exclusion list** (P0.2g): add to Chapter 4, after the leakage table:
`The dataset authors recommend excluding some clips from rPPG evaluation because of video quality; within this cohort, these include all three clips of participant s8 and [list others]. The pre-registered analysis included these clips; an exploratory analysis excluding them will be reported in the final version.`
If P1.8 has already run, replace the second sentence with its result.

**H. WESAD** (P0.3m): if it is used for pre-training, add to Chapter 3: "The physiological encoder was pre-trained on WESAD~\cite{schmidt2018} before training on UBFC-Phys." Otherwise, do nothing.

## 7. Metric definitions (confirm against the code first)

With \(\hat{y} = \mathbb{1}[p \ge 0.5]\): DP gap \(= |P(\hat{y}=1 \mid s=1) - P(\hat{y}=1 \mid s=0)|\); EO gap \(= \max(|\Delta\text{TPR}|, |\Delta\text{FPR}|)\); CF gap \(=\) mean over scar-positive windows of \(|p(x) - p(x_{S\leftarrow0})|\); worst-group accuracy \(=\) the minimum over the four (scar, label) groups. If the code's definitions differ, stop and ask. Never switch definitions silently. Bootstrap intervals for absolute gaps are biased upward near zero, so use `--compare` for model-versus-model statements.

## 8. "Pending" wording

> "[Quantity] will be reported in the final version, following recovery of the master checkpoint."

or

> "[Quantity] has not yet been computed and will be reported in the final version."

Keep the `% TODO(authors)` marker next to it.

## 9. Wording and LaTeX rules

| Do not write | Write instead |
|---|---|
| prove / proof (empirical) | show / evidence / indicates |
| guarantee (empirical) | ensures (exact algebra only) / is designed to |
| silicon-level / hardware guarantee | the compiled graph does not require the mask |
| formally proven compliance | is consistent with / supports |
| zero CF gap | near-zero CF gap (give the value) |
| catastrophic | substantial (give the number) |
| Counterfactual Risk Minimization | counterfactual invariance |
| Adebayo check (for the sham probe) | perturbation-based sanity check |

**LaTeX rules:**
- Put `%` comments **on their own lines, between complete sentences**, never inside a sentence.
- Use `~\cite{}` and `\ref{}`, never typed numbers.
- Math goes in `$...$` or `\(...\)`. The string "USD" is always a corrupted `$`.
- Run `tools\latex_lint.py` before every commit that touches a `.tex` file.

## 10. Report template (after P0.9, and after each P1 item)

```
1. Freeze: tag=<>, dataset hash match=<yes/no>, lock handled=<yes/no>
2. Checkpoint: <MATCH path | NO MATCH>; inventory: <n files, notable ones>
3. Integrity: N=<training>/<leakage>; demographic table traced=<yes/no>;
   gender counts=<F:n, M:n>; AGE SOURCE=<documented | estimated by X | none>;
   participants per subgroup=<...>; split=<participant | window>;
   excluded-list clips in leakage test=<list>; scenarios=<test/ctrl per participant>
4. Code facts: focus scale=<same | rescaled | MISMATCH>; gate=<scalar | vector>;
   counterfactual=<clean frame | blur | other>; training rho=<>; window=<len/stride/frame>;
   X_P features=<>; energy=<>; losses=<summary>; params total/trainable/backbone=<>;
   Model C=<CGF|CGP>; camera-off 72.15%=<val|test, regime>; latency setup=<>; WESAD used=<yes/no>
5. Branches applied (section 6): <A..H with the option chosen>
6. Ledger: <n> numbers; <n> confirmed; <n> pending; <n> removed
7. TODO(authors) remaining: <count + reasons>
8. Quarantined: <paths>; claims removed: <list>
9. Build: success=<>; lint errors=<0>; undefined refs/cites=<0>
10. Stop-and-ask items awaiting the user: <list>
11. (P1) New results, each with its output path
```
