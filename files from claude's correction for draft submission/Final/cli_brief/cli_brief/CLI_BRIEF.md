# CLI Brief: 48-Hour Plan for the EQUITAS-RCMF Thesis

Read this whole file before running anything. Follow it in order. When this brief conflicts with any earlier instruction, audit document, or previous session's notes, **this brief wins**.

## 0. Context

- **Project:** EQUITAS-RCMF, a multimodal stress-detection model (MobileNetV3-Small vision + BVP/EDA physiology) with a Stiefel orthogonal decomposition, a LUPI scar-mask branch, and a focus-driven gate. Evaluated on a pre-registered UBFC-Phys pilot cohort with a synthetic brow-line scar.
- **Thesis files:** the corrected LaTeX set (`main.tex`, `core/abstract.tex`, `chapters/chapter_1.tex`, `chapter_3.tex` to `chapter_6.tex`, `appendix/appendix_a.tex`, `bibliography/references_additions.bib`). They contain `% TODO(authors)` markers that you will resolve. The existing `chapters/chapter_2.tex` is the literature review.
- **Helper scripts** (in `tools/`, provided with this brief): `find_checkpoint.py`, `evidence_ledger.py`, `check_split.py`, `fairness_bootstrap.py`, `ortho_precision.py`.
- **Deadlines:** initial draft PDF **Sept 26**; slides Sept 29; defense Oct 3; final report Oct 7.
- **Environment:** Windows with PowerShell; training GPU RTX 4060. Adapt paths if the repository layout differs, and report every adaptation.

## 0.1 Decisions already made by the authors (do not re-ask)

- **Final pipeline:** EQUITAS-RCMF on UBFC-Phys (facial video + wrist BVP/EDA). The CGF model on WESAD + FFHQ is an obsolete earlier milestone.
- **Cohort:** N = 8 video-capable participants (s1–s8). Still verify this count in A3.1 and report any mismatch; do not change the thesis text if it matches.
- **WESAD:** a legacy artifact from physiological pre-training. It has been removed from Appendix A. If the final physiology encoder *is* pre-trained on WESAD, report that in A3.2 so the authors can restore and describe it.
- **Name:** use **EQUITAS-RCMF** exclusively. The earlier name was GWPACDNet (checkpoint prefix `gw_cd_`). The acronym expansion has been removed from the abstract; the title page and approval page now use the full EQUITAS-RCMF title.
- **Modality collapse:** EQUITAS-RCMF is less accurate than the camera-off physiology-only model (72.15%). This is stated in Chapter 4's limitations and must stay.
- **Three-regime results:** there are no tables yet, so the draft makes no claim about consistency across regimes, modes, or seeds. Chapter 4 says these results "will be reported in the final version." Do not add such claims unless B1 produces the table.
- **Bangladesh data-protection law:** removed from Chapter 5 for this draft. Do not add it back or add a bibliography entry for it.

## 1. Hard rules (never break these)

1. **No synthetic, dummy, simulated, or hand-typed data may produce any number that appears in the thesis, the slides, or a report to the user.** Every reported number must come from running the real model or the real analysis on the real dataset, or from an existing output file generated that way.
2. **Never reverse-engineer a value to match a target** (for example, deriving a constant from the result it is supposed to explain).
3. **Every number you place in a `.tex` file must be traceable** to a file path and a field, line, or cell. Record it in `docs/evidence_ledger.csv`.
4. **No training runs before the draft is submitted.** Inference and analysis only. After submission, no single job may exceed 1 hour without asking the user.
5. **Never delete files.** Move files if needed (`git mv`), and never overwrite an existing output; write new outputs to new, timestamped paths.
6. **Never modify** the pre-registered dataset, the pre-registration document, or the model definition in `src/models/`. If a change seems necessary, stop and ask.
7. **Never change an analysis after seeing its result** to make it look better: no changing thresholds, seeds, subsets, test splits, or metric definitions post hoc. If a result is bad, report it as it is.
8. **Pre-registration hierarchy is fixed:** POS (I = 0.5), U = 1050.0, p = 0.0314 is the primary test. CHROM and PBV are exploratory corroboration. Never present CHROM as the headline.
9. **If something cannot be verified, write "pending"** in the thesis (see §6); never fill a gap with a plausible number.
10. **Log every command** you run, with its output location, in `docs/cli_session_log.md`.

## 2. Stop and ask the user when

- the master checkpoint's SHA-256 does not match, or two candidates match;
- the dataset hash does not match the pre-registered hash;
- a verified fact contradicts the thesis (for example, the split is window-level, or N differs);
- a metric definition in the repository differs from §5;
- any result would require removing or weakening a claim in the abstract or Chapter 6;
- a naming decision is needed (acronym expansion, model names);
- any task would take more than 1 hour or requires training.

When you stop, state what you found, the evidence (file and line), and two or three options. Then wait.

## 3. Work plan

### Block A — before the draft is submitted (target: 8 hours)

**A1. Freeze and snapshot (20 min)**
```powershell
git status
git add -A; git commit -m "Freeze before initial draft"   # skip large data; respect .gitignore
git tag draft-freeze-2026-09-25
Copy-Item outputs outputs_snapshot_2026-09-25 -Recurse
Get-FileHash data\publishable_scar_production\multimodal_publishable.csv -Algorithm SHA256
```
**Accept when:** the tag exists, the snapshot exists, and the hash equals `2035777f957fa4c5cc830989f4fe40fb9190589f17c567cb29a3ef0e6b2acae2` (if not, stop).

**A2. Search for the master checkpoint (30–60 min, can run in the background)**
```powershell
python tools\find_checkpoint.py --roots C:\ D:\ "$env:USERPROFILE\OneDrive" --out docs\checkpoint_search.csv
git log --all --stat -- "*.pt" "*.pth"
git lfs ls-files 2>$null
```
Also list every other checkpoint on disk (other seeds, Model A/B/C) with its path, size, and hash, in `docs/checkpoint_inventory.csv`. Ask the user to check the Recycle Bin and any cloud storage (Google Drive, Colab) manually.
**Accept when:** `docs/checkpoint_search.csv` exists and you have reported MATCH or NO MATCH. The full target hash is `4c2dcad470271ad7109ec302a2e4d31eac429cc054ba3d243ce0d39266b3bc5b`.

**A3. Establish the facts the thesis depends on (1.5 hours)**
For each item, record the answer and its evidence (file:line or command output) in `docs/facts.md`.

1. **Number of participants.** Count unique participant IDs in the pre-registered CSV, and separately in the leakage-test outputs (`outputs/leakage_run/`). The authors state N = 8; an earlier session reported N = 4. Report both counts and which analysis each applies to. If the leakage test used fewer participants than training, stop and ask.
2. **Source of the physiological input (confirmation only).** The authors state it is UBFC-Phys wrist BVP/EDA. Confirm this in the data loaders, and report whether WESAD is used anywhere in the final training path (for example, to pre-train the physiology encoder):
   ```powershell
   Select-String -Path src\*.py,src\*\*.py -Pattern "WESAD|wesad|UBFC|E4|BVP|EDA" | Select-Object Path,LineNumber,Line
   ```
   Also record which BVP/EDA features the model uses (for example mean IBI, SDNN, RMSSD, EDA statistics); Chapter 2's Costantini et al. entry needs this.
3. **Split unit.**
   ```powershell
   python tools\check_split.py --csv data\publishable_scar_production\multimodal_publishable.csv --subject-col <SUBJECT_COLUMN> --split-col <SPLIT_COLUMN>
   ```
   (or `--split-json <file>`). Report "participant-level" or "window-level" exactly as the script prints it. A window-level result is a **stop-and-ask** item.
4. **Parameter counts.** Instantiate the model class from `src/models/equitas_rcmf.py` (no checkpoint needed) and print total and trainable parameters, and those of the MobileNetV3-Small backbone alone.
5. **Camera-off 72.15%.** Find the file that contains it and record the split, the regime, and whether it is validation or test accuracy.
6. **Model C's name.** Find whether the code calls it CGF (Causal Gated Fusion) or CGP.
7. **Latency setup.** Record the CPU model (`Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors`), PyTorch thread count, batch size, and whether preprocessing is included, from `src/benchmark_edge_latency.py` and its JSON report.

**A4. Build the evidence ledger (1 hour)**
```powershell
python tools\evidence_ledger.py --tex-dir chapters core appendix --search-dirs outputs docs logs --out docs\evidence_ledger.csv
```
Then go through every row by hand. Set `human_confirmed` to `yes` only after opening the hit file and confirming that it is the **same quantity** (not just the same digits). For every `NOT_FOUND` row, either locate the true source or mark the number for replacement with "pending".
**Accept when:** every row has `human_confirmed` = `yes` or a replacement plan.

**A5. Resolve every `% TODO(authors)` marker (1.5 hours)**
```powershell
Select-String -Path chapters\*.tex,core\*.tex,appendix\*.tex -Pattern "TODO\(authors\)" | Select-Object Path,LineNumber
```
For each marker, do one of the following:
- replace the flagged text with a verified fact from A3 or A4, then delete the marker; or
- replace it with the "pending" wording in §6 and keep the marker.

Specific rules:
- **N:** update every mention (abstract, Chapters 1, 3, 4, 6, and Appendix A) consistently with A3.1.
- **Split:** if participant-level, delete the split TODO and state "split by participant." If window-level, stop and ask (the text must then say so, and Chapter 4's limitations must mention it).
- **Name:** already resolved (EQUITAS-RCMF only). Search all `.tex` files and `docs/` for `GWPACDNet`, `GWPACD`, and `Equivariant Q-Attention`; replace remaining uses of the old name with EQUITAS-RCMF, except where a file name or checkpoint path must be quoted exactly.
- **Per-regime / per-seed claims:** already removed. Keep the pending sentence in Chapter 4; do not reintroduce such claims without the B1 table.

**A6. Quarantine the invalid scripts (10 min)**
The "magic constant" derivation, the dummy-data bootstrap, and the synthetic bfloat16 Newton–Schulz script must not be cited.
```powershell
New-Item -ItemType Directory -Force scripts\_quarantine
git mv <path_to_each_script> scripts\_quarantine\
```
Add `scripts/_quarantine/README.md`: "These scripts use circular, hardcoded, or synthetic inputs. Their outputs are not evidence and must not be cited." Then search the thesis for their outputs (the strings `3.566283`, `NSMR`, `Newton-Schulz`, `bfloat16`, `TRL-3`) and remove any claims based on them. The crop-width constant may be described only as: *"The crop width was set to 3.566 × 1.5 × the inter-ocular distance, a value chosen to reproduce the crop geometry used in the pre-registered extraction."*

**A7. Bibliography (30 min)**
1. Append `references_additions.bib` to `bibliography/references.bib`, together with the earlier entries for `geirhos2020`, `pearl2009`, `sabour2023`, `sagawa2020`, `arjovsky2019`, and `liu2021jtt` if missing.
2. Do **not** add a Bangladesh data-protection entry (removed for this draft).
3. Check for duplicate keys and for cited keys missing from the `.bib` file:
   ```powershell
   python -c "import re,glob;b=open('bibliography/references.bib',encoding='utf-8').read();k=re.findall(r'@\w+\{([^,]+),',b);import collections;print('duplicates:',[x for x,c in collections.Counter(k).items() if c>1]);c=set();[c.update(x.strip() for m in re.findall(r'\\cite\{([^}]*)\}',open(f,encoding='utf-8').read()) for x in m.split(',')) for f in glob.glob('chapters/*.tex')+glob.glob('core/*.tex')+glob.glob('appendix/*.tex')];print('missing:',sorted(c-set(k)))"
   ```
**Accept when:** no duplicates and no missing keys.

**A8. Apply the Chapter 2 revisions (45 min)**
Open `chapters/chapter_2_revisions.tex`. Replace the five marked passages in `chapter_2.tex` (the Relevance paragraphs for Schmidt et al., Costantini et al., Toneva et al., and Vapnik & Vashist, and the 2.3 dataset paragraph). Then work through its CHECKLIST: for each item, apply the stated correction if the A3 facts confirm it, otherwise leave a `% TODO(authors)` comment. Do not add `chapter_2_revisions.tex` itself to `main.tex`.

**A9. Compile and lint (45 min)**
```powershell
latexmk -pdf -interaction=nonstopmode main.tex
Select-String -Path main.log -Pattern "undefined|multiply defined|Missing character"
Select-String -Path chapters\*.tex,core\*.tex,appendix\*.tex -Pattern "USD"
```
Fix every undefined citation or reference and every "USD" artifact (it is a corrupted `$`). Apply the wording rules in §7 with a final search.
**Accept when:** the PDF builds, there are no undefined citations or references, and no banned wording remains.

**A10. Report to the user (15 min)** using the template in §8. The user then submits the draft.

### Block B — after submission (hours 8–40): real analyses

Run only what the available artefacts allow. Every analysis is inference-only. Save predictions in the schema expected by `tools/fairness_bootstrap.py` (`id, subject, scar, y, p, p_cf`, plus `model, regime, mode, seed`) under `outputs/analysis_2026-09-2X/`.

**B1. If the master checkpoint was recovered (or other checkpoints exist):**
1. **Regime × mode table:** accuracy, DP gap, EO gap, CF gap, and worst-group accuracy for ρ ∈ {0.85, 0.50, 0.15} in privileged and autonomous modes, for every seed with a checkpoint. This fills the most important pending table.
2. **Gate statistics:** mean and standard deviation of the gate \(G\) for scar-positive and scar-negative inputs in each regime. This shows how much the model actually uses the visual branch.
3. **Vision ablation:** accuracy when the image is replaced with a constant (mean) image, using the same test set. Report it even if accuracy does not change; that is an important finding.
4. **Physiology-only comparison:** Model B's accuracy on exactly the same test windows and regime as Model D, so the 72.15% comparison is like-for-like.
5. **Orthogonality under reduced precision:**
   ```powershell
   python tools\ortho_precision.py --ckpt <checkpoint> --param <name of W_raw> --k 192
   ```
   Use the result in Chapter 6's quantization paragraph (replace "has not been measured" with the measured values).
6. **Sanity checks:** run `src/evaluation/sham_edit_probe.py`. Then implement the actual Adebayo et al. (2018) cascading randomization test for Integrated Gradients: randomize the model's layers from the classifier head downward, one block at a time, recompute IG on 50 fixed test images at each step, and report the Spearman rank correlation with the original maps, **both signed and absolute**. Keep this under 1 hour of compute.

**B2. With any saved per-window predictions (checkpoint or not):**
```powershell
python tools\fairness_bootstrap.py --preds <preds_D.csv> --B 10000 --seed 0
python tools\fairness_bootstrap.py --preds <preds_D.csv> --compare <preds_A.csv> --B 10000 --seed 0
```
Report the printed intervals as **participant-clustered bootstrap intervals** and quote the script's warning about the small number of participants. The comparison output (Model D minus Model A) is the correct way to state whether one model's fairness gap is smaller than another's.

**B3. If neither checkpoints nor per-window predictions exist,** keep the pending wording and tell the user. Do not reconstruct predictions from aggregate JSON files.

### Block C — hours 40–48: consistency and final build

**C1. Verify the Chapter 2 alignment (A8 did the edits).** Search `chapter_2.tex` for sentences that describe *this thesis's own* pipeline:
```powershell
Select-String -Path chapters\chapter_2.tex -Pattern "WESAD|FFHQ|CelebA|80/20|random split|RMSSD|CGF|chest|mask at inference|future work|this thesis"
```
Write `docs/ch2_alignment_report.md` listing each sentence, what it currently says, and what A3 established. Apply only factual corrections. In particular:
- the thesis **now evaluates** biased (ρ = 0.85), neutral, and inverted (ρ = 0.15) test regimes, so sentences in the Anthis & Veitch, Arjovsky et al., and Cohen et al. entries and in 2.3 that call this "future work" must be updated;
- the Vapnik & Vashist entry now fits exactly (the mask is used only in training); remove any conditional caveat about the mask at inference;
- update the WESAD and Costantini et al. entries to match the physiology source from A3.2 (wrist BVP/EDA makes the Costantini wrist-reliability findings directly relevant).
Leave any change of interpretation to the user as a TODO.

**C2. Numbers for the slides.** Create `docs/slide_numbers.md` containing only numbers with `human_confirmed = yes` in the ledger, each with its source.

**C3. Final build and report**, repeating A9 and A10.

## 4. Allowed and forbidden actions

| Allowed | Forbidden |
|---|---|
| Reading any file; running existing evaluation scripts | Training or fine-tuning before the draft; any job over 1 h without asking |
| Writing new analysis scripts under `tools/` or `scripts/analysis/` | Editing `src/models/`, the dataset, or the pre-registration |
| Writing new outputs to new timestamped folders | Overwriting or deleting outputs, logs, or checkpoints |
| Editing `.tex` and `.bib` files per this brief | Adding any number not in the evidence ledger |
| `git mv`, `git commit`, `git tag` | `git reset --hard`, `git push --force`, `git clean` |

## 5. Metric definitions (confirm against the repository before use)

With \(\hat{y} = \mathbb{1}[p \ge 0.5]\), scar \(s\), and label \(y\):
- **DP gap** \(= |P(\hat{y}=1 \mid s=1) - P(\hat{y}=1 \mid s=0)|\)
- **EO gap** \(= \max\bigl(|\mathrm{TPR}_{s=1} - \mathrm{TPR}_{s=0}|,\ |\mathrm{FPR}_{s=1} - \mathrm{FPR}_{s=0}|\bigr)\)
- **CF gap** \(=\) mean over scar-positive windows of \(|p(x) - p(x_{S\leftarrow 0})|\)
- **Worst-group accuracy** \(=\) the lowest accuracy among the four (scar, label) groups

Print the repository's own metric functions first. If any definition differs (for example, soft probabilities instead of hard predictions for DP), **stop and ask**, and never silently switch definitions.

Note: bootstrap intervals for absolute gaps are biased upward near zero, so the point estimate can sit near the lower end of the interval. For "is model X fairer than model Y", use the paired `--compare` output.

## 6. "Pending" wording (use exactly this pattern)

> "[Quantity] will be reported in the final version, following recovery of the master checkpoint."

or, when the issue is not the checkpoint:

> "[Quantity] has not yet been computed and will be reported in the final version."

Keep the corresponding `% TODO(authors)` marker next to it.

## 7. Wording rules for the thesis text

Search for and replace these before every build:

| Do not write | Write instead |
|---|---|
| prove / proof / proves (for empirical results) | show / evidence / indicates |
| guarantee / mathematically guaranteed | ensures (only for exact algebraic facts) / is designed to |
| silicon-level / hardware guarantee | the compiled graph does not require the mask |
| formally proven compliance | is consistent with / supports |
| zero CF gap | near-zero CF gap (report the value) |
| catastrophic / catastrophically | substantial / significant (with the number) |
| definitively / absolute ground truth | (omit) |
| Counterfactual Risk Minimization (for fairness) | counterfactual invariance |
| Adebayo sanity check (for the sham-edit probe) | perturbation-based sanity check |

Exact statements that **are** correct and may stay: the rows of \(W_{\text{St}}\) are orthonormal, and \(W_{\text{causal}}W_{\text{conf}}^\top = 0\) up to numerical precision.

## 8. Report template (send after A10 and after C3)

```
1. Freeze: tag=<name>, dataset hash match=<yes/no>
2. Checkpoint: <MATCH path | NO MATCH>; other checkpoints found: <list>
3. Facts (docs/facts.md): N=<value> (training) / <value> (leakage test); physiology source=<...>;
   split=<participant-level | window-level>; params total=<...> trainable=<...>;
   camera-off 72.15% is <val|test>, regime <...>; Model C name=<...>; latency setup=<CPU, threads, batch, preprocessing yes/no>
4. Evidence ledger: <n> numbers; <n> confirmed; <n> replaced with pending; <n> removed
5. TODO(authors) remaining: <count>, each with reason
6. Quarantined scripts: <paths>; claims removed: <list>
7. Build: success=<yes/no>; undefined refs/cites=<0>; banned words remaining=<0>
8. Stop-and-ask items awaiting the user: <list>
9. (Block B) New results, each with its output file path
```
