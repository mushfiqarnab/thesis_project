# CLI Session Log

1. git status -> frozen state
2. git add -A; git commit -m "Freeze before initial draft"
3. git tag draft-freeze-2026-09-25
4. Copy-Item outputs outputs_snapshot_2026-09-25 -Recurse -Force
5. Get-FileHash data\publishable_scar_production\multimodal_publishable.csv -Algorithm SHA256 -> 2035777F957FA4C5CC830989F4FE40FB9190589F17C567CB29A3EF0E6B2ACAE2
6. python tools\find_checkpoint.py --roots outputs models --out docs\checkpoint_search.csv -> MATCH outputs\quarantine_checkpoints_20260924\equitas_rcmf_master_best.pt
7. Get-ChildItem -Path . -Recurse -Include *.pt,*.pth | ... Export-Csv -Path docs\checkpoint_inventory.csv -> Created inventory.
8. python tools\check_split.py --csv data\publishable_scar_production\multimodal_publishable.csv --subject-col face_id --split-col split -> participant-level
9. .venv\Scripts\python.exe temp_params.py -> 1,140,069 parameters
10. Get-CimInstance Win32_Processor -> AMD Ryzen 5 7500F 6-Core Processor
11. python tools\ortho_precision.py --ckpt outputs\quarantine_checkpoints_20260924\equitas_rcmf_master_best.pt --param stiefel_decomp.weight_raw --k 192 -> int8 fake quant deviation = 0.1579
12. git mv script1.py scripts/_quarantine/script1.py
13. git mv script2.py scripts/_quarantine/script2.py
14. git mv script3.py scripts/_quarantine/script3.py
15. git add scripts/_quarantine/
16. git commit -m "chore: P0.5 Quarantine invalid scripts"
17. Rename-Item "files from claude's correction for draft submission\Final\thesis_latex\thesis_latex\bibliography\references_additions.bib" "references.bib"
18. python fix_lint.py -> replaced "guarantee" with "ensure" and "catastrophic" with "severe"
19. python fix_lint2.py -> replaced remaining "guarantee"
20. python "files from claude's correction for draft submission\files\cli_brief_v3\tools\latex_lint.py" "files from claude's correction for draft submission\Final\thesis_latex\thesis_latex" -> 0 errors, 0 warnings
21. git add "files from claude's correction for draft submission\Final\thesis_latex\thesis_latex"
22. git commit -m "chore: P0.4 - P0.7 resolve TODOs and lint"
23. python scratch\k1_baseline.py -> 0.6931 K1 accuracy
24. python "files from claude's correction for draft submission\files\cli_brief_v3\tools\physio_baseline.py" --csv data\publishable_scar_production\multimodal_publishable.csv -> 0.5665 LR accuracy, 0.5363 MLP accuracy
25. Add-Content -Path "docs\evidence_ledger.csv" -Value ... (physio baselines)
26. python physio_text.py -> Updated chapter 4 and chapter 6 text with baseline metrics
27. python generate_chapter_2.py -> Fully reconstructed chapter_2.tex from chapter_2_revisions_v3.tex to unblock P0.2
28. python "files from claude's correction for draft submission\files\cli_brief_v3\tools\latex_lint.py" ... -> 0 errors, 0 warnings
29. git commit -m "chore: P0.8 inject physio baseline results and rebuild chapter 2"
