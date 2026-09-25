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
