from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, shell=False)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description="Generate fairness-under-compression + edge metrics table.")
    p.add_argument("--csv", type=str, default=str(PROJECT_ROOT / "data" / "csv" / "multimodal.csv"))
    p.add_argument("--split", type=str, default=str(PROJECT_ROOT / "data" / "csv" / "split_seed42.json"))
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    p.add_argument("--fusion", type=str, default="cgf")
    p.add_argument("--base_ckpt", type=str, required=True, help="Base fair model ckpt (e.g., counterfactual_cgf_js_...best.pt)")
    p.add_argument("--pruned_ckpt", type=str, required=True, help="Pruned ckpt you already generated")
    p.add_argument("--repaired_ckpt", type=str, required=True, help="Repaired ckpt output from fair_repair_finetune.py")
    p.add_argument("--out_csv", type=str, default=str(OUT_DIR / "compression_audit.csv"))
    args = p.parse_args()

    py = sys.executable

    base = Path(args.base_ckpt)
    pruned = Path(args.pruned_ckpt)
    repaired = Path(args.repaired_ckpt)

    rows = []

    def eval_and_bench(tag: str, ckpt: Path, qdyn: bool):
        fair_out = OUT_DIR / f"tmp_fair_{tag}.json"
        edge_out = OUT_DIR / f"tmp_edge_{tag}.json"

        run([py, "src/eval_fairness.py",
             "--ckpt", str(ckpt),
             "--fusion", args.fusion,
             "--backbone", args.backbone,
             "--csv", args.csv,
             "--split", args.split,
             "--out", str(fair_out)] + (["--quantize_dynamic"] if qdyn else []))

        run([py, "src/edge_benchmark.py",
             "--ckpt", str(ckpt),
             "--fusion", args.fusion,
             "--backbone", args.backbone,
             "--csv", args.csv,
             "--out", str(edge_out)] + (["--quantize_dynamic"] if qdyn else []))

        fr = read_json(fair_out)
        er = read_json(edge_out)

        row = {
            "tag": tag,
            "ckpt": str(ckpt),
            "quantize_dynamic": qdyn,

            "acc": fr.get("acc"),
            "f1": fr.get("f1"),
            "balanced_acc": fr.get("balanced_acc"),
            "auc_roc": fr.get("auc_roc"),

            "dp_gap_abs": fr.get("dp_gap_abs"),
            "eo_max_gap": fr.get("eo", {}).get("eo_max_gap"),
            "cf_prob_gap_mean_abs": fr.get("cf_prob_gap_mean_abs"),

            "ece_overall": fr.get("ece", {}).get("overall"),
            "ece_gap_abs": fr.get("ece", {}).get("ece_gap_abs"),

            "ckpt_size_mb": er.get("ckpt_size_mb"),
            "latency_ms_mean": er.get("latency_ms_mean"),
            "latency_ms_p95": er.get("latency_ms_p95"),
            "throughput_fps_est": er.get("throughput_fps_est"),
            "rss_mb_delta": er.get("rss_mb_delta"),
        }
        rows.append(row)

    # fp32
    eval_and_bench("base_fp32", base, qdyn=False)
    eval_and_bench("pruned_fp32", pruned, qdyn=False)
    eval_and_bench("repaired_fp32", repaired, qdyn=False)

    # qdyn
    eval_and_bench("base_qdyn", base, qdyn=True)
    eval_and_bench("pruned_qdyn", pruned, qdyn=True)
    eval_and_bench("repaired_qdyn", repaired, qdyn=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\nSaved table:", out_csv)


if __name__ == "__main__":
    main()
