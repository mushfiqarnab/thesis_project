"""Measure how much reduced numerical precision perturbs the orthogonality of the
REAL trained Stiefel projection. Replaces the synthetic bfloat16 script.

Usage:
    python ortho_precision.py --ckpt equitas_rcmf_master_best.pt \
        --param stiefel_decomp.W_raw --k 192
    (add --projected if the stored parameter is already the orthogonal matrix)
"""
import argparse
import sys

import torch


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    sys.exit("Unrecognised checkpoint format; inspect it and adapt load_state explicitly.")


def project(w):
    q, _ = torch.linalg.qr(w.T, mode="reduced")  # thin QR of W^T
    return q.T                                    # rows orthonormal


def fake_int8(w, per_row):
    if per_row:
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    else:
        scale = w.abs().max().clamp_min(1e-12) / 127.0
    return torch.clamp(torch.round(w / scale), -127, 127) * scale


def errors(w, k):
    w = w.double()
    eye = torch.eye(w.shape[0], dtype=torch.float64)
    cross = torch.linalg.norm(w[:k] @ w[k:].T).item()
    full = torch.linalg.norm(w @ w.T - eye).item()
    return cross, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--param", required=True)
    ap.add_argument("--k", type=int, required=True, help="number of causal rows")
    ap.add_argument("--projected", action="store_true")
    args = ap.parse_args()

    sd = load_state(args.ckpt)
    if args.param not in sd:
        cands = [n for n, t in sd.items() if getattr(t, "ndim", 0) == 2]
        sys.exit(f"Parameter {args.param!r} not found. 2-D parameters: {cands}")
    w = sd[args.param].float()
    w_st = w if args.projected else project(w)
    print(f"Matrix shape: {tuple(w_st.shape)}; causal rows k = {args.k}")
    variants = {
        "fp32": w_st,
        "fp16 (cast)": w_st.half().float(),
        "bf16 (cast)": w_st.bfloat16().float(),
        "int8 per-tensor (fake quant)": fake_int8(w_st, per_row=False),
        "int8 per-row (fake quant)": fake_int8(w_st, per_row=True),
    }
    print(f"{'precision':<32}{'||W_c W_conf^T||_F':>22}{'||W W^T - I||_F':>20}")
    for name, v in variants.items():
        c, f = errors(v, args.k)
        print(f"{name:<32}{c:>22.3e}{f:>20.3e}")


if __name__ == "__main__":
    main()
