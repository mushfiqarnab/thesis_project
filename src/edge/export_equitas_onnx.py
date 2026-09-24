"""
export_equitas_onnx.py
================================================================================
EQUITAS-RCMF Hardware Compilation and Edge Deployment Suite
Target: ONNX Opset 14 + INT8 Dynamic Quantization
================================================================================
Compiles the master trained EQUITAS-RCMF model into:
1. Pure autonomous edge ONNX graph (zero masks required at edge).
2. Frozen Stiefel weights (zero manifold projection overhead, microsecond speed).
3. Highly optimized INT8 dynamic quantization graph for sub-2MB binary footprint.
4. Comprehensive hardware profiling (p50, p95, p99 latency benchmarks).
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.equitas_rcmf import EquitasRCMFModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("EdgeCompiler")


class EquitasEdgeWrapper(nn.Module):
    """
    Production Edge Wrapper for EQUITAS-RCMF:
    Takes raw camera image (1, 3, 224, 224) and wearable physiology (1, 2)
    and returns threat probability, dynamic thermodynamic gate, and confounder focus.
    Requires NO masks at inference time.
    """

    def __init__(self, core: EquitasRCMFModel):
        super().__init__()
        self.core = core

    def forward(self, vision_input: torch.Tensor, phys_input: torch.Tensor):
        out = self.core(vision_input, phys_input, mask=None)
        threat_prob = torch.softmax(out.logits, dim=1)[:, 1:2]
        return threat_prob, out.gate, out.focus


def compile_and_benchmark(
    ckpt_path: Path,
    phys_dim: int = 2,
    d_causal: int = 192,
    d_confounder: int = 64,
):
    print("=" * 80)
    print("      EQUITAS-RCMF EDGE HARDWARE COMPILER & BENCHMARK SUITE")
    print("=" * 80)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1. Initialize & Restore Weights
    LOGGER.info("[1/5] Instantiating EQUITAS-RCMF Architecture...")
    model = EquitasRCMFModel(
        phys_dim=phys_dim,
        vision_backbone="mobilenet_v3_small",
        d_causal=d_causal,
        d_confounder=d_confounder,
        num_classes=2,
    )

    LOGGER.info("[2/5] Restoring Golden Master Checkpoint from %s...", ckpt_path.name)
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Precompute and bake Stiefel weights as fixed buffers for zero-overhead edge inference
    LOGGER.info("      Baking Stiefel Orthogonal Subspace weights into static edge buffers...")
    model.freeze_stiefel_weights()
    ortho_dev = model.stiefel_decomp.verify_mutual_orthogonality()
    LOGGER.info("      Verified Stiefel Mutual Orthogonality: %.2e", ortho_dev)

    edge_model = EquitasEdgeWrapper(model)
    edge_model.eval()

    # 2. Compile to ONNX FP32
    out_dir = PROJECT_ROOT / "outputs" / "edge"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_fp32_path = out_dir / "equitas_rcmf_master_fp32.onnx"
    onnx_int8_path = out_dir / "equitas_rcmf_master_int8.onnx"

    LOGGER.info("[3/5] Exporting Static Graph to ONNX Opset 14 (LayerNorm native)...")
    dummy_img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    dummy_phys = torch.randn(1, phys_dim, dtype=torch.float32)

    with torch.no_grad():
        pt_prob, pt_gate, pt_focus = edge_model(dummy_img, dummy_phys)

    torch.onnx.export(
        edge_model,
        (dummy_img, dummy_phys),
        str(onnx_fp32_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["vision_input", "phys_input"],
        output_names=["threat_probability", "thermodynamic_gate", "confounder_focus"],
        dynamic_axes={
            "vision_input": {0: "batch_size"},
            "phys_input": {0: "batch_size"},
            "threat_probability": {0: "batch_size"},
            "thermodynamic_gate": {0: "batch_size"},
            "confounder_focus": {0: "batch_size"},
        },
    )

    fp32_size_mb = onnx_fp32_path.stat().st_size / (1024 * 1024)
    LOGGER.info("      Saved ONNX FP32 Model: %s (%.2f MB)", onnx_fp32_path.name, fp32_size_mb)

    # 3. Dynamic INT8 Quantization
    LOGGER.info("[4/5] Performing Dynamic INT8 Quantization...")
    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QInt8,
    )
    int8_size_mb = onnx_int8_path.stat().st_size / (1024 * 1024)
    compression_ratio = (1.0 - int8_size_mb / fp32_size_mb) * 100
    LOGGER.info(
        "      Saved ONNX INT8 Model: %s (%.2f MB) -> %.1f%% compression!",
        onnx_int8_path.name,
        int8_size_mb,
        compression_ratio,
    )

    # 4. Parity & Latency Profiling
    LOGGER.info("[5/5] Running Hardware Profiling (p50, p95, p99 Latency on CPU)...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    session_fp32 = ort.InferenceSession(str(onnx_fp32_path), sess_options, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(str(onnx_int8_path), sess_options, providers=["CPUExecutionProvider"])

    img_np = dummy_img.numpy()
    phys_np = dummy_phys.numpy()
    ort_inputs = {"vision_input": img_np, "phys_input": phys_np}

    # Verify numerical output parity
    ort_out_fp32 = session_fp32.run(None, ort_inputs)
    ort_out_int8 = session_int8.run(None, ort_inputs)

    diff_fp32 = np.max(np.abs(pt_prob.numpy() - ort_out_fp32[0]))
    diff_int8 = np.max(np.abs(pt_prob.numpy() - ort_out_int8[0]))
    LOGGER.info("      PyTorch vs ONNX FP32 Max Absolute Error: %.2e", diff_fp32)
    LOGGER.info("      PyTorch vs ONNX INT8 Max Absolute Error: %.2e", diff_int8)

    # Warmup
    for _ in range(15):
        _ = session_fp32.run(None, ort_inputs)
        _ = session_int8.run(None, ort_inputs)

    # Benchmark 100 inference passes
    n_iters = 100
    times_fp32, times_int8 = [], []

    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = session_fp32.run(None, ort_inputs)
        times_fp32.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = session_int8.run(None, ort_inputs)
        times_int8.append((time.perf_counter() - t0) * 1000.0)

    p50_fp32, p95_fp32, p99_fp32 = np.percentile(times_fp32, [50, 95, 99])
    p50_int8, p95_int8, p99_int8 = np.percentile(times_int8, [50, 95, 99])

    report = {
        "model": "EQUITAS-RCMF (Autonomous Edge Model)",
        "orthogonality_deviation": ortho_dev,
        "fp32_size_mb": fp32_size_mb,
        "int8_size_mb": int8_size_mb,
        "compression_percent": compression_ratio,
        "parity_max_abs_err_fp32": float(diff_fp32),
        "parity_max_abs_err_int8": float(diff_int8),
        "latency_fp32_ms": {
            "p50": float(p50_fp32),
            "p95": float(p95_fp32),
            "p99": float(p99_fp32),
            "mean": float(np.mean(times_fp32)),
        },
        "latency_int8_ms": {
            "p50": float(p50_int8),
            "p95": float(p95_int8),
            "p99": float(p99_int8),
            "mean": float(np.mean(times_int8)),
        },
    }

    report_path = PROJECT_ROOT / "outputs" / "reports" / "equitas_rcmf_edge_benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("      EQUITAS-RCMF EDGE HARDWARE COMPILATION REPORT")
    print("=" * 80)
    print(f"{'Metric':<30} | {'FP32 ONNX':<18} | {'INT8 Quantized':<18}")
    print("-" * 80)
    print(f"{'Model File Size':<30} | {fp32_size_mb:>14.2f} MB | {int8_size_mb:>14.2f} MB")
    print(f"{'Inference Latency (p50)':<30} | {p50_fp32:>14.2f} ms | {p50_int8:>14.2f} ms")
    print(f"{'Inference Latency (p95)':<30} | {p95_fp32:>14.2f} ms | {p95_int8:>14.2f} ms")
    print(f"{'Inference Latency (p99)':<30} | {p99_fp32:>14.2f} ms | {p99_int8:>14.2f} ms")
    print(f"{'Max Absolute Parity Error':<30} | {diff_fp32:>14.2e}    | {diff_int8:>14.2e}")
    print(f"{'Stiefel Orthogonality Error':<30} | {ortho_dev:>14.2e}    | {ortho_dev:>14.2e}")
    print("=" * 80)
    print(f"Deployment artifacts saved to: {out_dir}")
    print(f"Report saved to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export and Profile EQUITAS-RCMF on ONNX Edge Runtime.")
    parser.add_argument(
        "--ckpt",
        default="outputs/checkpoints/equitas_rcmf_master_best.pt",
        help="Path to trained EQUITAS-RCMF checkpoint",
    )
    parser.add_argument("--phys_dim", type=int, default=2)
    parser.add_argument("--d_causal", type=int, default=192)
    parser.add_argument("--d_confounder", type=int, default=64)
    args = parser.parse_args()

    compile_and_benchmark(
        ckpt_path=PROJECT_ROOT / args.ckpt,
        phys_dim=args.phys_dim,
        d_causal=args.d_causal,
        d_confounder=args.d_confounder,
    )
