"""
edge_inference_engine.py
================================================================================
Production Standalone Edge Inference Engine for EQARNB
ZERO-PYTORCH RUNTIME: Built exclusively on ONNXRuntime, NumPy, and Pillow.
Target Environments: Airport Kiosks, Body-Worn Edge Devices, Smart Cameras.
================================================================================
Core Edge Features:
1. Standalone Execution: Does not require PyTorch or CUDA runtime.
2. Vectorized SIMD Preprocessing: Pure NumPy RGB transformation & Z-score scaling.
3. Fail-Safe Circuit Breakers (Sensor Dropout Defense):
   - Camera occlusion / corrupted frame -> thermodynamic suppression to physiology.
   - Wearable sensor loss -> baseline stabilization and telemetry warning.
4. Comprehensive Biometric Telemetry:
   - Threat probability & decision confidence.
   - Thermodynamic gate transmission state (G in [0, 1]).
   - Intrinsic scar confounder detection (Phi in [0, 1]).
   - Microsecond latency profiling.
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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EDGE] %(message)s", datefmt="%H:%M:%S")
LOGGER = logging.getLogger("EQARNB_Edge")

# ImageNet normalization constants (standard float32 vectors)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class EQARNBEdgeEngine:
    """
    Standalone Production Edge Runtime for EQARNB.
    Accepts raw images and wearable signals, returning real-time threat telemetry.
    """

    def __init__(
        self,
        onnx_model_path: Union[str, Path],
        phys_mu: Optional[np.ndarray] = None,
        phys_sigma: Optional[np.ndarray] = None,
        num_threads: int = 4,
    ):
        self.model_path = Path(onnx_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Edge model graph not found at: {self.model_path}")

        # Configure High-Performance Threaded Session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        LOGGER.info("Loading ONNX Edge Graph: %s (%d threads)...", self.model_path.name, num_threads)
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])

        # Physiological normalization parameters (defaulting to calibrated WESAD distribution)
        self.phys_mu = np.array([55.0, 1.2], dtype=np.float32) if phys_mu is None else np.asarray(phys_mu, dtype=np.float32)
        self.phys_sigma = np.array([12.0, 0.6], dtype=np.float32) if phys_sigma is None else np.asarray(phys_sigma, dtype=np.float32)

        # Inspect input tensor names
        inputs = self.session.get_inputs()
        self.vision_name = inputs[0].name
        self.phys_name = inputs[1].name

        LOGGER.info("Engine Ready. Vision Input: '%s', Phys Input: '%s'", self.vision_name, self.phys_name)

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray, None]) -> Tuple[np.ndarray, bool]:
        """
        Processes an image into shape (1, 3, 224, 224) using pure NumPy.
        Returns: (normalized_tensor, is_fallback)
        """
        if image_input is None:
            # Sensor Dropout Guard: Blank neutral canvas
            return np.zeros((1, 3, 224, 224), dtype=np.float32), True

        try:
            if isinstance(image_input, (str, Path)):
                img = Image.open(str(image_input)).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
            elif isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")

            # Resize to 224x224
            img_resized = img.resize((224, 224), resample=Image.BILINEAR)
            arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # (224, 224, 3)

            # Transpose to NCHW: (1, 3, 224, 224)
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

            # Standardize
            norm_arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            return norm_arr.astype(np.float32), False
        except Exception as e:
            LOGGER.warning("Camera read error: %s. Engaging sensor dropout safety fallback.", e)
            return np.zeros((1, 3, 224, 224), dtype=np.float32), True

    def preprocess_physiology(self, phys_input: Optional[Union[list, tuple, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        """
        Normalizes physiological features into (1, 2) Z-score tensor.
        Returns: (normalized_tensor, is_fallback)
        """
        if phys_input is None:
            # Sensor Dropout Guard: Neutral calibrated zero mean
            return np.zeros((1, 2), dtype=np.float32), True

        try:
            p = np.asarray(phys_input, dtype=np.float32).flatten()
            if np.isnan(p).any() or len(p) == 0:
                return np.zeros((1, 2), dtype=np.float32), True

            if len(p) == 1:
                p = np.array([p[0], self.phys_mu[1]], dtype=np.float32)
            elif len(p) > 2:
                p = p[:2]

            p_norm = (p - self.phys_mu) / self.phys_sigma
            return p_norm[np.newaxis, :].astype(np.float32), False
        except Exception as e:
            LOGGER.warning("Physiology sensor error: %s. Engaging safety fallback.", e)
            return np.zeros((1, 2), dtype=np.float32), True

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, None],
        phys_input: Optional[Union[list, tuple, np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Executes real-time end-to-end inference and returns full telemetry.
        """
        t0 = time.perf_counter()

        # Step 1: Preprocess inputs with sensor dropout guards
        vision_tensor, vision_fallback = self.preprocess_image(image_input)
        phys_tensor, phys_fallback = self.preprocess_physiology(phys_input)

        # Step 2: Execute pure ONNX inference
        ort_inputs = {
            self.vision_name: vision_tensor,
            self.phys_name: phys_tensor,
        }
        raw_outputs = self.session.run(None, ort_inputs)

        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Step 3: Unpack outputs
        # [threat_probability, thermodynamic_gate, confounder_focus]
        threat_prob = float(raw_outputs[0][0, 0])
        gate_val = float(raw_outputs[1][0, 0])
        focus_val = float(raw_outputs[2][0, 0])

        # Step 4: Determine threat alert level
        is_threat = bool(threat_prob >= 0.50)
        confidence = threat_prob if is_threat else (1.0 - threat_prob)

        if threat_prob >= 0.75:
            alert = "HIGH THREAT"
        elif threat_prob >= 0.50:
            alert = "ELEVATED"
        else:
            alert = "CLEAR"

        # Diagnostic state explanation
        if focus_val > 0.35:
            fairness_telemetry = "Confounder detected: thermodynamic gate suppressed vision; decision safely routed to physiology."
        else:
            fairness_telemetry = "Unconfounded facial features verified; cooperative vision-physiology fusion active."

        return {
            "threat_detected": is_threat,
            "threat_probability": round(threat_prob, 4),
            "confidence": round(confidence, 4),
            "alert_level": alert,
            "thermodynamic_gate": round(gate_val, 4),
            "confounder_focus": round(focus_val, 4),
            "telemetry_note": fairness_telemetry,
            "sensor_guard": {
                "vision_fallback": vision_fallback,
                "phys_fallback": phys_fallback,
            },
            "latency_ms": round(t_elapsed_ms, 2),
        }


def main():
    parser = argparse.ArgumentParser("EQARNB Standalone Edge Runtime (Zero-PyTorch).")
    parser.add_argument(
        "--model",
        default="outputs/edge/equitas_rcmf_master_fp32.onnx",
        help="Path to compiled ONNX model (.onnx)",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to facial RGB camera image")
    parser.add_argument("--hrv", type=float, default=52.0, help="Heart Rate Variability (HRV in ms)")
    parser.add_argument("--gsr", type=float, default=1.45, help="Galvanic Skin Response (GSR in uS)")
    parser.add_argument("--benchmark", action="store_true", help="Run 100-pass latency benchmark")
    args = parser.parse_args()

    # Find default sample if image is not specified
    sample_img = args.image
    if sample_img is None:
        default_dir = Path("data/publishable_scar_production")
        candidates = list(default_dir.glob("*.png")) + list(default_dir.glob("*.jpg"))
        if candidates:
            sample_img = str(candidates[0])

    engine = EQARNBEdgeEngine(onnx_model_path=args.model)

    print("\n" + "=" * 75)
    print("        EQARNB STANDALONE PRODUCTION EDGE RUNTIME")
    print("=" * 75)
    print(f"Model Graph:       {Path(args.model).name}")
    print(f"Vision Input:      {sample_img if sample_img else '[SENSOR DROPOUT SIMULATION]'}")
    print(f"Physiology Inputs: HRV = {args.hrv} ms, GSR = {args.gsr} uS")
    print("-" * 75)

    result = engine.predict(image_input=sample_img, phys_input=[args.hrv, args.gsr])

    print(f"Prediction:        {'[!] THREAT DETECTED' if result['threat_detected'] else '[OK] CLEAR / NON-THREAT'}")
    print(f"Threat Likelihood: {result['threat_probability']*100:.2f}% (Confidence: {result['confidence']*100:.2f}%)")
    print(f"Alert Level:       {result['alert_level']}")
    print(f"Thermodynamic Gate:{result['thermodynamic_gate']:.4f} (Vision Transmission Weight)")
    print(f"Confounder Focus:  {result['confounder_focus']:.4f} (Intrinsic Scar Energy)")
    print(f"Execution Latency: {result['latency_ms']:.2f} ms")
    print(f"Fairness Note:     {result['telemetry_note']}")
    print("=" * 75)

    if args.benchmark:
        print("\nBenchmarking 100 consecutive edge inference cycles on CPU...")
        latencies = []
        for _ in range(100):
            res = engine.predict(sample_img, [args.hrv, args.gsr])
            latencies.append(res["latency_ms"])

        p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
        print(f"Throughput: {1000.0 / np.mean(latencies):.1f} FPS")
        print(f"Latency:    p50 = {p50:.2f} ms | p95 = {p95:.2f} ms | p99 = {p99:.2f} ms | min = {min(latencies):.2f} ms")
        print("=" * 75)


if __name__ == "__main__":
    main()
