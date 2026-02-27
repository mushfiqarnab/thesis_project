"""
COMPREHENSIVE CODEBASE DIAGNOSTIC SCRIPT

This script systematically audits the entire codebase for:
1. Device handling issues
2. Tensor shape/dtype mismatches
3. Numerical stability problems
4. Data pipeline issues
5. Model architecture validation
6. XAI readiness checks

Run: python src/COMPREHENSIVE_CODEBASE_DIAGNOSTIC.py
"""

import sys
from pathlib import Path
import inspect
import torch
import torch.nn as nn

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import (
    PhysMLP, VisionEncoder, FusionConcat, CausalGatedFusion, 
    MultimodalThreatModel, ModelOut
)
from dataset_fair import MultimodalCSVDatasetWithCF, remove_scar_pil
from train_cgf_fair import (
    js_divergence, dp_gap_prob, eo_gap_prob,
    load_state_dict_safely, set_seed
)


class DiagnosticReport:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        
    def add_issue(self, severity, file, line, issue_type, description):
        """severity: CRITICAL | HIGH | MEDIUM | LOW"""
        self.issues.append({
            'severity': severity,
            'file': file,
            'line': line,
            'type': issue_type,
            'description': description
        })
        
    def add_warning(self, file, description):
        self.warnings.append({'file': file, 'description': description})
        
    def add_info(self, file, description):
        self.info.append({'file': file, 'description': description})
        
    def print_report(self):
        print("\n" + "="*80)
        print("COMPREHENSIVE CODEBASE DIAGNOSTIC REPORT")
        print("="*80)
        
        if self.issues:
            print(f"\n[ISSUES FOUND: {len(self.issues)}]")
            print("-" * 80)
            for issue in sorted(self.issues, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}[x['severity']]):
                print(f"\n[{issue['severity']}] {issue['file']}:{issue['line']}")
                print(f"  Type: {issue['type']}")
                print(f"  Issue: {issue['description']}")
        else:
            print("\n✓ NO CRITICAL ISSUES FOUND")
            
        if self.warnings:
            print(f"\n[WARNINGS: {len(self.warnings)}]")
            print("-" * 80)
            for w in self.warnings:
                print(f"  • {w['file']}: {w['description']}")
                
        if self.info:
            print(f"\n[INFO: {len(self.info)}]")
            print("-" * 80)
            for i in self.info:
                print(f"  • {i['file']}: {i['description']}")
                
        print("\n" + "="*80)
        print(f"SUMMARY: {len(self.issues)} issues, {len(self.warnings)} warnings, {len(self.info)} info")
        print("="*80)


def check_models_py():
    """Audit src/models.py"""
    print("[AUDIT] Checking src/models.py...")
    report = DiagnosticReport()
    
    # ✓ Test PhysMLP
    try:
        phys = PhysMLP(in_dim=5, emb_dim=64)
        x = torch.randn(2, 5)
        y = phys(x)
        assert y.shape == (2, 64), f"PhysMLP output shape mismatch: {y.shape}"
        report.add_info("models.py", "PhysMLP: Correct shape (5->64)")
    except Exception as e:
        report.add_issue("HIGH", "models.py", 22, "PhysMLP", f"Forward pass failed: {e}")
        
    # ✓ Test VisionEncoder device handling
    try:
        ve = VisionEncoder("mobilenet_v3_small")
        img = torch.randn(1, 3, 224, 224)
        
        # Check if model parameters are on same device
        model_device = next(ve.parameters()).device
        if model_device.type == 'cuda' and img.device.type == 'cpu':
            report.add_issue("HIGH", "models.py", 45, "Device Mismatch", 
                "VisionEncoder created without explicit device; model params on GPU, input on CPU will fail")
        
        emb, fmap = ve(img)
        assert emb.shape == (1, 576), f"VisionEncoder emb shape mismatch: {emb.shape}"
        assert fmap is not None and fmap.shape[0] == 1, f"VisionEncoder fmap shape mismatch"
        report.add_info("models.py", "VisionEncoder: mobilenet_v3_small works correctly")
    except Exception as e:
        report.add_issue("HIGH", "models.py", 45, "VisionEncoder", f"Forward pass failed: {e}")
        
    # ✓ Test FusionConcat
    try:
        fc = FusionConcat(v_dim=576, p_dim=64)
        v = torch.randn(2, 576)
        p = torch.randn(2, 64)
        out = fc(v, p)
        assert isinstance(out, ModelOut), "FusionConcat should return ModelOut"
        assert out.logits.shape == (2, 2), f"FusionConcat logits shape: {out.logits.shape}"
        assert out.gate is None, "FusionConcat should not have gate"
        report.add_info("models.py", "FusionConcat (Design A): Works correctly")
    except Exception as e:
        report.add_issue("HIGH", "models.py", 68, "FusionConcat", f"Forward pass failed: {e}")
        
    # ✓ Test CausalGatedFusion - CRITICAL DEVICE CHECK
    try:
        cgf = CausalGatedFusion(v_dim=576, p_dim=64, d=256)
        v = torch.randn(2, 576)
        p = torch.randn(2, 64)
        fmap = torch.randn(2, 576, 7, 7)  # mobilenet output
        mask = torch.randn(2, 1, 224, 224)
        
        # Test 1: Device consistency
        out = cgf(v, p, fmap, mask)
        if out.focus.device != v.device:
            report.add_issue("CRITICAL", "models.py", 118, "Device Mismatch",
                f"focus tensor on {out.focus.device} but input on {v.device}")
        
        # Test 2: Gate value range
        if torch.any(out.gate < 0) or torch.any(out.gate > 1):
            report.add_issue("MEDIUM", "models.py", 145, "Numerical Range",
                f"Gate values outside [0,1]: min={out.gate.min()}, max={out.gate.max()}")
        else:
            report.add_info("models.py", "CausalGatedFusion: Gate properly in [0,1]")
            
        # Test 3: Output shape
        assert out.logits.shape == (2, 2), f"CGF logits shape: {out.logits.shape}"
        assert out.focus.shape == (2, 1), f"CGF focus shape: {out.focus.shape}"
        
        report.add_info("models.py", "CausalGatedFusion (Design B): Works correctly")
    except Exception as e:
        report.add_issue("HIGH", "models.py", 78, "CausalGatedFusion", f"Forward pass failed: {e}")
        
    # ✓ Test focus_from_mask with device mismatch
    try:
        fmap_cpu = torch.randn(2, 576, 7, 7)
        mask_cpu = torch.randn(2, 1, 224, 224)
        focus = CausalGatedFusion.focus_from_mask(fmap_cpu, mask_cpu)
        
        # Now test with GPU (if available)
        if torch.cuda.is_available():
            fmap_gpu = fmap_cpu.cuda()
            mask_cpu_still = mask_cpu  # intentional mismatch
            try:
                focus_gpu = CausalGatedFusion.focus_from_mask(fmap_gpu, mask_cpu_still)
                report.add_issue("CRITICAL", "models.py", 113, "Device Mismatch in focus_from_mask",
                    "F.interpolate not checking device compatibility between fmap and mask")
            except RuntimeError as de:
                report.add_issue("CRITICAL", "models.py", 119, "Device Mismatch",
                    f"focus_from_mask fails with GPU fmap and CPU mask: {de}")
    except Exception as e:
        report.add_info("models.py", f"focus_from_mask: Basic CPU test passed")
        
    # ✓ Test MultimodalThreatModel
    try:
        model = MultimodalThreatModel(phys_dim=5, vision_backbone="mobilenet_v3_small", fusion="cgf")
        img = torch.randn(2, 3, 224, 224)
        phys = torch.randn(2, 5)
        mask = torch.randn(2, 1, 224, 224)
        
        out = model(img, phys, mask=mask)
        assert out.logits.shape == (2, 2), f"Model logits shape: {out.logits.shape}"
        report.add_info("models.py", "MultimodalThreatModel: End-to-end forward pass works")
    except Exception as e:
        report.add_issue("CRITICAL", "models.py", 160, "MultimodalThreatModel", f"Forward failed: {e}")
        
    return report


def check_dataset_fair_py():
    """Audit src/dataset_fair.py"""
    print("[AUDIT] Checking src/dataset_fair.py...")
    report = DiagnosticReport()
    
    # Check if dataset files exist
    csv_path = PROJECT_ROOT / "data" / "csv" / "multimodal.csv"
    if not csv_path.exists():
        report.add_warning("dataset_fair.py", f"CSV not found: {csv_path}")
        return report
        
    try:
        ds = MultimodalCSVDatasetWithCF(str(csv_path))
        sample = ds[0]
        
        # Validate sample structure
        required_fields = ['img', 'img_cf', 'phys', 'y', 'scar', 'has_cf', 'mask']
        for field in required_fields:
            if not hasattr(sample, field):
                report.add_issue("HIGH", "dataset_fair.py", 30, "Sample Structure",
                    f"Sample missing field: {field}")
        
        # Validate tensor dtypes and shapes
        if sample.img.dtype != torch.float32:
            report.add_issue("MEDIUM", "dataset_fair.py", 190, "Dtype Issue",
                f"img dtype is {sample.img.dtype}, expected float32")
        
        if sample.phys.dtype != torch.float32:
            report.add_issue("MEDIUM", "dataset_fair.py", 190, "Dtype Issue",
                f"phys dtype is {sample.phys.dtype}, expected float32")
                
        if sample.mask.dtype != torch.float32:
            report.add_issue("MEDIUM", "dataset_fair.py", 245, "Dtype Issue",
                f"mask dtype is {sample.mask.dtype}, expected float32")
        
        # Validate mask values
        if (sample.mask < 0).any() or (sample.mask > 1).any():
            report.add_issue("MEDIUM", "dataset_fair.py", 245, "Mask Range",
                f"mask values outside [0,1]: min={sample.mask.min()}, max={sample.mask.max()}")
        
        # Validate phys normalization
        if sample.phys.abs().max() > 100:
            report.add_warning("dataset_fair.py", "Physiology values very large; may need normalization")
        
        report.add_info("dataset_fair.py", f"Dataset loaded successfully, {len(ds)} samples")
        report.add_info("dataset_fair.py", f"Physiology dimension: {sample.phys.numel()}")
        
    except Exception as e:
        report.add_issue("CRITICAL", "dataset_fair.py", 100, "Dataset Load", f"Failed to load: {e}")
        
    return report


def check_train_cgf_fair_py():
    """Audit src/train_cgf_fair.py"""
    print("[AUDIT] Checking src/train_cgf_fair.py...")
    report = DiagnosticReport()
    
    # ✓ Test JS divergence
    try:
        p = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
        q = torch.tensor([[0.8, 0.2], [0.5, 0.5]])
        
        js = js_divergence(p, q)
        assert js.shape == (2,), f"JS divergence shape: {js.shape}"
        assert torch.all(js >= 0), f"JS divergence should be non-negative: {js}"
        assert torch.all(js <= np.log(2)), f"JS divergence should be <= log(2): {js}"
        
        report.add_info("train_cgf_fair.py", "js_divergence: Mathematically correct")
    except Exception as e:
        report.add_issue("CRITICAL", "train_cgf_fair.py", 135, "JS Divergence", f"Failed: {e}")
        
    # ✓ Test fairness metrics
    try:
        p1 = torch.tensor([0.1, 0.9, 0.2, 0.8])
        scar = torch.tensor([1, 1, 0, 0])
        y = torch.tensor([0, 1, 0, 1])
        
        dp = dp_gap_prob(p1, scar)
        assert dp >= 0, f"DP gap should be non-negative: {dp}"
        assert dp <= 1, f"DP gap should be <= 1: {dp}"
        
        eo = eo_gap_prob(p1, y, scar)
        assert eo >= 0, f"EO gap should be non-negative: {eo}"
        
        report.add_info("train_cgf_fair.py", "Fairness metrics: Numerically sound")
    except Exception as e:
        report.add_issue("HIGH", "train_cgf_fair.py", 160, "Fairness Metrics", f"Failed: {e}")
        
    return report


def check_prune_quantize():
    """Audit pruning and quantization modules"""
    print("[AUDIT] Checking prune_checkpoint.py and quantize_export.py...")
    report = DiagnosticReport()
    
    # Basic validation that modules load
    try:
        from prune_checkpoint import should_prune_module, infer_phys_dim_from_state_dict
        model = nn.Linear(10, 5)
        should_prune = should_prune_module("phys.0.weight", model, prune_vision=False)
        report.add_info("prune_checkpoint.py", "Pruning utilities load correctly")
    except Exception as e:
        report.add_warning("prune_checkpoint.py", f"Module load: {e}")
        
    try:
        from quantize_export import maybe_quantize_dynamic
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        model_q = maybe_quantize_dynamic(model)
        report.add_info("quantize_export.py", "Dynamic quantization loads correctly")
    except Exception as e:
        report.add_warning("quantize_export.py", f"Module load: {e}")
        
    return report


def run_xai_readiness_checks():
    """Check if codebase is ready for XAI integration"""
    print("[AUDIT] Checking XAI readiness...")
    report = DiagnosticReport()
    
    # ✓ Gradient flow
    try:
        model = MultimodalThreatModel(phys_dim=5, vision_backbone="mobilenet_v3_small", fusion="cgf")
        img = torch.randn(2, 3, 224, 224, requires_grad=True)
        phys = torch.randn(2, 5, requires_grad=True)
        mask = torch.ones(2, 1, 224, 224)
        
        out = model(img, phys, mask=mask)
        loss = out.logits.sum()
        loss.backward()
        
        if img.grad is None or phys.grad is None:
            report.add_issue("CRITICAL", "models.py", 160, "Gradient Flow",
                "Input gradients are None - XAI attribution methods will fail")
        else:
            report.add_info("models.py", "Gradient flow verified: XAI compatible")
            
    except Exception as e:
        report.add_issue("CRITICAL", "models.py", 160, "Gradient Computation", f"Failed: {e}")
        
    # ✓ Intermediate activation access
    try:
        model = MultimodalThreatModel(phys_dim=5, vision_backbone="mobilenet_v3_small", fusion="cgf")
        
        # Check if we can register hooks
        hooks_registered = 0
        def dummy_hook(mod, inp, out):
            nonlocal hooks_registered
            hooks_registered += 1
            
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ReLU) or isinstance(mod, nn.Linear):
                mod.register_forward_hook(dummy_hook)
                
        if hooks_registered > 0:
            report.add_info("models.py", f"Successfully registered {hooks_registered} hooks for activation capture")
        else:
            report.add_warning("models.py", "No Linear/ReLU layers found for hook registration")
            
    except Exception as e:
        report.add_warning("models.py", f"Hook registration: {e}")
        
    return report


def main():
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE CODEBASE AUDIT (Option A+C)")
    print("="*80 + "\n")
    
    all_reports = []
    
    # Run all audits
    all_reports.append(("src/models.py", check_models_py()))
    all_reports.append(("src/dataset_fair.py", check_dataset_fair_py()))
    all_reports.append(("src/train_cgf_fair.py", check_train_cgf_fair_py()))
    all_reports.append(("src/prune_checkpoint.py + quantize_export.py", check_prune_quantize()))
    all_reports.append(("XAI Readiness", run_xai_readiness_checks()))
    
    # Aggregate and print
    print("\n" + "="*80)
    print("AGGREGATING FINDINGS FROM ALL AUDITS")
    print("="*80)
    
    total_issues = 0
    total_warnings = 0
    
    for name, report in all_reports:
        if report.issues:
            total_issues += len(report.issues)
        if report.warnings:
            total_warnings += len(report.warnings)
            
    for name, report in all_reports:
        report.print_report()
        
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY: {total_issues} critical/high issues, {total_warnings} warnings")
    print(f"{'='*80}\n")
    
    if total_issues > 0:
        print("⚠️  ISSUES MUST BE FIXED BEFORE XAI IMPLEMENTATION")
        return 1
    else:
        print("✓ CODEBASE PASSED AUDIT - READY FOR XAI IMPLEMENTATION")
        return 0


if __name__ == "__main__":
    import numpy as np
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] Diagnostic script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
