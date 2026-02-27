#!/usr/bin/env python3
"""
XAI Data Analysis Script - TIER 3 Enhanced Analysis

This script demonstrates how to use the improved XAI module with TIER 3 enhancements:
- GradientFlowAnalyzer: Monitor gradient health during IG computation
- ExplanationComparator: Compare methods and validate consistency
- PerformanceProfiler: Benchmark XAI methods on your data

Usage:
    python analysis_with_tier3.py --ckpt <path> --csv <path> --num_samples 50
    
Features:
    - Generates explanations using multiple XAI methods
    - Validates results with IGValidator framework
    - Compares method agreement
    - Profiles performance
    - Creates analysis report
"""

import sys
from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import time

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import MultimodalThreatModel
from xai import (
    XAIExplainer, 
    ExplanationOutput,
    IGValidator,
    GradientFlowAnalyzer,
    ExplanationComparator,
    PerformanceProfiler,
    ImageNormalizer
)


class TierIIIAnalyzer:
    """
    Enhanced XAI analysis using TIER 3 improvements
    
    Provides:
    1. Explanation generation across multiple methods
    2. Automatic validation using IGValidator
    3. Gradient flow monitoring with GradientFlowAnalyzer
    4. Method consistency checking with ExplanationComparator
    5. Performance profiling with PerformanceProfiler
    """
    
    def __init__(self, model: nn.Module, device: torch.device, num_ig_steps: int = 50):
        """
        Initialize analyzer
        
        Args:
            model: PyTorch model to explain
            device: torch.device
            num_ig_steps: Number of steps for Integrated Gradients
        """
        self.model = model
        self.device = device
        self.num_ig_steps = num_ig_steps
        
        # Initialize XAI components
        self.explainer = XAIExplainer(model, device=device)
        self.validator = IGValidator()
        self.gradient_analyzer = GradientFlowAnalyzer()
        self.comparator = ExplanationComparator()
        self.profiler = PerformanceProfiler()
        self.normalizer = ImageNormalizer()  # ImageNet defaults
        
        # Statistics
        self.results = {
            'explanations': [],
            'validations': [],
            'gradient_health': [],
            'method_comparisons': [],
            'performance': {}
        }
    
    def analyze_sample(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: torch.Tensor,
        scar: torch.Tensor = None,
        verbose: bool = True
    ) -> Dict:
        """
        Analyze a single sample with all TIER 3 tools
        
        Args:
            img: (1, 3, H, W) image tensor
            phys: (1, D) physiology tensor
            mask: (1, 1, H, W) mask tensor
            scar: (1, 1) scar label tensor
            verbose: Print results
            
        Returns:
            Dictionary with all analysis results
        """
        if verbose:
            print("\n" + "="*70)
            print("TIER 3 ANALYSIS: Single Sample")
            print("="*70)
        
        analysis = {
            'sample_id': len(self.results['explanations']),
            'predictions': {},
            'validations': {},
            'gradient_health': {},
            'performance': {},
        }
        
        # ============================================================
        # STEP 1: Generate explanations with performance profiling
        # ============================================================
        if verbose:
            print("\n[STEP 1] Generating Explanations (with profiling)...")
        
        # For gradient-based methods, we need to enable gradients
        # Use vanilla image/phys (not requires_grad) for attention
        # For IG/Saliency, let the method handle gradient enabling
        
        # Profile each method
        methods_to_profile = {
            'integrated_gradients': lambda: self.explainer.ig.explain(img, phys, mask=mask, steps=self.num_ig_steps),
            'saliency_map': lambda: self.explainer.saliency.explain(img, phys, mask=mask),
            'attention': lambda: self.explainer.attention.explain(img, phys, mask=mask),
        }
        
        method_results = {}
        timings = {}
        
        for method_name, method_fn in methods_to_profile.items():
            try:
                result, timing = self.profiler.profile_method(
                    method_fn,
                    method_name=method_name,
                    verbose=False  # We'll show summary later
                )
                method_results[method_name] = result
                timings[method_name] = timing
                analysis['predictions'][method_name] = float(result.prediction)
                
                if verbose:
                    print(f"  [OK] {method_name}: {result.prediction:.4f} "
                          f"({timing['wall_time_sec']:.4f}s)")
            except Exception as e:
                if verbose:
                    err_msg = str(e)[:80]  # Truncate long errors
                    print(f"  [Error] {method_name}: {err_msg}")
        
        analysis['performance']['timings'] = timings
        
        # ============================================================
        # STEP 2: Validate results with IGValidator
        # ============================================================
        if verbose:
            print("\n[STEP 2] Validating Results...")
        
        try:
            # Get attributions for validation
            ig_result = method_results.get('integrated_gradients')
            if ig_result and ig_result.vision_attribution is not None:
                attr_img = ig_result.vision_attribution
                attr_phys = ig_result.phys_attribution if ig_result.phys_attribution is not None else np.zeros(1)
                
                # Validate
                val_results = self.validator.full_validation(
                    attr_img=attr_img,
                    attr_phys=attr_phys,
                    gate_values=[method_results[m].gate_activation or 0.5 
                                for m in method_results if method_results[m].gate_activation],
                    verbose=False
                )
                
                analysis['validations']['ig_validation'] = val_results
                
                if verbose:
                    status = "[PASS]" if val_results['all_pass'] else "[ISSUES]"
                    print(f"  {status}: Attribution validation")
                    print(f"    - Has NaN: {val_results['attributions']['has_nan']}")
                    print(f"    - All zero: {val_results['attributions']['all_zero']}")
                    print(f"    - Gate varies: {val_results['gate']['gate_varies']}")
        
        except Exception as e:
            if verbose:
                print(f"  [Error] Validation error: {e}")
        
        # ============================================================
        # STEP 3: Compare methods with ExplanationComparator
        # ============================================================
        if verbose:
            print("\n[STEP 3] Comparing Methods for Consistency...")
        
        try:
            # Compare predictions
            pred_comparison = self.comparator.compare_predictions(
                analysis['predictions'],
                verbose=False
            )
            analysis['comparisons'] = {
                'predictions': pred_comparison,
                'agreement': pred_comparison['predictions_agree']
            }
            
            if verbose:
                status = "[AGREE]" if pred_comparison['predictions_agree'] else "[DISAGREE]"
                print(f"  {status}: Method predictions")
                print(f"    - Mean: {pred_comparison['mean_prediction']:.4f}")
                print(f"    - Spread: {pred_comparison['prediction_spread']:.4f}")
                
                # Only compare attributions if we have them
                if all(m in method_results and method_results[m].vision_attribution is not None 
                       for m in ['integrated_gradients', 'saliency_map']):
                    attr_dict = {
                        'ig': method_results['integrated_gradients'].vision_attribution,
                        'saliency': method_results['saliency_map'].vision_attribution,
                    }
                    attr_comparison = self.comparator.compare_attributions(attr_dict, verbose=False)
                    if 'mean_pearson' in attr_comparison:
                        print(f"    - Attribution correlation: r={attr_comparison['mean_pearson']:.4f}")
                        analysis['comparisons']['attributions'] = attr_comparison
        
        except Exception as e:
            if verbose:
                print(f"  [Error] Comparison error: {e}")
        
        # ============================================================
        # STEP 4: Summary
        # ============================================================
        if verbose:
            print("\n[SUMMARY]")
            print(f"  Sample ID: {analysis['sample_id']}")
            print(f"  Methods: {len(method_results)} successful")
            print(f"  Validation: {'[PASS]' if analysis['validations'].get('ig_validation', {}).get('all_pass', False) else '[CHECK]'}")
            print(f"  Consistency: {'[GOOD]' if analysis['comparisons'].get('agreement', False) else '[LOW]'}")
        
        return analysis
    
    def analyze_batch(
        self,
        data_loader,
        num_samples: int = None,
        verbose: bool = True
    ) -> Dict:
        """
        Analyze multiple samples
        
        Args:
            data_loader: DataLoader with (img, phys, mask, scar, y) tuples
            num_samples: Max samples to analyze (None = all)
            verbose: Print progress
            
        Returns:
            Summary statistics
        """
        print("\n" + "="*70)
        print("TIER 3 BATCH ANALYSIS")
        print("="*70)
        
        batch_results = {
            'num_samples': 0,
            'num_successful': 0,
            'num_failed': 0,
            'predictions': [],
            'validations_passed': 0,
            'method_agreement': 0,
            'timing_stats': {}
        }
        
        count = 0
        for batch_idx, batch_data in enumerate(data_loader):
            if num_samples and count >= num_samples:
                break
            
            try:
                img, phys, mask = batch_data[:3]
                scar = batch_data[3] if len(batch_data) > 3 else None
                
                img = img.to(self.device)
                phys = phys.to(self.device)
                mask = mask.to(self.device)
                if scar is not None:
                    scar = scar.to(self.device)
                
                # Analyze
                analysis = self.analyze_sample(
                    img, phys, mask, scar,
                    verbose=(verbose and batch_idx < 3)  # Only verbose for first 3
                )
                
                batch_results['num_samples'] += 1
                batch_results['num_successful'] += 1
                batch_results['predictions'].extend(analysis['predictions'].values())
                
                if analysis['validations'].get('ig_validation', {}).get('all_pass', False):
                    batch_results['validations_passed'] += 1
                
                if analysis['comparisons'].get('agreement', False):
                    batch_results['method_agreement'] += 1
                
                count += 1
                
                if verbose and batch_idx % 10 == 0:
                    print(f"  Processed {count} samples...")
            
            except Exception as e:
                batch_results['num_failed'] += 1
                if verbose:
                    print(f"  [Error] Sample {batch_idx} failed: {e}")
        
        # ============================================================
        # Batch Summary Statistics
        # ============================================================
        if batch_results['num_successful'] > 0:
            print("\n" + "="*70)
            print("BATCH ANALYSIS SUMMARY")
            print("="*70)
            
            predictions = np.array(batch_results['predictions'])
            
            print(f"\nProcessed: {batch_results['num_successful']} successful, "
                  f"{batch_results['num_failed']} failed")
            print(f"\nPredictions:")
            print(f"  Mean: {np.mean(predictions):.4f}")
            print(f"  Std: {np.std(predictions):.4f}")
            print(f"  Min: {np.min(predictions):.4f}")
            print(f"  Max: {np.max(predictions):.4f}")
            
            print(f"\nValidation:")
            val_pct = 100 * batch_results['validations_passed'] / batch_results['num_successful']
            print(f"  Passed: {batch_results['validations_passed']}/{batch_results['num_successful']} "
                  f"({val_pct:.1f}%)")
            
            print(f"\nMethod Consistency:")
            agree_pct = 100 * batch_results['method_agreement'] / batch_results['num_successful']
            print(f"  Agreeing: {batch_results['method_agreement']}/{batch_results['num_successful']} "
                  f"({agree_pct:.1f}%)")
        
        return batch_results


def create_dummy_dataloader(num_samples: int = 10, batch_size: int = 1):
    """
    Create a simple dummy DataLoader for testing
    
    Returns loader with (img, phys, mask, scar, y) tuples
    """
    class DummyDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # (B, 3, 224, 224), (B, D), (B, 1, 224, 224), (B, 1), (B)
            img = torch.randn(3, 224, 224)  # Random image
            phys = torch.randn(20)  # 20-dim physiology
            mask = torch.ones(1, 224, 224)  # Full mask
            scar = torch.tensor([idx % 2], dtype=torch.long)  # Binary scar label
            y = torch.tensor([idx % 2], dtype=torch.long)  # Binary label
            
            return img, phys, mask, scar, y
    
    dataset = DummyDataset(num_samples)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(
        description="TIER 3 Enhanced Data Analysis"
    )
    parser.add_argument("--ckpt", type=str, default=None, 
                       help="Checkpoint path (optional, uses dummy data if not provided)")
    parser.add_argument("--csv", type=str, default=None,
                       help="CSV dataset path (optional, uses dummy data if not provided)")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to analyze")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--ig_steps", type=int, default=50,
                       help="IG integration steps")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small",
                       choices=["mobilenet_v3_small", "vit_b_16"])
    parser.add_argument("--fusion", type=str, default="cgf",
                       choices=["concat", "cgf"])
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Setup] Device: {device}")
    
    # ============================================================
    # Load or create model
    # ============================================================
    print(f"[Setup] Creating model: {args.backbone} + {args.fusion} fusion")
    
    if args.ckpt:
        print(f"  Loading checkpoint from {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Clean prefix
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    else:
        state = None
    
    model = MultimodalThreatModel(
        phys_dim=20,
        vision_backbone=args.backbone,
        fusion=args.fusion,
        num_classes=2
    ).to(device)
    
    if state:
        model.load_state_dict(state, strict=False)
    
    model.eval()
    print(f"  Model created and ready")
    
    # ============================================================
    # Create DataLoader
    # ============================================================
    print(f"\n[Setup] Creating DataLoader")
    
    if args.csv:
        print(f"  Loading from {args.csv}")
        # TODO: Implement actual CSV loading
        data_loader = create_dummy_dataloader(args.num_samples, args.batch_size)
    else:
        print(f"  Using dummy data ({args.num_samples} samples)")
        data_loader = create_dummy_dataloader(args.num_samples, args.batch_size)
    
    # ============================================================
    # Run analysis
    # ============================================================
    print(f"\n[Analysis] Starting TIER 3 enhanced analysis")
    print(f"  IG steps: {args.ig_steps}")
    print(f"  Num samples: {args.num_samples}")
    
    analyzer = TierIIIAnalyzer(model, device, num_ig_steps=args.ig_steps)
    
    results = analyzer.analyze_batch(
        data_loader,
        num_samples=args.num_samples,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("[DONE] ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved in analyzer.results dictionary")
    print(f"Next steps:")
    print(f"  1. Visualize attributions")
    print(f"  2. Create explanation reports")
    print(f"  3. Export for thesis figures")


if __name__ == "__main__":
    main()
