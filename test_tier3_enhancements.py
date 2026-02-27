#!/usr/bin/env python3
"""
TIER 3 Test Suite: Advanced Enhancements
Tests for GradientFlowAnalyzer, ExplanationComparator, and PerformanceProfiler
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Callable
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.xai import (
    GradientFlowAnalyzer,
    ExplanationComparator,
    PerformanceProfiler
)


print("=" * 70)
print("TIER 3 ENHANCEMENT TEST SUITE")
print("=" * 70)

# ============================================================================
# TEST 1: GradientFlowAnalyzer - Single Layer Analysis
# ============================================================================
print("\n[TEST 1] Testing GradientFlowAnalyzer - Single Layer Analysis...")

try:
    # Create test gradients (healthy)
    healthy_grads = torch.randn(100, 64, 7, 7) * 0.01  # Small but non-zero
    analyzer = GradientFlowAnalyzer()
    results = analyzer.analyze_gradients(healthy_grads, layer_name="conv5", verbose=False)
    
    # Check results structure
    assert 'layer' in results
    assert 'shape' in results
    assert 'mean' in results
    assert 'std' in results
    assert 'is_healthy' in results
    assert results['layer'] == "conv5"
    assert results['shape'] == (100, 64, 7, 7)
    
    print("✅ Healthy gradient analysis works")
    
    # Test with NaN detection
    nan_grads = torch.randn(100, 64, 7, 7)
    nan_grads[0, 0, 0, 0] = float('nan')
    results_nan = analyzer.analyze_gradients(nan_grads, layer_name="nan_test", verbose=False)
    assert results_nan['has_nan'] == True, "Should detect NaN"
    print("✅ NaN detection works")
    
    # Test with vanishing gradients
    vanishing_grads = torch.ones(100, 64, 7, 7) * 1e-8
    results_vanish = analyzer.analyze_gradients(vanishing_grads, layer_name="vanishing", verbose=False)
    assert results_vanish['vanishing_pct'] > 90, "Should detect vanishing gradients"
    print("✅ Vanishing gradient detection works")
    
    # Test with exploding gradients
    exploding_grads = torch.ones(100, 64, 7, 7) * 1e4
    results_explode = analyzer.analyze_gradients(exploding_grads, layer_name="exploding", verbose=False)
    assert results_explode['exploding_pct'] > 90, "Should detect exploding gradients"
    print("✅ Exploding gradient detection works")
    
    print("✅ TEST 1 PASSED: GradientFlowAnalyzer single layer")
    
except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}")
    raise


# ============================================================================
# TEST 2: GradientFlowAnalyzer - Multiple Layers
# ============================================================================
print("\n[TEST 2] Testing GradientFlowAnalyzer - Multiple Layers...")

try:
    # Create multiple layer gradients
    layer_grads = {
        'conv1': torch.randn(32, 3, 3, 3) * 0.01,
        'conv2': torch.randn(64, 32, 3, 3) * 0.01,
        'fc1': torch.randn(128, 256) * 0.01,
    }
    
    results_multi = GradientFlowAnalyzer.analyze_multiple_layers(
        layer_grads, 
        verbose=False
    )
    
    # Check results
    assert 'conv1' in results_multi
    assert 'conv2' in results_multi
    assert 'fc1' in results_multi
    assert '_overall_healthy' in results_multi
    
    print("✅ Multiple layer analysis works")
    print(f"  Overall health: {results_multi['_overall_healthy']}")
    
    # Test with one bad layer
    bad_layer_grads = {
        'good1': torch.randn(32, 3, 3, 3) * 0.01,
        'bad': torch.ones(64, 32, 3, 3) * 1e-9,  # Vanishing
        'good2': torch.randn(128, 256) * 0.01,
    }
    
    results_bad = GradientFlowAnalyzer.analyze_multiple_layers(
        bad_layer_grads,
        verbose=False
    )
    
    assert results_bad['bad']['vanishing_pct'] > 90, "Should detect bad layer"
    print("✅ Bad layer detection works")
    
    print("✅ TEST 2 PASSED: GradientFlowAnalyzer multiple layers")
    
except Exception as e:
    print(f"❌ TEST 2 FAILED: {e}")
    raise


# ============================================================================
# TEST 3: ExplanationComparator - Attribution Comparison
# ============================================================================
print("\n[TEST 3] Testing ExplanationComparator - Attribution Comparison...")

try:
    # Create similar attributions from different methods
    base_attr = np.random.rand(224, 224)
    
    attr_dict = {
        'integrated_gradients': base_attr + np.random.randn(224, 224) * 0.05,
        'saliency_map': base_attr + np.random.randn(224, 224) * 0.05,
        'attention': base_attr + np.random.randn(224, 224) * 0.05,
    }
    
    # Compare
    comparison = ExplanationComparator.compare_attributions(
        attr_dict,
        verbose=False
    )
    
    # Check results
    assert 'method_pairs' in comparison
    assert 'mean_pearson' in comparison
    assert len(comparison['method_pairs']) == 3  # C(3,2) = 3 pairs
    
    # Should have high correlation since they're similar
    assert comparison['mean_pearson'] > 0.5, "Similar attributions should correlate"
    
    print("✅ Attribution comparison works")
    print(f"  Mean correlation: {comparison['mean_pearson']:.4f}")
    
    # Test with dissimilar attributions
    dissimilar_dict = {
        'method1': np.random.rand(224, 224),
        'method2': np.random.rand(224, 224),  # Completely independent
    }
    
    comparison_dissim = ExplanationComparator.compare_attributions(
        dissimilar_dict,
        verbose=False
    )
    
    # Should have lower correlation
    assert comparison_dissim['mean_pearson'] < 0.5, "Dissimilar attributions should have low correlation"
    
    print("✅ Dissimilarity detection works")
    
    print("✅ TEST 3 PASSED: ExplanationComparator attribution comparison")
    
except Exception as e:
    print(f"❌ TEST 3 FAILED: {e}")
    raise


# ============================================================================
# TEST 4: ExplanationComparator - Prediction Comparison
# ============================================================================
print("\n[TEST 4] Testing ExplanationComparator - Prediction Comparison...")

try:
    # Test with similar predictions
    similar_preds = {
        'method_a': 0.72,
        'method_b': 0.73,
        'method_c': 0.71,
    }
    
    pred_comparison = ExplanationComparator.compare_predictions(
        similar_preds,
        verbose=False
    )
    
    # Check results
    assert 'mean_prediction' in pred_comparison
    assert 'std_prediction' in pred_comparison
    assert 'prediction_spread' in pred_comparison
    assert 'predictions_agree' in pred_comparison
    
    # Should agree
    assert pred_comparison['predictions_agree'] == True, "Similar predictions should agree"
    
    print("✅ Prediction agreement detection works")
    print(f"  Mean: {pred_comparison['mean_prediction']:.4f}")
    print(f"  Spread: {pred_comparison['prediction_spread']:.4f}")
    
    # Test with dissimilar predictions
    dissimilar_preds = {
        'method_a': 0.2,
        'method_b': 0.8,
    }
    
    pred_comparison_dissim = ExplanationComparator.compare_predictions(
        dissimilar_preds,
        verbose=False
    )
    
    assert pred_comparison_dissim['predictions_agree'] == False, "Dissimilar should not agree"
    print("✅ Prediction disagreement detection works")
    
    print("✅ TEST 4 PASSED: ExplanationComparator prediction comparison")
    
except Exception as e:
    print(f"❌ TEST 4 FAILED: {e}")
    raise


# ============================================================================
# TEST 5: PerformanceProfiler - Single Method Profiling
# ============================================================================
print("\n[TEST 5] Testing PerformanceProfiler - Single Method Profiling...")

try:
    # Create a simple test function
    def dummy_method(img, phys):
        """Dummy XAI method for testing"""
        # Simulate some computation
        result = (img.sum() + phys.sum())
        return float(result.item())
    
    img_test = torch.randn(1, 3, 224, 224)
    phys_test = torch.randn(1, 20)
    
    result, timing = PerformanceProfiler.profile_method(
        dummy_method,
        img_test,
        phys_test,
        method_name="dummy_method",
        verbose=False
    )
    
    # Check results
    assert 'method' in timing
    assert 'wall_time_sec' in timing
    assert 'gpu_memory_mb' in timing
    assert 'samples_per_sec' in timing
    
    assert timing['method'] == "dummy_method"
    assert timing['wall_time_sec'] > 0
    assert timing['samples_per_sec'] > 0
    
    print("✅ Single method profiling works")
    print(f"  Wall time: {timing['wall_time_sec']:.6f}s")
    print(f"  Throughput: {timing['samples_per_sec']:.2f} samples/sec")
    
    print("✅ TEST 5 PASSED: PerformanceProfiler single method")
    
except Exception as e:
    print(f"❌ TEST 5 FAILED: {e}")
    raise


# ============================================================================
# TEST 6: PerformanceProfiler - Multiple Methods
# ============================================================================
print("\n[TEST 6] Testing PerformanceProfiler - Multiple Methods...")

try:
    def slow_method(img, phys):
        for _ in range(100):
            img = img + 1
        return float(img.sum().item())
    
    def fast_method(img, phys):
        return float((img.sum() + phys.sum()).item())
    
    methods = {
        'fast': fast_method,
        'slow': slow_method,
    }
    
    img_test = torch.randn(1, 3, 224, 224)
    phys_test = torch.randn(1, 20)
    
    results_multi = PerformanceProfiler.profile_multiple_methods(
        methods,
        img_test,
        phys_test,
        verbose=False
    )
    
    # Check results
    assert 'fast' in results_multi
    assert 'slow' in results_multi
    assert 'wall_time_sec' in results_multi['fast']
    assert 'wall_time_sec' in results_multi['slow']
    
    # Slow should be slower
    assert results_multi['slow']['wall_time_sec'] > results_multi['fast']['wall_time_sec']
    
    print("✅ Multiple method profiling works")
    print(f"  Fast: {results_multi['fast']['wall_time_sec']:.6f}s")
    print(f"  Slow: {results_multi['slow']['wall_time_sec']:.6f}s")
    
    print("✅ TEST 6 PASSED: PerformanceProfiler multiple methods")
    
except Exception as e:
    print(f"❌ TEST 6 FAILED: {e}")
    raise


# ============================================================================
# TEST 7: PerformanceProfiler - Complexity Estimation
# ============================================================================
print("\n[TEST 7] Testing PerformanceProfiler - Complexity Estimation...")

try:
    complexity = PerformanceProfiler.estimate_complexity(
        batch_size=32,
        image_size=(224, 224),
        phys_dim=20,
        num_steps=50,
        verbose=False
    )
    
    # Check results
    assert 'fwd_passes_ig' in complexity
    assert 'bwd_passes_ig' in complexity
    assert 'fwd_passes_saliency' in complexity
    assert 'ig_complexity_ratio' in complexity
    
    # IG should require more passes
    assert complexity['fwd_passes_ig'] > complexity['fwd_passes_saliency']
    assert complexity['ig_complexity_ratio'] == 50.0  # num_steps
    
    print("✅ Complexity estimation works")
    print(f"  IG is {complexity['ig_complexity_ratio']:.1f}x more expensive than Saliency")
    print(f"  IG forward passes: {complexity['fwd_passes_ig']}")
    print(f"  Saliency forward passes: {complexity['fwd_passes_saliency']}")
    
    print("✅ TEST 7 PASSED: PerformanceProfiler complexity estimation")
    
except Exception as e:
    print(f"❌ TEST 7 FAILED: {e}")
    raise


# ============================================================================
# TEST 8: Import and Class Existence
# ============================================================================
print("\n[TEST 8] Verifying all TIER 3 classes are importable...")

try:
    # All should be importable
    assert GradientFlowAnalyzer is not None
    assert ExplanationComparator is not None
    assert PerformanceProfiler is not None
    
    # Create instances
    analyzer = GradientFlowAnalyzer()
    comparator = ExplanationComparator()
    profiler = PerformanceProfiler()
    
    assert analyzer is not None
    assert comparator is not None
    assert profiler is not None
    
    print("✅ All TIER 3 classes are importable and instantiable")
    print("✅ TEST 8 PASSED: Class imports")
    
except Exception as e:
    print(f"❌ TEST 8 FAILED: {e}")
    raise


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ ALL TIER 3 TESTS PASSED")
print("=" * 70)
print("\nTEST SUMMARY:")
print("  [1] GradientFlowAnalyzer - Single Layer         ✅ PASS")
print("  [2] GradientFlowAnalyzer - Multiple Layers      ✅ PASS")
print("  [3] ExplanationComparator - Attributions        ✅ PASS")
print("  [4] ExplanationComparator - Predictions         ✅ PASS")
print("  [5] PerformanceProfiler - Single Method         ✅ PASS")
print("  [6] PerformanceProfiler - Multiple Methods      ✅ PASS")
print("  [7] PerformanceProfiler - Complexity Estimate   ✅ PASS")
print("  [8] Class Imports & Instantiation               ✅ PASS")
print("\nTotal: 8/8 test categories passed")
print("=" * 70)
