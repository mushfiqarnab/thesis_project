"""
TIER 2 Test Suite: Comprehensive tests for all improvements
Tests new classes: ImageNormalizer, IGValidator
Tests updated functionality: L2 norm saliency
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("TIER 2 COMPREHENSIVE TEST SUITE")
print("=" * 80)

# TEST 1: Import new classes
print("\n[TEST 1] Importing new classes from xai module...")
try:
    from xai import ImageNormalizer, IGValidator, IntegratedGradients
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# TEST 2: ImageNormalizer functionality
print("\n[TEST 2] Testing ImageNormalizer class...")
try:
    # Test default ImageNet normalization
    normalizer = ImageNormalizer()
    print("  ✓ Default ImageNet normalizer created")
    
    # Test with custom values
    custom_normalizer = ImageNormalizer(
        mean=[0.5, 0.5, 0.5],
        std=[0.2, 0.2, 0.2],
        dataset_name="custom"
    )
    print("  ✓ Custom normalizer created")
    
    # Test normalization
    img = np.array([[[1.0, 0.5, 0.2]]])  # (1, 1, 3)
    normalized = normalizer.normalize(img)
    denormalized = normalizer.denormalize(normalized)
    
    assert np.allclose(img, denormalized, atol=1e-6), "Denormalization failed"
    print("  ✓ Normalize/denormalize round-trip successful")
    
    # Test torch transform
    torch_denorm = normalizer.get_torch_denorm_transform()
    print("  ✓ PyTorch denormalization transform created")
    
    # Test with tensor
    tensor = torch.randn(2, 3, 224, 224)
    denorm_tensor = torch_denorm(tensor)
    assert denorm_tensor.shape == tensor.shape, "Shape mismatch"
    print("  ✓ PyTorch transform works correctly")
    
    # Test repr
    repr_str = str(normalizer)
    assert "ImageNormalizer" in repr_str, "Repr format incorrect"
    print("  ✓ String representation works")
    
    print("✅ ImageNormalizer tests passed")
except Exception as e:
    print(f"❌ ImageNormalizer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 3: IGValidator attribution validation
print("\n[TEST 3] Testing IGValidator attribution validation...")
try:
    # Test with valid attributions
    valid_attr_img = np.random.randn(3, 224, 224) * 0.1  # Small random values
    valid_attr_phys = np.random.randn(10) * 0.1
    
    results = IGValidator.validate_attributions(
        valid_attr_img, valid_attr_phys, verbose=False
    )
    
    assert not results['has_nan'], "Should not have NaN"
    assert not results['all_zero'], "Should not be all zeros"
    assert results['max_img_attribution'] > 0, "Should have positive attribution"
    print("  ✓ Valid attribution detection works")
    
    # Test with all-zero attributions
    zero_attr_img = np.zeros((3, 224, 224))
    zero_attr_phys = np.zeros(10)
    
    results_zero = IGValidator.validate_attributions(
        zero_attr_img, zero_attr_phys, verbose=False
    )
    
    assert results_zero['all_zero'], "Should detect all-zero attributions"
    print("  ✓ All-zero detection works")
    
    # Test with NaN
    nan_attr_img = np.full((3, 224, 224), np.nan)
    nan_attr_phys = np.zeros(10)
    
    results_nan = IGValidator.validate_attributions(
        nan_attr_img, nan_attr_phys, verbose=False
    )
    
    assert results_nan['has_nan'], "Should detect NaN"
    print("  ✓ NaN detection works")
    
    print("✅ Attribution validation tests passed")
except Exception as e:
    print(f"❌ Attribution validation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 4: IGValidator gate mechanism validation
print("\n[TEST 4] Testing IGValidator gate mechanism validation...")
try:
    # Test with varying gate values
    gate_values = np.linspace(0.1, 0.9, 100)  # Values across range
    
    results = IGValidator.validate_gate_mechanism(
        gate_values, verbose=False
    )
    
    assert results['gate_varies'], "Gate should show variation"
    assert results['gate_range_used'], "Gate should use full range"
    assert results['gate_min'] > 0, "Gate min should be > 0"
    assert results['gate_max'] < 1, "Gate max should be < 1"
    print("  ✓ Gate variation detection works")
    
    # Test with constant gate (saturated)
    const_gate = np.full(100, 0.5)
    results_const = IGValidator.validate_gate_mechanism(
        const_gate, verbose=False
    )
    
    # Constant gate will have variance 0, so it won't be marked as varying
    assert results_const['gate_variance'] < 1e-6, "Should detect constant gate"
    print("  ✓ Gate saturation detection works")
    
    # Test with gate at extremes
    extreme_gate = np.concatenate([
        np.full(50, 0.01),  # Near 0
        np.full(50, 0.99)   # Near 1
    ])
    results_extreme = IGValidator.validate_gate_mechanism(
        extreme_gate, verbose=False
    )
    
    assert results_extreme['gate_varies'], "Should detect variation"
    print("  ✓ Gate extreme range detection works")
    
    print("✅ Gate mechanism validation tests passed")
except Exception as e:
    print(f"❌ Gate mechanism test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 5: IGValidator full validation
print("\n[TEST 5] Testing IGValidator full validation...")
try:
    # Create realistic test data
    attr_img = np.random.randn(3, 64, 64) * 0.05
    attr_phys = np.random.randn(10) * 0.05
    gate_values = np.linspace(0.2, 0.8, 50)
    
    full_results = IGValidator.full_validation(
        attr_img, attr_phys, gate_values, verbose=False
    )
    
    assert 'attributions' in full_results, "Should have attribution results"
    assert 'gate' in full_results, "Should have gate results"
    assert 'all_pass' in full_results, "Should have overall status"
    assert full_results['all_pass'], "Should pass all checks"
    print("  ✓ Full validation framework works")
    
    print("✅ Full validation tests passed")
except Exception as e:
    print(f"❌ Full validation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 6: L2 norm justification code present
print("\n[TEST 6] Verifying L2 norm justification...")
try:
    import inspect
    from xai import SaliencyMap
    
    source = inspect.getsource(SaliencyMap.explain)
    
    # Check for academic references
    assert "L2 NORM AGGREGATION JUSTIFICATION" in source, "Missing justification comment"
    assert "Simonyan" in source, "Missing Simonyan reference"
    assert "Montavon" in source, "Missing Montavon reference"
    assert "OpenCV" in source or "PyTorch" in source, "Missing implementation reference"
    
    print("  ✓ L2 norm justification includes academic references")
    print("  ✓ References include Simonyan et al. (2013)")
    print("  ✓ References include Montavon et al. (2015)")
    
    print("✅ L2 norm justification verified")
except Exception as e:
    print(f"❌ L2 norm verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 7: Syntax validation
print("\n[TEST 7] Verifying Python syntax...")
try:
    import py_compile
    py_compile.compile('src/xai/__init__.py', doraise=True)
    print("✅ Python syntax verified")
except Exception as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TIER 2 TESTS PASSED ✅")
print("=" * 80)
print("\nSummary:")
print("  ✅ ImageNormalizer class works correctly")
print("    - Default ImageNet normalization")
print("    - Custom normalization support")
print("    - Normalize/denormalize round-trip")
print("    - PyTorch tensor support")
print("")
print("  ✅ IGValidator class works correctly")
print("    - Attribution validation (NaN, zero, magnitude)")
print("    - Gate mechanism validation (variance, range)")
print("    - Full validation framework")
print("")
print("  ✅ L2 norm justification added with academic references")
print("    - Simonyan et al. (2013)")
print("    - Montavon et al. (2015)")
print("    - Implementation best practices")
print("")
print("  ✅ All syntax valid")
print("\nThe code is ready for analysis phase!")
print("=" * 80)
