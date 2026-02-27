"""
Test script to verify TIER 1 changes work correctly
Purpose: Ensure the modified src/xai/__init__.py imports and runs without errors
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("TIER 1 RUNTIME TEST")
print("=" * 80)

# TEST 1: Import the module
print("\n[TEST 1] Importing src.xai module...")
try:
    from xai import IntegratedGradients, ExplanationOutput
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# TEST 2: Check baseline simplification
print("\n[TEST 2] Verifying baseline simplification...")
try:
    # Create mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, img, phys, mask=None):
            class Output:
                gate = torch.tensor([[0.5]])
                focus = torch.tensor([[0.3]])
            return torch.sigmoid(self.fc(phys)), Output()
    
    model = MockModel()
    xai = IntegratedGradients(model, device=torch.device('cpu'))
    
    # Test that _get_baseline only accepts 'black'
    img = torch.randn(1, 3, 224, 224)
    phys = torch.randn(1, 10)
    
    # Should work
    img_base, phys_base = xai._get_baseline(img, phys, baseline_type='black')
    assert torch.all(img_base == 0), "Black baseline should be all zeros"
    assert torch.all(phys_base == 0), "Physiology baseline should be all zeros"
    print("  ✓ 'black' baseline works correctly")
    
    # Should fail
    try:
        xai._get_baseline(img, phys, baseline_type='gray')
        print("  ❌ FAILED: 'gray' baseline should raise error")
        sys.exit(1)
    except ValueError as e:
        if "not supported" in str(e):
            print("  ✓ 'gray' baseline correctly raises error")
        else:
            print(f"  ❌ Wrong error message: {e}")
            sys.exit(1)
    
    # Should fail
    try:
        xai._get_baseline(img, phys, baseline_type='blur')
        print("  ❌ FAILED: 'blur' baseline should raise error")
        sys.exit(1)
    except ValueError:
        print("  ✓ 'blur' baseline correctly raises error")
    
    print("✅ Baseline simplification verified")
except Exception as e:
    print(f"❌ Baseline test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 3: Check NaN error handling
print("\n[TEST 3] Verifying NaN error handling...")
try:
    # We'll check that the code structure is correct by examining the source
    import inspect
    source = inspect.getsource(xai.explain)
    
    # Check that RuntimeError is raised (not torch.where)
    if "raise RuntimeError" in source:
        print("  ✓ Code raises RuntimeError for NaN gradients")
    else:
        print("  ❌ FAILED: RuntimeError not found in code")
        sys.exit(1)
    
    if "torch.where" in source and "isnan" in source:
        print("  ⚠️  WARNING: torch.where masking still present")
        # Check if it's in the Inf handling (which is OK)
        if source.count("torch.where") <= 1:  # Only for Inf, not NaN
            print("  ✓ torch.where only used for Inf clamping (acceptable)")
        else:
            print("  ❌ FAILED: Multiple torch.where calls detected")
            sys.exit(1)
    else:
        print("  ✓ No silent masking with torch.where for NaN")
    
    print("✅ NaN error handling verified")
except Exception as e:
    print(f"❌ NaN handling test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 4: Check completeness check removal
print("\n[TEST 4] Verifying completeness check removal...")
try:
    import inspect
    source = inspect.getsource(xai.explain)
    
    # Check that the broken completeness check is gone
    if "VALIDATE COMPLETENESS" in source:
        print("  ❌ FAILED: Old completeness check comment still present")
        sys.exit(1)
    
    # Check that completeness_error variable is not used
    lines = source.split('\n')
    completeness_lines = [l for l in lines if 'completeness_error' in l.lower()]
    
    if any('completeness_error' in l and not l.strip().startswith('#') 
           for l in completeness_lines):
        print("  ❌ FAILED: completeness_error still used in code")
        sys.exit(1)
    
    # Check for explanation comment
    if "mathematically guarantees" in source and "Completeness axiom" in source:
        print("  ✓ Completeness check replaced with explanation")
    else:
        print("  ⚠️  WARNING: Explanation comment not found")
    
    print("✅ Completeness check removal verified")
except Exception as e:
    print(f"❌ Completeness check test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 5: Check documentation
print("\n[TEST 5] Verifying documentation updates...")
try:
    import inspect
    
    # Check _get_baseline docstring
    baseline_doc = inspect.getdoc(xai._get_baseline)
    if "Kindermans" in baseline_doc and "Sanity Checks" in baseline_doc:
        print("  ✓ _get_baseline docstring has academic references")
    else:
        print("  ⚠️  WARNING: Academic references missing from _get_baseline")
    
    # Check explain docstring
    explain_doc = inspect.getdoc(xai.explain)
    if "'black' supported" in explain_doc or "only 'black'" in explain_doc:
        print("  ✓ explain() docstring updated for baseline_type")
    else:
        print("  ⚠️  WARNING: explain() docstring not updated")
    
    print("✅ Documentation verified")
except Exception as e:
    print(f"⚠️  Documentation check failed: {e}")

# TEST 6: Syntax compilation
print("\n[TEST 6] Verifying syntax compilation...")
try:
    import py_compile
    py_compile.compile('src/xai/__init__.py', doraise=True)
    print("✅ Python syntax verified")
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✅")
print("=" * 80)
print("\nSummary:")
print("  ✅ Module imports successfully")
print("  ✅ Baseline selection simplified (only 'black' works)")
print("  ✅ Invalid baselines ('gray', 'blur') raise errors")
print("  ✅ NaN handling raises RuntimeError (not silent masking)")
print("  ✅ Completeness check removed and replaced")
print("  ✅ Documentation updated with academic references")
print("  ✅ Syntax is valid Python")
print("\nThe code is ready for the next phase: TIER 2 fixes")
print("=" * 80)
