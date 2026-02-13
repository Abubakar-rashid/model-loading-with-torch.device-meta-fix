#!/usr/bin/env python3
"""
Test script to verify that the meta device fix works correctly.

This script tests:
1. The python_linspace helper function
2. Model loading with meta device (the original error case)
"""

import sys

import torch


# Test 1: Verify python_linspace function
print("=" * 80)
print("Test 1: Testing python_linspace helper function")
print("=" * 80)

from transformers.modeling_utils import python_linspace


# Test basic functionality
result = python_linspace(0, 1, 5)
expected = [0.0, 0.25, 0.5, 0.75, 1.0]
print(f"python_linspace(0, 1, 5) = {result}")
print(f"Expected: {expected}")
assert result == expected, f"Expected {expected}, got {result}"
print("✓ Basic test passed")

# Test single step
result = python_linspace(0, 10, 1)
expected = [0]
print(f"\npython_linspace(0, 10, 1) = {result}")
print(f"Expected: {expected}")
assert result == expected, f"Expected {expected}, got {result}"
print("✓ Single step test passed")

# Test with drop_path_rate pattern (common in models)
result = python_linspace(0, 0.1, 12)
print(f"\npython_linspace(0, 0.1, 12) = {result}")
assert len(result) == 12, f"Expected 12 values, got {len(result)}"
assert result[0] == 0.0, f"Expected first value to be 0.0, got {result[0]}"
assert abs(result[-1] - 0.1) < 1e-10, f"Expected last value to be 0.1, got {result[-1]}"
print("✓ Drop path rate pattern test passed")

print("\n" + "=" * 80)
print("Test 2: Testing model initialization on meta device")
print("=" * 80)

# Test that models can be initialized on meta device without errors
try:
    # Test with a model that uses drop_path_rate
    from transformers import AutoConfig
    
    # Try TimesformerConfig
    print("\nTesting TimesformerConfig initialization on meta device...")
    config = AutoConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
    config.num_hidden_layers = 4  # Reduce size for faster testing
    
    # Initialize on meta device - this should not raise an error anymore
    with torch.device("meta"):
        from transformers.models.timesformer.modeling_timesformer import TimesformerModel
        model = TimesformerModel(config)
    
    print("✓ Model successfully initialized on meta device")
    
except Exception as e:
    print(f"✗ Error during model initialization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("Test 3: Testing original error case (if possible)")
print("=" * 80)

# Note: We can't test the exact original case (briaai/RMBG-2.0) without downloading
# the model, but we can test the pattern that was failing

try:
    print("\nTesting pattern that was causing 'Tensor.item() cannot be called on meta tensors'...")
    
    # This pattern was failing before
    with torch.device("meta"):
        # Create a config with drop_path_rate
        from transformers import BeitConfig
        config = BeitConfig(drop_path_rate=0.1, num_hidden_layers=4)
        
        # This will internally use python_linspace instead of torch.linspace(...).item()
        from transformers.models.beit.modeling_beit import BeitModel
        model = BeitModel(config)
    
    print("✓ Pattern test passed - no 'Tensor.item() cannot be called on meta tensors' error")
    
except RuntimeError as e:
    if "Tensor.item() cannot be called on meta tensors" in str(e):
        print(f"✗ The fix didn't work! Still getting the error: {e}")
        sys.exit(1)
    else:
        # Some other error
        raise

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nThe meta device fix is working correctly.")
print("Models can now be initialized on meta device without")
print("encountering 'Tensor.item() cannot be called on meta tensors' errors.")
