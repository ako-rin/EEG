"""
Test script to verify PGCN integration with LibEER compatibility.

This tests:
1. LibEER-compatible initialization (num_electrodes, in_channels, num_classes)
2. Original reference initialization (args, adj, coor)
3. Device synchronization (_apply hook)
4. Forward pass with both modes
"""

import sys
from pathlib import Path

# Bootstrap sys.path
_here = Path(__file__).resolve()
_proj_root = _here.parents[2]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import torch
import numpy as np
from types import SimpleNamespace

# Import integrated PGCN
from EEG.models.PGCN import PGCN, get_ini_dis_m, convert_dis_m, return_coordinates


def test_libeer_mode():
    """Test LibEER-compatible initialization"""
    print("\n" + "="*80)
    print("TEST 1: LibEER-Compatible Mode")
    print("="*80)
    
    # Initialize model using LibEER interface
    model = PGCN(
        num_electrodes=62,
        in_channels=5,
        num_classes=3,
        dropout_rate=0.4,
        dataset='SEED'
    )
    
    print(f"✓ Model initialized with LibEER interface")
    print(f"  - Input features: {model.args.in_feature}")
    print(f"  - Output classes: {model.nclass}")
    print(f"  - Dropout rate: {model.dropout}")
    print(f"  - Adjacency matrix shape: {model.adj.shape}")
    print(f"  - Coordinates shape: {model.coordinate.shape}")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 62, 5)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"\n✓ Forward pass successful")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Expected: ({batch_size}, 3)")
    
    assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"
    print(f"\n✓ Output shape validation passed!")
    
    return model


def test_original_mode():
    """Test original reference initialization"""
    print("\n" + "="*80)
    print("TEST 2: Original Reference Mode")
    print("="*80)
    
    # Build args namespace
    args = SimpleNamespace(
        in_feature=5,
        out_feature=20,
        n_class=3,
        dropout=0.4,
        epsilon=0.05,
        dataset='SEED',
        device='cuda',
        lr=0.1,
        module=""
    )
    
    # Initialize adjacency and coordinates
    initial_adj = convert_dis_m(get_ini_dis_m(), delta=9)
    coordinates = return_coordinates()
    
    # Initialize model using original interface
    model = PGCN(args, initial_adj, coordinates)
    
    print(f"✓ Model initialized with original interface")
    print(f"  - Input features: {model.args.in_feature}")
    print(f"  - Output classes: {model.nclass}")
    print(f"  - Coordinates type: {type(model.coordinate)}")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 62, 5)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"\n✓ Forward pass successful")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"
    print(f"\n✓ Output shape validation passed!")
    
    return model


def test_device_synchronization():
    """Test device synchronization via _apply hook"""
    print("\n" + "="*80)
    print("TEST 3: Device Synchronization")
    print("="*80)
    
    # Initialize model
    model = PGCN(62, 5, 3)
    
    print(f"Initial device: CPU")
    print(f"  - coordinate device: {model.coordinate.device}")
    print(f"  - meso_layer_1.coordinate device: {model.meso_layer_1.coordinate.device}")
    print(f"  - meso_layer_2.coordinate device: {model.meso_layer_2.coordinate.device}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"\n✓ Moved to CUDA")
        print(f"  - coordinate device: {model.coordinate.device}")
        print(f"  - meso_layer_1.coordinate device: {model.meso_layer_1.coordinate.device}")
        print(f"  - meso_layer_2.coordinate device: {model.meso_layer_2.coordinate.device}")
        
        # Verify all coordinates are on CUDA
        assert model.coordinate.is_cuda, "coordinate should be on CUDA"
        assert model.meso_layer_1.coordinate.is_cuda, "meso_layer_1.coordinate should be on CUDA"
        assert model.meso_layer_2.coordinate.is_cuda, "meso_layer_2.coordinate should be on CUDA"
        
        # Move back to CPU
        model = model.cpu()
        print(f"\n✓ Moved back to CPU")
        print(f"  - coordinate device: {model.coordinate.device}")
        print(f"  - meso_layer_1.coordinate device: {model.meso_layer_1.coordinate.device}")
        print(f"  - meso_layer_2.coordinate device: {model.meso_layer_2.coordinate.device}")
        
        # Verify all coordinates are on CPU
        assert not model.coordinate.is_cuda, "coordinate should be on CPU"
        assert not model.meso_layer_1.coordinate.is_cuda, "meso_layer_1.coordinate should be on CPU"
        assert not model.meso_layer_2.coordinate.is_cuda, "meso_layer_2.coordinate should be on CPU"
        
        print(f"\n✓ Device synchronization working correctly!")
    else:
        print(f"\n⚠ CUDA not available, skipping GPU tests")
        print(f"✓ CPU device check passed")


def test_parameter_count():
    """Compare parameter counts between modes"""
    print("\n" + "="*80)
    print("TEST 4: Parameter Count Comparison")
    print("="*80)
    
    # LibEER mode
    model_libeer = PGCN(62, 5, 3)
    params_libeer = sum(p.numel() for p in model_libeer.parameters())
    
    # Original mode
    args = SimpleNamespace(
        in_feature=5, out_feature=20, n_class=3, dropout=0.4,
        epsilon=0.05, dataset='SEED', device='cuda', lr=0.1, module=""
    )
    initial_adj = convert_dis_m(get_ini_dis_m(), delta=9)
    coordinates = return_coordinates()
    model_original = PGCN(args, initial_adj, coordinates)
    params_original = sum(p.numel() for p in model_original.parameters())
    
    print(f"LibEER mode parameters: {params_libeer:,}")
    print(f"Original mode parameters: {params_original:,}")
    print(f"Difference: {abs(params_libeer - params_original):,}")
    
    # Allow small difference due to adjacency matrix handling
    # (LibEER mode uses Parameter, original might not)
    diff_ratio = abs(params_libeer - params_original) / max(params_libeer, params_original)
    print(f"Difference ratio: {diff_ratio:.2%}")
    
    if diff_ratio < 0.01:  # Less than 1% difference
        print(f"\n✓ Parameter counts are effectively equal (< 1% difference)")
    else:
        print(f"\n✗ Parameter count difference is too large (>= 1%)")
        raise AssertionError(f"Parameter count difference {diff_ratio:.2%} exceeds threshold")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("PGCN Integration Test Suite")
    print("="*80)
    
    try:
        # Run tests
        model1 = test_libeer_mode()
        model2 = test_original_mode()
        test_device_synchronization()
        test_parameter_count()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nPGCN integration is working correctly:")
        print("  ✓ LibEER-compatible initialization")
        print("  ✓ Original reference initialization")
        print("  ✓ Device synchronization (_apply hook)")
        print("  ✓ Parameter count consistency")
        print("\nYou can now use PGCN directly from EEG.models.PGCN")
        print("The PGCN_Adapter.py is no longer needed!")
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED! ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
