"""Smoke test for 6 radical architectures (tests 39-44)."""
import torch
from utils.model import create_model
from utils.config import Config

def test_architecture(arch_name, input_shape=(2, 1, 24, 16, 16)):
    """Test single architecture instantiation and forward pass."""
    print(f"\n{'='*70}")
    print(f"Testing: {arch_name}")
    print(f"{'='*70}")
    
    # Create minimal config
    config = Config()
    config.model.arch = arch_name
    config.data.tile_size = 16
    config.data.depth = 24
    config.device = torch.device('cpu')
    
    try:
        # Create model
        model, n_params = create_model(config)
        print(f"Model created: {n_params:,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(*input_shape)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Input shape:  {tuple(dummy_input.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Verify output shape
        assert output.shape == (input_shape[0], 1) or output.shape == (input_shape[0],), \
            f"Expected output shape (B, 1) or (B,), got {output.shape}"
        
        print(f"OK  {arch_name:30s} - params: {n_params:,}")
        return True
        
    except Exception as e:
        print(f"ERR {arch_name:30s} - ERROR: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("RADICAL ARCHITECTURES SMOKE TEST")
    print("="*70)
    
    architectures = [
        "vit3d",        # Vision Transformer 3D
        "swin3d",       # Swin Transformer 3D
        "convnext3d",   # ConvNeXt 3D
        "xcit3d",       # XCiT 3D
        "nnunet3d",     # nnU-Net 3D
        "slot3d",       # Slot Attention 3D
    ]
    
    results = {}
    for arch in architectures:
        results[arch] = test_architecture(arch)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for arch, success in results.items():
        status = "OK " if success else "ERR"
        print(f"  {status} {arch}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\nOK  All radical architectures work correctly!")
    else:
        print(f"\nERR {total - passed} architecture(s) failed")


if __name__ == "__main__":
    main()
