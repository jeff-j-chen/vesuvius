"""Extended smoke test with context-sized inputs for radical architectures."""
import torch
from utils.model import create_model
from utils.config import Config

def test_with_context(arch_name):
    """Test architecture with context-sized inputs (48x48)."""
    print(f"\nTesting {arch_name} with context (48x48)...")
    
    config = Config()
    config.model.arch = arch_name
    config.data.tile_size = 16
    config.data.depth = 24
    config.data.context_size = 48
    config.device = torch.device('cpu')
    
    try:
        model, n_params = create_model(config)
        
        # Test with context-sized input
        dummy_input = torch.randn(2, 1, 24, 48, 48)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  Input: {tuple(dummy_input.shape)} -> Output: {tuple(output.shape)}")
        print(f"  OK  Context test passed")
        return True
        
    except Exception as e:
        print(f"  ERR Context test failed: {str(e)[:80]}")
        return False


def main():
    print("="*70)
    print("EXTENDED CONTEXT TEST FOR RADICAL ARCHITECTURES")
    print("="*70)
    
    architectures = ["vit3d", "swin3d", "convnext3d", "xcit3d", "nnunet3d", "slot3d"]
    
    results = {}
    for arch in architectures:
        results[arch] = test_with_context(arch)
    
    print("\n" + "="*70)
    passed = sum(results.values())
    print(f"Context tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("OK  All architectures handle context inputs!")
    else:
        print(f"ERR {len(results) - passed} failed with context inputs")


if __name__ == "__main__":
    main()
