"""
Quick test to verify uncertainty extraction works correctly.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from benchmarks.benchmark_transformer_comparison import BayesianTransformerWrapper

def test_uncertainty_extraction():
    """Test that BayesianTransformerWrapper returns uncertainty correctly."""

    config = {
        'd_model': 128,
        'n_heads': 4,
        'vocab_size': 10000,
        'dropout': 0.1,
        'd_ff': 512,
        'k_permutations': 10,
        'num_samples': 5,
        'using_improved_stats': True  # Use improved stats encoder v2
    }

    model = BayesianTransformerWrapper(config)
    model.eval()

    # Create dummy input
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, config['vocab_size'], (batch_size, seq_len))

    print("Testing BayesianTransformerWrapper uncertainty extraction...")
    print(f"Input shape: {x.shape}")

    # Test with return_uncertainty=True
    with torch.no_grad():
        outputs = model(x, return_uncertainty=True)

    print(f"\n[OK] Output type: {type(outputs)}")

    if isinstance(outputs, dict):
        print(f"[OK] Output keys: {outputs.keys()}")

        # Check logits
        logits = outputs.get('logits')
        if logits is not None:
            print(f"[OK] Logits shape: {logits.shape} (expected: ({batch_size}, 2))")
            assert logits.shape == (batch_size, 2), f"Wrong logits shape: {logits.shape}"

        # Check epistemic uncertainty
        epistemic = outputs.get('epistemic_uncertainty')
        if epistemic is not None:
            print(f"[OK] Epistemic uncertainty shape: {epistemic.shape} (expected: ({batch_size},))")
            print(f"  - Mean: {epistemic.mean():.6f}")
            print(f"  - Std: {epistemic.std():.6f}")
            print(f"  - Range: [{epistemic.min():.6f}, {epistemic.max():.6f}]")
            assert epistemic.shape == (batch_size,), f"Wrong uncertainty shape: {epistemic.shape}"

            # Check that uncertainty is not all zeros
            if epistemic.abs().max() > 1e-6:
                print(f"[OK] Uncertainty is non-zero (good!)")
            else:
                print(f"[WARN] Warning: Uncertainty is all zeros or very small")
        else:
            print(f"[FAIL] No epistemic_uncertainty in output!")
            return False

        # Check aleatoric uncertainty
        aleatoric = outputs.get('aleatoric_uncertainty')
        if aleatoric is not None:
            print(f"[OK] Aleatoric uncertainty shape: {aleatoric.shape}")

    else:
        print(f"[FAIL] Output is not a dict when return_uncertainty=True!")
        return False

    print("\n[SUCCESS] All tests passed!")
    return True

if __name__ == '__main__':
    success = test_uncertainty_extraction()
    sys.exit(0 if success else 1)
