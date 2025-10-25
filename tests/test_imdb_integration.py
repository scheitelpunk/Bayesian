"""
Quick test to verify real IMDB dataset integration.
"""

import sys
sys.path.insert(0, 'C:\\dev\\coding\\Bayesian')

def test_imdb_loading():
    """Test that IMDB dataset loads correctly."""
    from datasets import load_dataset

    print("Testing IMDB dataset loading...")

    # Load a small sample
    dataset = load_dataset('stanfordnlp/imdb', split='train', streaming=True)

    # Get first 3 samples
    samples = list(dataset.take(3))

    print(f"\nLoaded {len(samples)} samples")

    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Label: {'POSITIVE' if sample['label'] == 1 else 'NEGATIVE'}")
        print(f"  Text: {sample['text'][:100]}...")

    print("\n[SUCCESS] IMDB dataset integration working!")
    return True


def test_tokenizer():
    """Test SimpleTokenizer."""
    from examples.real_data_demo import SimpleTokenizer

    print("\nTesting SimpleTokenizer...")

    tokenizer = SimpleTokenizer(vocab_size=100, max_length=20)

    # Build vocab
    texts = ["this is a test", "another test sentence"]
    tokenizer.build_vocab(texts)

    # Tokenize
    result = tokenizer(texts, max_length=20)

    print(f"  Vocabulary size: {len(tokenizer.vocab)}")
    print(f"  Tokenized shape: {result['input_ids'].shape}")

    print("\n[SUCCESS] Tokenizer working!")
    return True


def test_dataloader():
    """Test DataLoader creation."""
    from datasets import load_dataset
    from examples.real_data_demo import SimpleTokenizer, create_dataloader

    print("\nTesting DataLoader creation...")

    # Load small dataset
    dataset = load_dataset('stanfordnlp/imdb', split='train', streaming=True)
    dataset = list(dataset.take(10))

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=100, max_length=20)
    texts = [item['text'] for item in dataset]
    tokenizer.build_vocab(texts)

    # Create dataloader
    loader = create_dataloader(dataset, tokenizer, batch_size=2, max_length=20)

    # Get first batch
    batch = next(iter(loader))

    print(f"  Batch keys: {batch.keys()}")
    print(f"  Input shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['label'].shape}")

    print("\n[SUCCESS] DataLoader working!")
    return True


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("IMDB INTEGRATION TEST SUITE")
        print("=" * 70)

        test_imdb_loading()
        test_tokenizer()
        test_dataloader()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED [SUCCESS]")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
