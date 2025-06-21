#!/usr/bin/env python3
"""
Test script to verify Rust k-mer extraction module builds and works correctly.
"""


def test_rust_import():
    """Test if we can import the Rust module."""
    try:
        from kmer_counter_rs import extract_kmers_rs, extract_kmer_rs

        print("✓ Successfully imported both functions from Rust module")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Rust module: {e}")
        return False


def test_kmer_extraction():
    """Test k-mer extraction functionality."""
    try:
        from kmer_counter_rs import extract_kmers_rs

        # Test sequence
        test_seq = b"ATCGATCG"
        k = 3

        # The Rust function expects (sequence, k, perform_strict_dna_check)
        kmers = extract_kmers_rs(test_seq, k, False)
        print(f"✓ Extracted {len(kmers)} k-mers from test sequence")
        print(f"  First few k-mers: {[kmer.decode() for kmer in kmers[:3]]}")
        return True
    except Exception as e:
        print(f"✗ Failed k-mer extraction test: {e}")
        return False


if __name__ == "__main__":
    print("Testing Rust k-mer extraction module...")

    rust_works = test_rust_import()
    if rust_works:
        test_kmer_extraction()
    # else: # Fallback logic removed
        # print("Testing Python fallback...")
        # test_fallback()

    print("\nTo build the Rust module, run:")
    print("  cd kmer_counter_rs")
    print("  maturin develop --release")
