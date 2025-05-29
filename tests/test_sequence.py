"""
Pytest unit tests for GenomicSequence dataclass and k-mer extraction functions
from strainr.sequence. These tests assume the file is in the root directory,
and 'src' is a subdirectory.
"""

import pytest
from typing import List, Set  # Added Set for type hinting if needed

import mmh3  # Ensure mmh3 is imported for direct use if needed in tests, though not strictly for this fix

from src.strainr.sequence import (
    GenomicSequence,
    extract_kmers_from_sequence,
    VALID_DNA_BYTES,
)

# --- Fixtures ---


@pytest.fixture
def valid_sequence_id_fixture() -> str:
    """Provides a valid sequence ID for testing."""
    return "test_seq_123"


@pytest.fixture
def valid_dna_bytes_fixture() -> bytes:
    """Provides valid DNA sequence data as bytes."""
    return b"ACGTNACGT"


@pytest.fixture
def genomic_sequence_fixture(
    valid_sequence_id_fixture: str, valid_dna_bytes_fixture: bytes
) -> GenomicSequence:
    """Provides a GenomicSequence instance with valid data."""
    return GenomicSequence(
        sequence_id=valid_sequence_id_fixture, sequence_data=valid_dna_bytes_fixture
    )


# --- Tests for GenomicSequence ---


def test_genomic_sequence_creation_success(
    genomic_sequence_fixture: GenomicSequence,
    valid_sequence_id_fixture: str,
    valid_dna_bytes_fixture: bytes,
):
    assert genomic_sequence_fixture.sequence_id == valid_sequence_id_fixture
    assert genomic_sequence_fixture.sequence_data == valid_dna_bytes_fixture


def test_genomic_sequence_empty_sequence_error(valid_sequence_id_fixture: str):
    with pytest.raises(ValueError, match="Sequence data must be non-empty."):
        GenomicSequence(sequence_id=valid_sequence_id_fixture, sequence_data=b"")


def test_genomic_sequence_invalid_type_error(valid_sequence_id_fixture: str):
    with pytest.raises(TypeError, match="Sequence data must be bytes, got str."):
        GenomicSequence(sequence_id=valid_sequence_id_fixture, sequence_data="ACGT")  # type: ignore


def test_genomic_sequence_invalid_dna_chars_error(valid_sequence_id_fixture: str):
    with pytest.raises(
        ValueError,
        match=r"Sequence contains invalid DNA bytes: \{'X' \(byte: 88\)\}. Allowed bytes correspond to characters: 'A', 'C', 'G', 'N', 'T'.",
    ):
        GenomicSequence(sequence_id=valid_sequence_id_fixture, sequence_data=b"ACGTX")


def test_genomic_sequence_lowercase_conversion_and_validation(
    valid_sequence_id_fixture: str,
):
    with pytest.raises(
        ValueError,
        match=r"Sequence contains invalid DNA bytes: \{'a' \(byte: 97\), 'c' \(byte: 99\), 'g' \(byte: 103\), 't' \(byte: 116\)\}. Allowed bytes correspond to characters: 'A', 'C', 'G', 'N', 'T'.",
    ):
        GenomicSequence(sequence_id=valid_sequence_id_fixture, sequence_data=b"acgt")


def test_genomic_sequence_non_ascii_decode_error(valid_sequence_id_fixture: str):
    # Bytes that are not valid DNA bytes (and also happen to be non-ASCII in this example)
    with pytest.raises(
        ValueError,
        match=r"Sequence contains invalid DNA bytes: \{byte 164, byte 195\}. Allowed bytes correspond to characters: 'A', 'C', 'G', 'N', 'T'.",
    ):  # Corrected regex
        GenomicSequence(
            sequence_id=valid_sequence_id_fixture, sequence_data=b"ACGT\xc3\xa4N"
        )


def test_genomic_sequence_length(
    genomic_sequence_fixture: GenomicSequence, valid_dna_bytes_fixture: bytes
):
    assert len(genomic_sequence_fixture) == len(valid_dna_bytes_fixture)


def test_genomic_sequence_string_representation(
    genomic_sequence_fixture: GenomicSequence, valid_dna_bytes_fixture: bytes
):
    assert str(genomic_sequence_fixture) == valid_dna_bytes_fixture.decode("ascii")


def test_genomic_sequence_getitem(genomic_sequence_fixture: GenomicSequence):
    assert genomic_sequence_fixture[0] == "A"
    assert genomic_sequence_fixture[-1] == "T"
    with pytest.raises(IndexError):
        _ = genomic_sequence_fixture[len(genomic_sequence_fixture.sequence_data)]


def test_genomic_sequence_length(
    genomic_sequence_fixture: GenomicSequence, valid_dna_bytes_fixture: bytes
):
    assert len(genomic_sequence_fixture) == len(valid_dna_bytes_fixture)


def test_genomic_sequence_string_representation(
    genomic_sequence_fixture: GenomicSequence, valid_dna_bytes_fixture: bytes
):
    assert str(genomic_sequence_fixture) == valid_dna_bytes_fixture.decode("ascii")


def test_genomic_sequence_getitem(genomic_sequence_fixture: GenomicSequence):
    assert genomic_sequence_fixture[0] == "A"
    assert genomic_sequence_fixture[-1] == "T"
    with pytest.raises(IndexError):
        _ = genomic_sequence_fixture[len(genomic_sequence_fixture.sequence_data)]


def test_genomic_sequence_hash(
    valid_sequence_id_fixture: str, valid_dna_bytes_fixture: bytes
):
    seq1 = GenomicSequence(
        sequence_id=valid_sequence_id_fixture, sequence_data=valid_dna_bytes_fixture
    )
    seq2 = GenomicSequence(
        sequence_id=valid_sequence_id_fixture, sequence_data=valid_dna_bytes_fixture
    )
    assert isinstance(hash(seq1), int)
    assert hash(seq1) == hash(seq2)

    seq3 = GenomicSequence(
        sequence_id="other_id", sequence_data=valid_dna_bytes_fixture
    )
    assert hash(seq1) == hash(seq3)

    seq4 = GenomicSequence(
        sequence_id=valid_sequence_id_fixture,
        sequence_data=b"ACGTACGT",  # Different sequence
    )

    assert isinstance(
        hash(seq1), int
    )  # This should pass if mmh3.hash_bytes returns an int
    assert hash(seq1) == hash(seq2)  # Same data, same hash
    assert (
        hash(seq1) == hash(seq3)
    )  # Same data (sequence_id not part of hash by default for frozen dataclass unless explicitly included)
    assert hash(seq1) != hash(seq4)  # Different data, different hash


def test_genomic_sequence_is_valid_dna(genomic_sequence_fixture: GenomicSequence):
    assert (
        genomic_sequence_fixture.is_valid_dna() is True
    )  # Should always be true for a successfully constructed object


# --- Tests for extract_kmers_from_sequence ---


def test_extract_kmers_basic(genomic_sequence_fixture: GenomicSequence):
    kmers = extract_kmers_from_sequence(genomic_sequence_fixture, kmer_length=3)
    expected = [b"ACG", b"CGT", b"GTN", b"TNA", b"NAC", b"ACG", b"CGT"]
    assert kmers == expected


def test_extract_kmers_kmer_length_equals_seq_length(
    genomic_sequence_fixture: GenomicSequence,
):
    k_len = len(genomic_sequence_fixture.sequence_data)
    kmers = extract_kmers_from_sequence(genomic_sequence_fixture, kmer_length=k_len)
    assert kmers == [genomic_sequence_fixture.sequence_data]


def test_extract_kmers_empty_sequence_error():
    # This case is tricky: GenomicSequence itself will raise error for empty sequence data
    with pytest.raises(ValueError, match="Sequence data must be non-empty."):
        gs = GenomicSequence(sequence_id="test_empty", sequence_data=b"")
        # The following line won't be reached if GenomicSequence init fails
        # extract_kmers_from_sequence(gs, kmer_length=3)


def test_extract_kmers_kmer_length_too_large_error(
    genomic_sequence_fixture: GenomicSequence,
):
    with pytest.raises(
        ValueError, match="K-mer length .* cannot exceed sequence length .*"
    ):
        extract_kmers_from_sequence(
            genomic_sequence_fixture,
            kmer_length=len(genomic_sequence_fixture.sequence_data) + 1,
        )


def test_extract_kmers_kmer_length_positive_error():
    gs = GenomicSequence(sequence_id="test_err_klen_pos", sequence_data=b"ACGT")
    with pytest.raises(ValueError, match=r"kmer_length must be positive, got 0\."):
        extract_kmers_from_sequence(gs, kmer_length=0)


def test_extract_kmers_invalid_kmer_length_type_error(
    genomic_sequence_fixture: GenomicSequence,
):
    with pytest.raises(TypeError, match="kmer_length must be an integer"):
        extract_kmers_from_sequence(genomic_sequence_fixture, kmer_length="3")  # type: ignore
