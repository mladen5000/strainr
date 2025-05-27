"""
Pytest unit tests for GenomicSequence dataclass and k-mer extraction functions
from src.strainr.sequence. These tests assume the file is in the root directory,
and 'src' is a subdirectory.
"""

import pytest
import dataclasses  # For FrozenInstanceError

# Assuming src.strainr.* is in PYTHONPATH or tests are run from a suitable root
from src.strainr.sequence import GenomicSequence, extract_kmers_from_sequence

# --- Fixtures ---


@pytest.fixture
def valid_sequence_id_fixture() -> str:  # Renamed for clarity
    return "test_seq_123"


@pytest.fixture
def valid_dna_bytes_fixture() -> bytes:  # Renamed for clarity
    return b"ACGTNACGT"  # Contains all valid characters including N


@pytest.fixture
def valid_genomic_sequence_fixture(
    valid_sequence_id_fixture: str, valid_dna_bytes_fixture: bytes
) -> GenomicSequence:  # Renamed
    """A valid GenomicSequence instance for testing."""
    return GenomicSequence(
        sequence_id=valid_sequence_id_fixture, sequence_data=valid_dna_bytes_fixture
    )


# --- Tests for GenomicSequence ---


# 1. Initialization Tests
def test_genomic_sequence_successful_init(
    valid_sequence_id_fixture: str, valid_dna_bytes_fixture: bytes
):
    seq = GenomicSequence(
        sequence_id=valid_sequence_id_fixture, sequence_data=valid_dna_bytes_fixture
    )
    assert seq.sequence_id == valid_sequence_id_fixture
    assert seq.sequence_data == valid_dna_bytes_fixture


def test_genomic_sequence_empty_data_error(valid_sequence_id_fixture: str):
    with pytest.raises(ValueError, match="Sequence data must be non-empty."):
        GenomicSequence(sequence_id=valid_sequence_id_fixture, sequence_data=b"")


def test_genomic_sequence_non_bytes_data_error(valid_sequence_id_fixture: str):
    with pytest.raises(TypeError, match="Sequence data must be bytes"):
        GenomicSequence(sequence_id=valid_sequence_id_fixture, sequence_data="ACGTN")  # type: ignore


def test_genomic_sequence_invalid_dna_chars_error(valid_sequence_id_fixture: str):
    with pytest.raises(ValueError, match="Sequence contains invalid nucleotides: {X}"):
        GenomicSequence(
            sequence_id=valid_sequence_id_fixture, sequence_data=b"ACGTX"
        )  # X is invalid
    with pytest.raises(
        ValueError, match="Sequence contains invalid nucleotides: {X, Y, Z}"
    ):  # Test sorting
        GenomicSequence(
            sequence_id=valid_sequence_id_fixture, sequence_data=b"ACGTXYZN"
        )


def test_genomic_sequence_non_ascii_decode_error(valid_sequence_id_fixture: str):
    # Bytes that are not valid ASCII (e.g., UTF-8 specific characters like 'ä')
    with pytest.raises(ValueError, match="Sequence data cannot be decoded as ASCII"):
        GenomicSequence(
            sequence_id=valid_sequence_id_fixture, sequence_data=b"ACGT\xc3\xa4N"
        )  # 'ä' in UTF-8


def test_genomic_sequence_frozen_instance_error(
    valid_genomic_sequence_fixture: GenomicSequence,
):
    with pytest.raises(dataclasses.FrozenInstanceError):
        valid_genomic_sequence_fixture.sequence_id = "new_id"  # type: ignore
    with pytest.raises(dataclasses.FrozenInstanceError):
        valid_genomic_sequence_fixture.sequence_data = b"NEWDATA"  # type: ignore


# 2. __hash__ Tests
def test_genomic_sequence_hash(
    valid_sequence_id_fixture: str, valid_dna_bytes_fixture: bytes
):
    seq1 = GenomicSequence(
        sequence_id=valid_sequence_id_fixture, sequence_data=valid_dna_bytes_fixture
    )
    seq2 = GenomicSequence(
        sequence_id=valid_sequence_id_fixture, sequence_data=valid_dna_bytes_fixture
    )
    seq3 = GenomicSequence(
        sequence_id="other_id", sequence_data=valid_dna_bytes_fixture
    )
    seq4 = GenomicSequence(
        sequence_id=valid_sequence_id_fixture, sequence_data=b"ACGTACGT"
    )

    assert isinstance(hash(seq1), int)
    assert hash(seq1) == hash(seq2)

    # Based on `mmh3.hash_bytes(self.sequence_data)`, hash depends only on data
    assert hash(seq1) == hash(seq3)

    assert hash(seq1) != hash(seq4)


# 3. __len__ Tests
def test_genomic_sequence_len(
    valid_genomic_sequence_fixture: GenomicSequence, valid_dna_bytes_fixture: bytes
):
    assert len(valid_genomic_sequence_fixture) == len(valid_dna_bytes_fixture)
    assert len(GenomicSequence("s", b"A")) == 1


# 4. __getitem__ Tests
def test_genomic_sequence_getitem(valid_genomic_sequence_fixture: GenomicSequence):
    # valid_dna_bytes_fixture = b"ACGTNACGT"
    assert valid_genomic_sequence_fixture[0] == "A"
    assert valid_genomic_sequence_fixture[2] == "G"
    assert valid_genomic_sequence_fixture[4] == "N"
    assert valid_genomic_sequence_fixture[8] == "T"
    assert isinstance(valid_genomic_sequence_fixture[0], str)


def test_genomic_sequence_getitem_index_error(
    valid_genomic_sequence_fixture: GenomicSequence,
):
    with pytest.raises(IndexError):
        _ = valid_genomic_sequence_fixture[len(valid_genomic_sequence_fixture)]
    with pytest.raises(IndexError):
        _ = valid_genomic_sequence_fixture[-(len(valid_genomic_sequence_fixture) + 1)]


# 5. __str__ Tests
def test_genomic_sequence_str(
    valid_genomic_sequence_fixture: GenomicSequence, valid_dna_bytes_fixture: bytes
):
    assert str(valid_genomic_sequence_fixture) == valid_dna_bytes_fixture.decode(
        "ascii"
    )


# 6. is_valid_dna Tests
def test_genomic_sequence_is_valid_dna(valid_genomic_sequence_fixture: GenomicSequence):
    # A successfully created instance should always be valid due to __post_init__ validation
    assert valid_genomic_sequence_fixture.is_valid_dna() is True

    # The False path of is_valid_dna is implicitly tested by the initialization error tests,
    # as is_valid_dna internally calls _validate_sequence_data.


# --- Tests for extract_kmers_from_sequence ---


def test_extract_kmers_successful():
    seq_data = b"ATCGATCG"
    gs = GenomicSequence(sequence_id="test_extract1", sequence_data=seq_data)
    kmers = extract_kmers_from_sequence(gs, kmer_length=3)
    expected = [b"ATC", b"TCG", b"CGA", b"GAT", b"ATC", b"TCG"]
    assert kmers == expected


def test_extract_kmers_kmer_len_equals_seq_len():
    seq_data = b"ACGT"
    gs = GenomicSequence(sequence_id="test_extract2", sequence_data=seq_data)
    kmers = extract_kmers_from_sequence(gs, kmer_length=4)
    assert kmers == [b"ACGT"]


def test_extract_kmers_short_sequence_k1():
    seq_data = b"AG"
    gs = GenomicSequence(sequence_id="test_extract3", sequence_data=seq_data)
    kmers = extract_kmers_from_sequence(gs, kmer_length=1)
    assert kmers == [b"A", b"G"]


# Error Handling for extract_kmers_from_sequence
def test_extract_kmers_kmer_length_positive_error():
    gs = GenomicSequence(sequence_id="test_err_klen_pos", sequence_data=b"ACGT")
    with pytest.raises(ValueError, match="K-mer length must be positive, got 0"):
        extract_kmers_from_sequence(gs, kmer_length=0)
    with pytest.raises(ValueError, match="K-mer length must be positive, got -1"):
        extract_kmers_from_sequence(gs, kmer_length=-1)


def test_extract_kmers_kmer_length_exceeds_seq_len_error():
    seq_data = b"ACG"
    gs = GenomicSequence(sequence_id="test_err_klen_exceed", sequence_data=seq_data)
    with pytest.raises(
        ValueError, match="K-mer length \\(4\\) cannot exceed sequence length \\(3\\)."
    ):
        extract_kmers_from_sequence(gs, kmer_length=4)


# GenomicSequence itself prevents empty sequence data, so no direct test here for extract_kmers
# with an empty GenomicSequence.sequence_data, as it wouldn't be constructible.
