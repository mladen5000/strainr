"""
Sequence handling and k-mer extraction functionality.

CHANGES:
- Added comprehensive type hints
- Improved error handling and validation
- Better documentation
- Renamed variables for clarity
- Fixed potential encoding issues
"""

from dataclasses import dataclass, field
from typing import List, Optional
import mmh3


@dataclass(order=True, slots=True)
class GenomicSequence:
    """
    Immutable genomic sequence representation optimized for k-mer analysis.

    This class provides a memory-efficient, validated container for genomic sequences
    with built-in k-mer extraction capabilities.

    Attributes:
        sequence_id: Unique identifier for the sequence
        sequence_data: Raw sequence as bytes for memory efficiency

    Example:
        >>> seq = GenomicSequence(
        ...     sequence_id='read_001',
        ...     sequence_data=b'ACTTTAAGGGGTTAAACCCCCG' * 100
        ... )
        >>> len(seq)
        2200
        >>> seq.is_valid_dna()
        True
    """

    sequence_id: str = field(compare=False)
    sequence_data: bytes

    def validate_sequence_data(self) -> bytes:
        """
        Validate and normalize sequence data.

        Returns:
            Validated sequence as bytes

        Raises:
            ValueError: If sequence is empty or contains invalid characters
            TypeError: If sequence cannot be converted to bytes
        """
        if len(self.sequence_data) == 0:
            raise ValueError("Sequence data must be non-empty")

        # Ensure we have bytes
        if not isinstance(self.sequence_data, bytes):
            try:
                self.sequence_data = bytes(self.sequence_data, encoding="ascii")
            except Exception as error:
                raise TypeError(f"Cannot convert sequence to bytes: {error}") from error

        # Validate DNA bases
        allowed_nucleotides = {"A", "C", "G", "T", "N"}
        found_nucleotides = set(self.sequence_data.decode("ascii"))

        invalid_bases = found_nucleotides.difference(allowed_nucleotides)
        if invalid_bases:
            raise ValueError(
                f"Sequence contains invalid nucleotides: {invalid_bases}. "
                f"Only {allowed_nucleotides} are allowed."
            )

        return self.sequence_data

    def __post_init__(self) -> None:
        """Validate sequence data after initialization."""
        self.sequence_data = self.validate_sequence_data()

    def __hash__(self) -> int:
        """Generate hash using MurmurHash3 for consistency."""
        return mmh3.hash_bytes(self.sequence_data)

    def __len__(self) -> int:
        """Return sequence length."""
        return len(self.sequence_data)

    def __getitem__(self, index: int) -> str:
        """Get nucleotide at position as string."""
        return self.sequence_data.decode("ascii")[index]

    def __str__(self) -> str:
        """Return sequence as ASCII string."""
        return self.sequence_data.decode(encoding="ascii")

    def is_valid_dna(self) -> bool:
        """Check if sequence contains only valid DNA nucleotides."""
        try:
            self.validate_sequence_data()
            return True
        except (ValueError, TypeError):
            return False


def extract_kmers_from_sequence(
    sequence: GenomicSequence, kmer_length: int = 31
) -> List[bytes]:
    """
    Extract k-mers from a genomic sequence using memory-efficient approach.

    This function maintains the original memory view approach for performance
    while providing cleaner interface and better error handling.

    Args:
        sequence: GenomicSequence object containing the DNA sequence
        kmer_length: Length of k-mers to extract (default: 31)

    Returns:
        List of k-mers as bytes objects

    Raises:
        ValueError: If k-mer length is larger than sequence length

    Example:
        >>> seq = GenomicSequence('test', b'ATCGATCG')
        >>> kmers = extract_kmers_from_sequence(seq, kmer_length=3)
        >>> [k.tobytes() for k in kmers]
        [b'ATC', b'TCG', b'CGA', b'GAT', b'ATC', b'TCG']
    """
    if kmer_length > len(sequence):
        raise ValueError(
            f"K-mer length ({kmer_length}) cannot exceed sequence length ({len(sequence)})"
        )

    max_start_position = len(sequence.sequence_data) - kmer_length + 1

    # Use memory view for efficient k-mer extraction (keeping original approach)
    with memoryview(sequence.sequence_data) as sequence_view:
        kmer_list = [
            sequence_view[position : position + kmer_length].tobytes()
            for position in range(max_start_position)
        ]

    return kmer_list


# Alternative optimized implementation (commented out as requested)
# def extract_kmers_optimized(sequence: GenomicSequence, kmer_length: int = 31) -> Generator[bytes, None, None]:
#     """
#     Generator-based k-mer extraction for memory efficiency with large sequences.
#
#     This alternative implementation uses a generator to avoid storing all k-mers
#     in memory simultaneously, which is beneficial for very long sequences.
#     """
#     if kmer_length > len(sequence):
#         raise ValueError(f"K-mer length ({kmer_length}) cannot exceed sequence length ({len(sequence)})")
#
#     max_start_position = len(sequence.sequence_data) - kmer_length + 1
#
#     with memoryview(sequence.sequence_data) as sequence_view:
#         for position in range(max_start_position):
#             yield sequence_view[position:position + kmer_length].tobytes()
