"""
Sequence handling and k-mer extraction functionality.

CHANGES:
- Added comprehensive type hints (already mostly present)
- Improved error handling and validation (already mostly present)
- Better documentation (already mostly present)
- Renamed variables for clarity (already mostly good)
- Fixed potential encoding issues (already mostly good)
- Refined GenomicSequence.validate_sequence_data for stricter bytes input.
- Corrected example in extract_kmers_from_sequence docstring.
"""

from dataclasses import dataclass, field
from typing import List, Set # Added Set for type hinting
import mmh3


@dataclass(order=True, slots=True)
class GenomicSequence:
    """
    Immutable genomic sequence representation optimized for k-mer analysis.

    This class provides a memory-efficient, validated container for genomic sequences
    with built-in k-mer extraction capabilities.

    Attributes:
        sequence_id: Unique identifier for the sequence.
        sequence_data: Raw sequence as bytes for memory efficiency and to ensure
                       consistent handling of character encodings.
    
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

    def _validate_sequence_data(self) -> bytes:
        """
        Validate the byte sequence data.

        Ensures the sequence is non-empty and contains only valid DNA characters
        (A, C, G, T, N).

        Returns:
            The validated sequence as bytes.

        Raises:
            ValueError: If sequence is empty or contains invalid characters.
            TypeError: If sequence_data is not bytes (though type hint should prevent this).
        """
        if not isinstance(self.sequence_data, bytes):
            # This case should ideally be caught by type checking before instantiation
            # or handled by a factory method if str input is allowed.
            raise TypeError(
                f"Sequence data must be bytes, got {type(self.sequence_data).__name__}."
            )

        if len(self.sequence_data) == 0:
            raise ValueError("Sequence data must be non-empty.")

        # Validate DNA bases
        allowed_nucleotides: Set[str] = {"A", "C", "G", "T", "N"}
        
        # Decode for character set validation, assuming ASCII or compatible (e.g., UTF-8 subset)
        # If other encodings are possible, this might need adjustment.
        try:
            sequence_str = self.sequence_data.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError(
                "Sequence data cannot be decoded as ASCII. Ensure it contains valid DNA characters."
            ) from error

        found_nucleotides: Set[str] = set(sequence_str)

        invalid_bases: Set[str] = found_nucleotides.difference(allowed_nucleotides)
        if invalid_bases:
            # Sort for consistent error messages
            sorted_invalid_bases = ", ".join(sorted(list(invalid_bases)))
            raise ValueError(
                f"Sequence contains invalid nucleotides: {{{sorted_invalid_bases}}}. "
                f"Only DNA characters {allowed_nucleotides} are allowed."
            )

        return self.sequence_data

    def __post_init__(self) -> None:
        """Validate sequence data after initialization."""
        # The validation method returns the data, but since it operates on
        # self.sequence_data directly and we expect bytes from the start,
        # direct assignment isn't strictly needed here but is kept for consistency
        # with the original structure.
        self.sequence_data = self._validate_sequence_data()

    def __hash__(self) -> int:
        """Generate hash using MurmurHash3 for consistency and performance."""
        return mmh3.hash_bytes(self.sequence_data)

    def __len__(self) -> int:
        """Return the length of the sequence."""
        return len(self.sequence_data)

    def __getitem__(self, index: int) -> str:
        """Get nucleotide at a specific position as a string.

        Args:
            index: The integer index of the nucleotide.

        Returns:
            The nucleotide at the specified position, decoded as an ASCII character.
        """
        return self.sequence_data.decode("ascii")[index]

    def __str__(self) -> str:
        """Return the sequence as an ASCII string."""
        return self.sequence_data.decode(encoding="ascii")

    def is_valid_dna(self) -> bool:
        """Check if the sequence contains only valid DNA nucleotides (A, C, G, T, N).

        Returns:
            True if the sequence is valid DNA, False otherwise.
        """
        try:
            self._validate_sequence_data()
            return True
        except (ValueError, TypeError):
            return False


def extract_kmers_from_sequence(
    sequence: GenomicSequence, kmer_length: int = 31
) -> List[bytes]:
    """
    Extract k-mers from a genomic sequence using a memory-efficient approach.

    This function utilizes a memory view for performance during k-mer extraction.

    Args:
        sequence: The `GenomicSequence` object containing the DNA sequence.
        kmer_length: The length of k-mers to extract. Defaults to 31.

    Returns:
        A list of k-mers, where each k-mer is represented as a bytes object.

    Raises:
        ValueError: If the specified `kmer_length` is greater than the
                    length of the sequence.

    Example:
        >>> seq_data = b'ATCGATCG'
        >>> gen_seq = GenomicSequence(sequence_id='test_seq', sequence_data=seq_data)
        >>> kmers = extract_kmers_from_sequence(gen_seq, kmer_length=3)
        >>> kmers
        [b'ATC', b'TCG', b'CGA', b'GAT', b'ATC', b'TCG']
    """
    if kmer_length <= 0:
        raise ValueError(f"K-mer length must be positive, got {kmer_length}.")
    if kmer_length > len(sequence):
        raise ValueError(
            f"K-mer length ({kmer_length}) cannot exceed sequence length ({len(sequence)})."
        )

    # The sequence_data is already bytes due to GenomicSequence validation
    raw_sequence_data = sequence.sequence_data
    max_start_position = len(raw_sequence_data) - kmer_length + 1

    kmer_list: List[bytes] = []
    # Use memory view for efficient k-mer extraction
    with memoryview(raw_sequence_data) as sequence_view:
        for position in range(max_start_position):
            kmer_list.append(sequence_view[position : position + kmer_length].tobytes())

    return kmer_list


# Alternative optimized implementation (commented out as requested in original file)
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
