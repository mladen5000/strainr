"""
Sequence handling and k-mer extraction functionality.
"""

from dataclasses import dataclass, field
from typing import List, Set

import mmh3

# Define allowed DNA bytes at the module level for clarity and reuse
VALID_DNA_BYTES: frozenset[int] = frozenset(b"ACGTN")


@dataclass(order=True, slots=True, frozen=True)
class GenomicSequence:
    """
    Immutable genomic sequence representation optimized for k-mer analysis.

    This class provides a memory-efficient, validated container for genomic sequences.
    Sequence data is stored as bytes and validated to contain only bytes corresponding
    to 'A', 'C', 'G', 'T', or 'N'.

    Attributes:
        sequence_id: Unique identifier for the sequence.
        sequence_data: Raw sequence as bytes, validated to contain only valid DNA bytes.


    Example:
        >>> seq = GenomicSequence(sequence_id='read_001', sequence_data=b'ACGTN')
        >>> len(seq)
        5
        >>> seq[0] # Accesses as character
        'A'
        >>> str(seq)
        'ACGTN'
    """

    sequence_id: str = field(compare=False)
    sequence_data: bytes

    def _validate_sequence_data(self) -> bytes:
        """
        Validate the byte sequence data.

        Ensures the sequence is non-empty and contains only valid DNA bytes
        (bytes corresponding to ASCII 'A', 'C', 'G', 'T', 'N').

        Returns:
            The validated sequence as bytes.

        Raises:
            TypeError: If sequence_data is not bytes.
            ValueError: If sequence is empty or contains invalid byte values.
        """
        if not isinstance(self.sequence_data, bytes):
            raise TypeError(
                f"Sequence data must be bytes, got {type(self.sequence_data).__name__}."
            )

        if not self.sequence_data:  # Check for empty bytes
            raise ValueError("Sequence data must be non-empty.")

        # Direct byte-level validation
        invalid_bytes_found: Set[int] = set()
        for byte_val in self.sequence_data:
            # Convert byte_val to uppercase equivalent for case-insensitivity if desired,
            # but VALID_DNA_BYTES currently only contains uppercase.
            # For strict ACGTN (uppercase only):
            if byte_val not in VALID_DNA_BYTES:
                invalid_bytes_found.add(byte_val)

        if invalid_bytes_found:
            # Try to represent invalid bytes as characters if printable, else as numeric values
            # for a clearer error message.
            invalid_chars_repr = []
            for b_val in sorted(list(invalid_bytes_found)):
                try:
                    char = bytes([b_val]).decode("ascii")
                    if char.isprintable():  # Check if it's a printable character
                        invalid_chars_repr.append(f"'{char}' (byte: {b_val})")
                    else:
                        invalid_chars_repr.append(f"byte {b_val}")
                except UnicodeDecodeError:
                    invalid_chars_repr.append(f"byte {b_val}")

            allowed_chars_str = ", ".join(
                f"'{chr(b)}'" for b in sorted(list(VALID_DNA_BYTES))
            )

            raise ValueError(
                f"Sequence contains invalid DNA bytes: {{{', '.join(invalid_chars_repr)}}}. "
                f"Allowed bytes correspond to characters: {allowed_chars_str}."
            )

        return self.sequence_data

    def __post_init__(self) -> None:
        """Validate sequence data after initialization."""
        # Re-assigning to self.sequence_data is not strictly necessary due to
        # frozen=True (original object won't change), but _validate_sequence_data
        # must be called. If it were not frozen, this would be crucial.
        # For frozen dataclasses, __post_init__ can't modify fields directly
        # if they are part of the hash/eq. However, validation that raises errors
        # is the primary goal here.
        # The direct assignment `self.sequence_data = self._validate_sequence_data()`
        # would fail for frozen=True. Instead, just call for validation.
        self._validate_sequence_data()

    def __hash__(self) -> int:
        """Generate hash using MurmurHash3 for consistency and performance."""
        hash_result = mmh3.hash_bytes(self.sequence_data)
        if isinstance(hash_result, bytes):
            # If mmh3.hash_bytes unexpectedly returns bytes, hash these bytes.
            # This ensures we return an int, though it's an extra hash.
            return hash(hash_result)
        # Otherwise, assume it's already an int (or int-like, e.g. np.int32)
        return int(hash_result)  # Cast to ensure it's a Python int

    def __len__(self) -> int:
        """Return the length of the sequence."""
        return len(self.sequence_data)

    def __getitem__(self, index: int) -> str:
        """Get nucleotide at a specific position as a string character.

        Args:
            index: The integer index of the nucleotide byte.

        Returns:
            The nucleotide character at the specified position, decoded as ASCII.

        Raises:
            IndexError: If the index is out of bounds.
            UnicodeDecodeError: If the byte at the index is not valid ASCII
                                (should not happen if validation passed ensuring ASCII chars).
        """
        return bytes([self.sequence_data[index]]).decode("ascii")

    def __str__(self) -> str:
        """Return the sequence as an ASCII string."""
        # Validation ensures all bytes correspond to 'A', 'C', 'G', 'T', 'N',
        # which are ASCII characters.
        return self.sequence_data.decode(encoding="ascii")

    def is_valid_dna(self) -> bool:
        """
        Checks if the sequence contains only valid DNA bytes (A, C, G, T, N).

        Note: For a successfully initialized GenomicSequence object (due to `frozen=True`
        and `__post_init__` validation), this method will always return True.
        It primarily serves as a way to re-trigger validation logic if needed,
        or for consistency if the immutability constraint were different.

        Returns:
            True if the sequence is valid DNA, False otherwise (though False is
            unreachable for successfully constructed frozen instances).
        """
        try:
            self._validate_sequence_data()  # Re-run validation
            return True
        except (ValueError, TypeError):  # Catch errors from _validate_sequence_data
            return False


def extract_kmers_from_sequence(
    sequence: GenomicSequence, kmer_length: int = 31
) -> List[bytes]:
    """
    Extract k-mers from a genomic sequence using a memory-efficient approach.

    This function utilizes a memory view for performance during k-mer extraction.

    Args:
        sequence: The `GenomicSequence` object containing the DNA sequence.
                  Its `sequence_data` attribute (bytes) is used.
        kmer_length: The length of k-mers to extract. Defaults to 31.

    Returns:
        A list of k-mers, where each k-mer is represented as a bytes object.

    Raises:
        TypeError: If `sequence` is not a `GenomicSequence` instance.
        ValueError: If `kmer_length` is not positive, or if `kmer_length`
                    is greater than the length of the sequence.
    """
    if not isinstance(sequence, GenomicSequence):
        raise TypeError(
            f"Input 'sequence' must be a GenomicSequence object, got {type(sequence)}."
        )
    if not isinstance(kmer_length, int):
        raise TypeError(f"kmer_length must be an integer, got {type(kmer_length)}.")
    if kmer_length <= 0:
        raise ValueError(f"kmer_length must be positive, got {kmer_length}.")

    sequence_len = len(
        sequence.sequence_data
    )  # Use len(sequence.sequence_data) for clarity
    if kmer_length > sequence_len:
        raise ValueError(
            f"K-mer length ({kmer_length}) cannot exceed sequence length ({sequence_len})."
        )

    raw_sequence_data = sequence.sequence_data

    kmer_list: List[bytes] = []
    # Using memoryview for efficient slicing of bytes
    # The number of k-mers is sequence_len - kmer_length + 1
    num_kmers = sequence_len - kmer_length + 1

    # Pre-allocate list if performance is critical for very many k-mers,
    # but simple append is usually fine and readable.
    # kmer_list = [None] * num_kmers # type: ignore

    sequence_view = memoryview(raw_sequence_data)
    for i in range(num_kmers):
        kmer_list.append(sequence_view[i : i + kmer_length].tobytes())
        # kmer_list[i] = sequence_view[i : i + kmer_length].tobytes() # If pre-allocated

    return kmer_list


# Alternative optimized implementation (commented out as requested in original file)
# def extract_kmers_optimized(sequence: GenomicSequence, kmer_length: int = 31) -> Generator[bytes, None, None]:
#     """
#     Generator-based k-mer extraction for memory efficiency with large sequences.
#
#     This alternative implementation uses a generator to avoid storing all k-mers
#     in memory simultaneously, which is beneficial for very long sequences.
#     """
#     if not isinstance(sequence, GenomicSequence):
#         raise TypeError(f"Input 'sequence' must be a GenomicSequence object, got {type(sequence)}.")
#     if not isinstance(kmer_length, int):
#         raise TypeError(f"kmer_length must be an integer, got {type(kmer_length)}.")
#     if kmer_length <= 0:
#         raise ValueError(f"K-mer length must be positive, got {kmer_length}.")
#
#     sequence_len = len(sequence.sequence_data)
#     if kmer_length > sequence_len:
#         raise ValueError(f"K-mer length ({kmer_length}) cannot exceed sequence length ({sequence_len})")
#
#     num_kmers = sequence_len - kmer_length + 1
#     sequence_view = memoryview(sequence.sequence_data)
#
#     for i in range(num_kmers):
#         yield sequence_view[i:i + kmer_length].tobytes()
