"""
K-mer database management for strain classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any  # Tuple removed

import numpy as np
import pandas as pd  # For pd.errors.EmptyDataError

from strainr.genomic_types import (  # KmerCountDict is Dict[bytes, CountVector]
    CountVector,
    KmerCountDict,
)


class StrainKmerDatabase:
    """
    K-mer database for strain classification.

    This class manages a database of k-mer frequencies across multiple strains,
    enabling efficient lookup of strain-specific k-mer signatures.

    Attributes:
        database_path: Path to the Parquet database file
        kmer_length: Length of k-mers in the database
        strain_names: List of strain identifiers
        kmer_lookup_dict: Dictionary mapping k-mers to strain frequency vectors
        num_strains: Number of strains in database
        num_kmers: Number of unique k-mers in database
    """

    def __init__(self, database_path: Union[str, Path], kmer_length: int = 31) -> None:
        """
        Initialize strain database from a Parquet file.

        Args:
            database_path: Path to the Parquet file. The DataFrame stored in Parquet
                           is expected to have k-mers (typically strings or bytes)
                           as its index and strain names (strings) as its columns.
                           Cell values should be k-mer counts (numeric, convertible to np.uint8).
            kmer_length: Expected length of k-mers. This is validated against the
                         first k-mer found in the database. If a mismatch occurs,
                         a warning is printed, and the database's k-mer length is used.

        Raises:
            FileNotFoundError: If the database_path does not point to a valid file.
            RuntimeError: If loading or processing the database fails due to issues
                          like file corruption, incorrect format, empty data, or unexpected data types.
            ValueError: If the loaded database is empty or if k-mer length validation
                        reveals issues (though currently it warns and updates self.kmer_length).

        Example:
            >>> # db = StrainKmerDatabase("path/to/your/database.parquet", kmer_length=31)
            >>> # print(f"Loaded {db.num_strains} strains with {db.num_kmers} k-mers of length {db.kmer_length}")
        """
        self.database_path = (
            Path(database_path).resolve().expanduser()
        )  # Use resolve for absolute path
        self.kmer_length = kmer_length

        # Initialize attributes that will be set in _load_database or elsewhere
        self.strain_names: List[str] = []
        self.kmer_lookup_dict: KmerCountDict = {}  # Dict[bytes, CountVector]
        self.num_strains: int = 0
        self.num_kmers: int = 0

        self._validate_database_file()
        self._load_database()

    def _validate_database_file(self) -> None:
        """Validate that database_path points to an existing file."""
        if not self.database_path.is_file():  # Check if it's a file, not just exists
            raise FileNotFoundError(
                f"Database file not found or is not a file: {self.database_path}"
            )

    def _load_database(self) -> None:
        """Load and validate the k-mer database from Parquet file.

        Raises:
            RuntimeError: For errors during file reading or DataFrame processing.
            ValueError: If the database DataFrame is empty or structure is invalid.
        """
        print(f"Loading k-mer database from {self.database_path} (Parquet format)...")

        try:
            # It's common for k-mers to be strings in DataFrames from bioinformatics tools
            kmer_frequency_dataframe: pd.DataFrame = pd.read_parquet(self.database_path)
        except (
            FileNotFoundError
        ):  # Should be caught by _validate_database_file, but good practice
            raise RuntimeError(
                f"Database file disappeared after validation: {self.database_path}"
            )
        except (IOError, ValueError, pd.errors.EmptyDataError) as e:
            raise RuntimeError(
                f"Failed to read or process Parquet database from {self.database_path}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Parquet database from {self.database_path} due to an unexpected error: {e}"
            ) from e

        if not isinstance(kmer_frequency_dataframe, pd.DataFrame):
            raise RuntimeError(
                f"Loaded database from {self.database_path} is not a pandas DataFrame. "
                f"Found type: {type(kmer_frequency_dataframe)}."
            )

        if kmer_frequency_dataframe.empty:
            raise ValueError(f"Database DataFrame from {self.database_path} is empty.")

        # Extract database components
        self.strain_names = list(kmer_frequency_dataframe.columns.astype(str))

        if not kmer_frequency_dataframe.index.is_unique:
            print(
                f"Warning: K-mer index in {self.database_path} is not unique. Counts for duplicate k-mers might be based on their last occurrence in the input DataFrame during conversion to dictionary."
            )

        kmer_sequences_from_index = kmer_frequency_dataframe.index

        if len(kmer_sequences_from_index) == 0:
            raise ValueError(
                f"Database DataFrame from {self.database_path} has no k-mers (empty index)."
            )

        first_kmer_in_index = kmer_sequences_from_index[0]
        kmer_needs_encoding: bool
        if isinstance(first_kmer_in_index, str):
            actual_kmer_length = len(first_kmer_in_index)
            kmer_needs_encoding = True
        elif isinstance(first_kmer_in_index, bytes):
            actual_kmer_length = len(first_kmer_in_index)
            kmer_needs_encoding = False
        else:
            raise ValueError(
                f"Unsupported k-mer type in DataFrame index: {type(first_kmer_in_index)}. "
                "Expected str or bytes."
            )

        if actual_kmer_length == 0:
            raise ValueError(
                f"First k-mer in database '{first_kmer_in_index}' has zero length. This is invalid."
            )

        if actual_kmer_length != self.kmer_length:
            print(
                f"Warning: Initial expected k-mer length was {self.kmer_length} for database {self.database_path}, "
                f"but found {actual_kmer_length} (based on first k-mer: '{first_kmer_in_index}'). "
                f"Using actual length from database: {actual_kmer_length}."
            )
            self.kmer_length = actual_kmer_length

        for idx, kmer_val in enumerate(kmer_sequences_from_index):
            if len(kmer_val) != self.kmer_length:
                raise ValueError(
                    f"Inconsistent k-mer length found in database {self.database_path} at index position {idx}. "
                    f"Expected {self.kmer_length}, found k-mer '{kmer_val}' (type: {type(kmer_val)}) with length {len(kmer_val)}."
                )
            if kmer_needs_encoding and not isinstance(kmer_val, str):
                raise ValueError(
                    f"K-mer at index position {idx} ('{kmer_val}') is type {type(kmer_val)}, expected str based on first k-mer."
                )
            if not kmer_needs_encoding and not isinstance(kmer_val, bytes):
                raise ValueError(
                    f"K-mer at index position {idx} ('{kmer_val}') is type {type(kmer_val)}, expected bytes based on first k-mer."
                )

        try:
            frequency_matrix = kmer_frequency_dataframe.to_numpy(dtype=np.uint8)
        except ValueError as e:
            raise RuntimeError(
                f"Could not convert DataFrame values to np.uint8 for {self.database_path}. Ensure all counts are valid integers within 0-255. Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error converting DataFrame to NumPy array for {self.database_path}: {e}"
            ) from e

        self.kmer_lookup_dict.clear()
        skipped_kmers_count = 0
        for i, kmer_in_idx in enumerate(kmer_sequences_from_index):
            kmer_bytes: bytes
            if kmer_needs_encoding:
                try:
                    kmer_bytes = str(kmer_in_idx).encode("utf-8")
                except UnicodeEncodeError as e:
                    print(
                        f"Warning: Failed to encode k-mer '{kmer_in_idx}' (index {i}) to UTF-8 bytes: {e}. Skipping this k-mer."
                    )
                    skipped_kmers_count += 1
                    continue
            else:
                kmer_bytes = bytes(kmer_in_idx)

            if len(kmer_bytes) != self.kmer_length:
                print(
                    f"Warning: K-mer '{kmer_in_idx}' (index {i}) resulted in byte length {len(kmer_bytes)} after encoding/casting, expected {self.kmer_length}. Skipping this k-mer."
                )
                skipped_kmers_count += 1
                continue
            self.kmer_lookup_dict[kmer_bytes] = frequency_matrix[i]

        self.num_strains = len(self.strain_names)
        self.num_kmers = len(self.kmer_lookup_dict)

        if self.num_kmers == 0 and len(kmer_sequences_from_index) > 0:
            raise ValueError(
                f"No k-mers were successfully loaded into the lookup dictionary from {self.database_path} "
                f"(skipped {skipped_kmers_count} out of {len(kmer_sequences_from_index)}). "
                "This might be due to encoding issues or length mismatches after encoding. Check warnings."
            )

        print(
            f"Successfully loaded database: {self.num_strains} strains, "
            f"{self.num_kmers} k-mers (k={self.kmer_length}). "
            f"Skipped {skipped_kmers_count} k-mers during loading."
            if skipped_kmers_count > 0
            else ""
        )

    def lookup_kmer(self, kmer_sequence: bytes) -> Optional[CountVector]:
        """
        Look up strain frequency vector for a given k-mer.

        Args:
            kmer_sequence: K-mer sequence as bytes

        Returns:
            A NumPy array representing the CountVector for the k-mer if found,
            otherwise None.
        """
        if not isinstance(kmer_sequence, bytes):
            pass
        return self.kmer_lookup_dict.get(kmer_sequence)

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded k-mer database.

        Returns:
            A dictionary containing key database statistics:
                - "num_strains" (int): Number of strains in the database.
                - "num_kmers" (int): Number of unique k-mers loaded into the lookup dictionary.
                - "kmer_length" (int): The validated length of k-mers in the database.
                - "database_path" (str): Absolute path to the database file.
                - "strain_names_preview" (List[str]): A preview of the first 5 strain names.
                - "total_strain_names" (int): Total number of strains (should match "num_strains").
        """
        return {
            "num_strains": self.num_strains,
            "num_kmers": self.num_kmers,
            "kmer_length": self.kmer_length,
            "database_path": str(self.database_path.resolve()),
            "strain_names_preview": self.strain_names[:5],
            "total_strain_names": len(self.strain_names),
        }

    def validate_kmer_length(self, test_kmer: Union[str, bytes]) -> bool:
        """
        Validates if a given k-mer (string or bytes) matches the database's k-mer length.

        If the input `test_kmer` is a string, its direct length is checked.
        If it's bytes, its byte length is checked. This method does not perform
        encoding; it assumes the provided form (str or bytes) is what needs checking.
        For checking against the database's internal byte representation of k-mers,
        ensure `test_kmer` is passed as bytes or handle encoding prior to calling.

        Args:
            test_kmer: The k-mer (string or bytes) to validate.

        Returns:
            True if the length of `test_kmer` matches `self.kmer_length`,
            False otherwise.
        """
        if not isinstance(test_kmer, (str, bytes)):
            return False
        return len(test_kmer) == self.kmer_length
