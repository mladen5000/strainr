"""
K-mer database management for strain classification.
"""

import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        database_path (pathlib.Path): Absolute path to the database file.
        kmer_length (int): The length of k-mers in the database. Determined from data
                           if not provided, or validated if provided.
        kmer_to_counts_map (Dict[bytes, np.ndarray]): A dictionary mapping each k-mer (bytes)
                                             to its count vector (np.ndarray[np.uint8]).
        strain_names (List[str]): A list of strain names present in the database.
        num_strains (int): The number of strains in the database.
        num_kmers (int): The number of unique k-mers successfully loaded into the database.
    """

    def __init__(
        self,
        database_path: Union[str, pathlib.Path],
        expected_kmer_length: Optional[int] = None,
    ) -> None:
        """
        Initializes and loads the StrainKmerDatabase from a file.

        Args:
            database_path: Path to the Parquet file containing the k-mer database.
                               The DataFrame stored in Parquet should have k-mers (strings or bytes)
                               as its index and strain names (strings) as its columns.
                               Cell values should be numeric and convertible to `np.uint8`.
            expected_kmer_length: Optional. If provided, this length is enforced. K-mers in the
                                  database must match this length. If None, the k-mer length is
                                  inferred from the first k-mer in the database and then
                                  enforced for all other k-mers.

        Raises:
            FileNotFoundError: If the `database_path` does not exist or is not a file.
            ValueError: If the database is empty, if k-mers have inconsistent lengths,
                        if `expected_kmer_length` is provided and does not match the k-mer
                        lengths in the file, or if other data validation checks fail.
            TypeError: If the data in the DataFrame is not of the expected type (e.g., k-mer
                       index contains types other than str/bytes, or counts are not
                       convertible to uint8).
            RuntimeError: For lower-level issues during file reading (e.g., corrupted Parquet file),
                          often wrapping underlying exceptions like `IOError`, `ValueError`,
                          or specific PyArrow errors.
        """
        self.database_path = (
            Path(database_path).resolve().expanduser()
        )  # Use resolve for absolute path
        self.kmer_length = expected_kmer_length

        # Initialize attributes that will be set in _load_database or elsewhere
        self.strain_names: List[str] = []
        self.kmer_to_counts_map: KmerCountDict = {}  # Dict[bytes, CountVector]
        self.num_strains: int = 0
        self.num_kmers: int = 0

        self._load_database(expected_kmer_length)

        print(
            f"Successfully loaded database from {self.database_path}\n"
            f" - K-mer length: {self.kmer_length}\n"
            f" - Number of k-mers: {self.num_kmers}\n"
            f" - Number of strains: {self.num_strains}\n"
            f" - Strain names preview: {', '.join(self.strain_names[:5])}{'...' if len(self.strain_names) > 5 else ''}"
        )

    def __len__(self) -> int:
        return self.num_kmers

    def __contains__(self, kmer: bytes) -> bool:
        return kmer in self.kmer_to_counts_map

    def _load_database(self, expected_kmer_length: Optional[int]) -> None:
        """
            Internal method to load data from the Parquet database file.
        def _load_database(self, expected_kmer_length: Optional[int]) -> None:

            Internal method to load data from the Parquet database file.
        """
        print(f"Loading k-mer database from {self.database_path} (Parquet format)...")
        try:
            kmer_strain_df: pd.DataFrame = pd.read_parquet(self.database_path)

        except (IOError, ValueError, pd.errors.EmptyDataError) as e:
            kmer_strain_df: pd.DataFrame = pd.read_parquet(self.database_path)
        except (IOError, ValueError, pd.errors.EmptyDataError) as e:
            raise RuntimeError(
                f"Failed to read or process Parquet database from {self.database_path}: {e}"
            ) from e
        except FileNotFoundError:
            raise RuntimeError(
                f"Database file {self.database_path} vanished after initial check."
            ) from None
        except Exception as e:
            print(f"Database file {self.database_path} vanished after initial check.")
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while reading Parquet file {self.database_path}: {e}"
            ) from e

        if not isinstance(kmer_strain_df, pd.DataFrame):
            raise RuntimeError(
                f"Data loaded from {self.database_path} is not a pandas DataFrame (type: {type(kmer_strain_df)})."
            )

        if kmer_strain_df.empty:
            raise ValueError(f"Loaded database is empty: {self.database_path}")

        self.strain_names = list(kmer_strain_df.columns.astype(str))

        self.num_strains = len(self.strain_names)
        if self.num_strains == 0:
            raise ValueError("Database contains no strain information (no columns).")

        if kmer_strain_df.index.empty:
            raise ValueError("Database contains no k-mers (empty index).")
        if kmer_strain_df.empty:
            raise ValueError(f"Loaded database is empty: {self.database_path}")

        self.strain_names = list(kmer_strain_df.columns.astype(str))

        self.num_strains = len(self.strain_names)
        if self.num_strains == 0:
            raise ValueError("Database contains no strain information (no columns).")

        if kmer_strain_df.index.empty:
            raise ValueError("Database contains no k-mers (empty index).")

        if not kmer_strain_df.index.is_unique:
            print(
                f"Warning: K-mer index in {self.database_path} is not unique. Duplicates will be resolved by last occurrence when creating the lookup dictionary."
            )

        first_kmer_obj = kmer_strain_df.index[0]
        kmer_type_is_str: bool
        inferred_k_len: int

        if isinstance(first_kmer_obj, str):
            inferred_k_len = len(first_kmer_obj)
            kmer_type_is_str = True
        elif isinstance(first_kmer_obj, bytes):
            inferred_k_len = len(first_kmer_obj)
            kmer_type_is_str = False

        first_kmer_obj = kmer_strain_df.index[0]
        kmer_type_is_str: bool
        inferred_k_len: int

        if isinstance(first_kmer_obj, str):
            inferred_k_len = len(first_kmer_obj)
            kmer_type_is_str = True
        elif isinstance(first_kmer_obj, bytes):
            inferred_k_len = len(first_kmer_obj)
            kmer_type_is_str = False
        else:
            raise TypeError(
                f"Unsupported k-mer type in DataFrame index: {type(first_kmer_obj)}. Expected str or bytes."
            )

        if inferred_k_len == 0:
            raise ValueError(
                "First k-mer in database has zero length, which is invalid."
            )

        try:
            count_matrix = kmer_strain_df.to_numpy(dtype=np.uint8)
        except ValueError as e:
            raise TypeError(
                f"Could not convert database values to count matrix (np.uint8). "
                f"Ensure all values are numeric and within 0-255. Error: {e}"
            ) from e

        self.kmer_to_counts_map.clear()
        skipped_kmers_count = 0

        for i, kmer_obj in enumerate(kmer_strain_df.index):
            # Validate type and encode if needed
            if kmer_type_is_str:
                if not isinstance(kmer_obj, str):
                    print(
                        f"Warning: Inconsistent k-mer type at index {i}. Expected str, got {type(kmer_obj)}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
                kmer_bytes = None
                try:
                    kmer_bytes = kmer_obj.encode("utf-8")
                except UnicodeEncodeError as e:
                    print(
                        f"Warning: Failed to encode k-mer '{kmer_obj}' (index {i}) to UTF-8 bytes: {e}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
            else:
                if not isinstance(kmer_obj, bytes):
                    print(
                        f"Warning: Inconsistent k-mer type at index {i}. Expected bytes, got {type(kmer_obj)}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
                kmer_bytes = kmer_obj

            # Validate length
            try:
                if self.kmer_length == len(kmer_bytes):
                    continue
            except ValueError as e:
                raise ValueError(
                    f"Warning: K-mer '{kmer_obj}' (index {i}) has inconsistent length: {len(kmer_bytes)}. Expected {self.kmer_length}. Skipping."
                )

                skipped_kmers_count += 1
                continue

            self.kmer_to_counts_map[kmer_bytes] = count_matrix[i]

        self.num_kmers = len(self.kmer_to_counts_map)

        if (
            self.num_kmers == 0
            and not kmer_strain_df.index.empty
            and skipped_kmers_count == len(kmer_strain_df.index)
        ):
            raise ValueError(
                f"No k-mers were successfully loaded into the database from {self.database_path} "
                f"(all {skipped_kmers_count} k-mers from input were skipped). "
                "This is likely due to encoding issues or consistent length mismatches. Check warnings."
            )
        if skipped_kmers_count > 0:
            print(
                f"Warning: Skipped {skipped_kmers_count} k-mers during loading due to type, length, or encoding issues."
            )

        print(
            f"Successfully loaded database: {self.num_strains} strains, "
            f"{self.num_kmers} k-mers (k={self.kmer_length}). "
            f"Skipped {skipped_kmers_count} k-mers during loading."
            if skipped_kmers_count > 0
            else ""
        )

    def get_strain_counts_for_kmer(self, kmer: bytes) -> Optional[np.ndarray]:
        """
        Retrieves the strain count vector for a given k-mer.

        Args:
            kmer: The k-mer (bytes) to look up.

        Returns:
            A NumPy array representing the CountVector for the k-mer if found,
            otherwise None.
        """
        if not isinstance(kmer, bytes):
            return None
        return self.kmer_to_counts_map.get(
            kmer
        )  # A NumPy array (np.ndarray[np.uint8]) of counts for each strain if the k-mer is found, otherwise None.

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded k-mer database.

        Returns:
            A dictionary containing key database statistics:
                - "num_strains" (int): Number of strains in the database.
                - "num_kmers" (int): Number of unique k-mers loaded.
                - "kmer_length" (int): The validated length of k-mers in the database.
                - "database_path" (str): Absolute path to the database file.
                - "strain_names_preview" (List[str]): A preview of the first 5 strain names.
                - "total_strain_names" (int): Total count of strain names.
        """
        return {
            "num_strains": self.num_strains,
            "num_kmers": self.num_kmers,
            "kmer_length": self.kmer_length,
            "database_path": str(self.database_path),
            "strain_names_preview": self.strain_names[:5],
            "total_strain_names": self.num_strains,
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
            False otherwise. Returns False for non-str/bytes input.
        """

        if not isinstance(test_kmer, (str, bytes)):
            return False
        return len(test_kmer) == self.kmer_length
