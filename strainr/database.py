"""
K-mer database management for strain classification.
"""

import pathlib
import logging # Added for logging
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq # Added for reading schema

from .exceptions import (
    DatabaseNotFoundError,
    DatabaseLoadError,
    DatabaseCorruptedError
)

logger = logging.getLogger(__name__) # Added logger


class StrainKmerDatabase:  # Renamed from StrainKmerDb
    """
    Represents a database of k-mers and their corresponding strain frequency vectors.

    This class loads a k-mer database from a Parquet file. The DataFrame stored in Parquet
    is expected to have k-mers as its index (typically strings or bytes) and strain names
    as its columns. The values should be counts or frequencies (convertible to uint8).

    Attributes:
        database_filepath (pathlib.Path): Absolute path to the database file.
        kmer_length (int): The length of k-mers in the database. Determined from data
                           if not provided, or validated if provided.
        kmer_to_counts_map (Dict[bytes, np.ndarray]): A dictionary mapping each k-mer (bytes)
                                             to its count vector (np.ndarray[np.uint8]).
        strain_names (List[str]): A list of strain names present in the database.
        num_strains (int): The number of strains in the database.
        num_kmers (int): The number of unique k-mers successfully loaded into the database.
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
        database_filepath: Union[str, pathlib.Path],
        expected_kmer_length: Optional[int] = None,
    ) -> None:
        """
        Initializes and loads the StrainKmerDatabase from a file.

        Args:
            database_filepath: Path to the Parquet file containing the k-mer database.
                               The DataFrame stored in Parquet should have k-mers (strings or bytes)
                               as its index and strain names (strings) as its columns.
                               Cell values should be numeric and convertible to `np.uint8`.
            expected_kmer_length: Optional. If provided, this length is enforced. K-mers in the
                                  database must match this length. If None, the k-mer length is
                                  inferred from the first k-mer in the database and then
                                  enforced for all other k-mers.

        Raises:
            FileNotFoundError: If the `database_filepath` does not exist or is not a file.
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
        self.database_path = Path(database_filepath).resolve().expanduser()
        self._expected_k_init: Optional[int] = expected_kmer_length # Store for later validation

        self.kmer_length: Optional[int] = None
        self.db_kmer_length: Optional[int] = None # Loaded from DB metadata
        self.db_skip_n_kmers: Optional[bool] = None # Loaded from DB metadata
        self.data_derived_kmer_length: Optional[int] = None # Inferred from actual data in Parquet

        self.kmer_to_counts_map: Dict[bytes, np.ndarray] = {}
        self.kmer_specificity_map: Dict[bytes, int] = {}  # Track how many strains each k-mer appears in
        self.strain_genome_lengths: Dict[str, int] = {}  # Track genome length per strain for normalization
        self.strain_names: List[str] = []
        self.num_strains: int = 0
        self.num_kmers: int = 0

        self._load_database()

        # Finalize k-mer length determination
        if self.db_kmer_length is not None: # Metadata is king
            self.kmer_length = self.db_kmer_length
            if self._expected_k_init is not None and self._expected_k_init != self.kmer_length:
                logger.warning(
                    f"Provided expected_kmer_length ({self._expected_k_init}) "
                    f"differs from k-mer length in database metadata ({self.kmer_length}). "
                    "Using k-mer length from database metadata as it's authoritative."
                )
        elif self.data_derived_kmer_length is not None: # No metadata, use data inference
            if self._expected_k_init is not None:
                if self._expected_k_init != self.data_derived_kmer_length:
                    raise ValueError(
                        f"Provided expected_kmer_length ({self._expected_k_init}) "
                        f"differs from k-mer length inferred from data ({self.data_derived_kmer_length}), "
                        "and no k-mer length metadata was found in the database."
                    )
                self.kmer_length = self._expected_k_init # User expectation matches data
            else:
                self.kmer_length = self.data_derived_kmer_length # No user expectation, use inferred
        else:
            # This case should ideally not be reached if _load_database ensures data_derived_kmer_length is set
            raise ValueError("K-mer length could not be determined. Database might be empty or corrupt.")

        if self.kmer_length is None or self.kmer_length == 0:
             raise ValueError("Final k-mer length is invalid (None or 0).")

        logger.info(
            f"Successfully loaded database from {self.database_path}\n"
            f" - K-mer length (final): {self.kmer_length}\n"
            f" - DB k-mer length (metadata): {self.db_kmer_length}\n"
            f" - DB skip_n_kmers (metadata): {self.db_skip_n_kmers}\n"
            f" - Number of k-mers: {self.num_kmers}\n"
            f" - Number of strains: {self.num_strains}\n"
            f" - Strain names preview: {', '.join(self.strain_names[:5])}{'...' if len(self.strain_names) > 5 else ''}"
        )

    def __len__(self) -> int:
        return self.num_kmers

    def __contains__(self, kmer: bytes) -> bool:
        return kmer in self.kmer_to_counts_map

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

    def cleanup(self):
        """
        Release resources and clear memory.

        Call this method when done with the database to free memory.
        Especially important for large databases in long-running processes.
        """
        logger.info(f"Cleaning up database resources for {self.database_path.name}")
        self.kmer_to_counts_map.clear()
        self.kmer_specificity_map.clear()
        self.strain_genome_lengths.clear()
        self.strain_names.clear()
        self.num_kmers = 0
        self.num_strains = 0

    def get_memory_usage(self) -> dict:
        """
        Estimate memory usage of database components.

        Returns:
            Dictionary with memory usage estimates in MB
        """
        import sys

        kmer_map_size = sys.getsizeof(self.kmer_to_counts_map)
        for k, v in self.kmer_to_counts_map.items():
            kmer_map_size += sys.getsizeof(k) + v.nbytes

        spec_map_size = sys.getsizeof(self.kmer_specificity_map)
        for k, v in self.kmer_specificity_map.items():
            spec_map_size += sys.getsizeof(k) + sys.getsizeof(v)

        return {
            'kmer_map_mb': kmer_map_size / (1024 * 1024),
            'specificity_map_mb': spec_map_size / (1024 * 1024),
            'total_mb': (kmer_map_size + spec_map_size) / (1024 * 1024)
        }

    def _read_and_validate_parquet_file(self) -> pd.DataFrame:
        """Reads and performs initial validation on the Parquet database file."""

        logger.info(f"Loading k-mer database from {self.database_path} (Parquet format)...")
        if not self.database_path.is_file():
            raise DatabaseNotFoundError(
                f"Database file not found: {self.database_path}",
                details={'path': str(self.database_path)}
            )

        try:
            kmer_strain_df: pd.DataFrame = pd.read_parquet(self.database_path)
        except (IOError, ValueError, pd.errors.EmptyDataError) as e:
            raise DatabaseLoadError(
                f"Failed to read or process Parquet database from {self.database_path}",
                details={'error': str(e), 'path': str(self.database_path)}
            ) from e
        except Exception as e:
            raise DatabaseCorruptedError(
                f"Database file appears corrupted or invalid: {self.database_path}",
                details={'error': str(e), 'path': str(self.database_path)}
            ) from e

        if not isinstance(
            kmer_strain_df, pd.DataFrame
        ):  # Should be caught by parquet reader, but good check
            raise TypeError(
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

        if not kmer_strain_df.index.is_unique:
            print(
                f"Warning: K-mer index in {self.database_path} is not unique. Duplicates will be resolved by last occurrence when creating the lookup dictionary."
            )
        return kmer_strain_df

    def _infer_and_set_kmer_length(
        self, kmer_strain_df: pd.DataFrame
    ) -> Tuple[bool, int]:
        """
        Infers k-mer length from DataFrame's first k-mer.
        This length is stored in self.data_derived_kmer_length.
        It also validates this derived length against self.db_kmer_length (from metadata)
        if metadata was available.
        Returns the type of k-mer (str or bytes) and the derived length.
        """
        if kmer_strain_df.index.empty:
            raise ValueError("Cannot infer k-mer length from an empty database index.")

        first_kmer_obj = kmer_strain_df.index[0]
        kmer_type_is_str: bool

        if isinstance(first_kmer_obj, str):
            self.data_derived_kmer_length = len(first_kmer_obj)
            kmer_type_is_str = True
        elif isinstance(first_kmer_obj, bytes):
            self.data_derived_kmer_length = len(first_kmer_obj)
            kmer_type_is_str = False
        else:
            raise TypeError(
                f"Unsupported k-mer type in DataFrame index: {type(first_kmer_obj)}. Expected str or bytes."
            )

        if self.data_derived_kmer_length == 0:
            raise ValueError("First k-mer in database has zero length, which is invalid.")

        # Validate against metadata k-mer length if it exists
        if self.db_kmer_length is not None and self.data_derived_kmer_length != self.db_kmer_length:
            raise ValueError(
                f"K-mer length inferred from data ({self.data_derived_kmer_length}) "
                f"does not match k-mer length from database metadata ({self.db_kmer_length})."
            )

        # Note: self.kmer_length is NOT set here. It's finalized in __init__.
        return kmer_type_is_str, self.data_derived_kmer_length

    def _convert_df_to_numpy_matrix(self, kmer_strain_df: pd.DataFrame) -> np.ndarray:
        """Converts the DataFrame values to a NumPy uint8 matrix."""
        try:
            # Ensure all data is numeric before converting
            if not all(
                kmer_strain_df.map(
                    lambda x: isinstance(x, (int, float, np.number))
                ).all()
            ):
                # Attempt to convert, or raise more specific error if mixed types found
                try:
                    # Convert to float first for broader compatibility before uint8
                    kmer_strain_df_numeric = kmer_strain_df.astype(float)
                except ValueError as e:
                    raise TypeError(
                        f"Non-numeric data found in DataFrame values. Cannot convert to count matrix. Error: {e}"
                    )
            else:
                kmer_strain_df_numeric = kmer_strain_df

            count_matrix = kmer_strain_df_numeric.to_numpy(dtype=np.uint8)
        except ValueError as e:  # Catches issues if values are out of np.uint8 range
            raise TypeError(
                f"Could not convert database values to count matrix (np.uint8). "
                f"Ensure all values are numeric and within 0-255. Error: {e}"
            ) from e
        except Exception as e:  # Catch other potential numpy conversion errors
            raise RuntimeError(
                f"Failed to convert DataFrame to NumPy array: {e}"
            ) from e
        return count_matrix

    def _populate_kmer_map(
        self,
        kmer_strain_df: pd.DataFrame,
        count_matrix: np.ndarray,
        kmer_type_is_str: bool,
        current_k_len: int,
    ) -> None:
        """Populates the self.kmer_to_counts_map from the DataFrame and count matrix."""
        self.kmer_to_counts_map.clear()  # Ensure it's empty before populating
        skipped_kmers_count = 0

        # current_k_len is now passed directly

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
            if len(kmer_bytes) != current_k_len: # Ensure this uses the passed current_k_len
                logger.warning(
                    f"K-mer '{kmer_obj}' (index {i}) has inconsistent length: {len(kmer_bytes)}. Expected {current_k_len}. Skipping."
                )
                skipped_kmers_count += 1
                continue

            self.kmer_to_counts_map[kmer_bytes] = count_matrix[i]
            # Calculate k-mer specificity (number of strains this k-mer appears in)
            # Higher specificity = appears in more strains = less discriminative
            self.kmer_specificity_map[kmer_bytes] = int(np.count_nonzero(count_matrix[i]))

        self.num_kmers = len(self.kmer_to_counts_map)

        # Calculate estimated genome lengths for normalization
        # Estimate by counting total k-mers per strain (not perfect but reasonable proxy)
        for strain_idx, strain_name in enumerate(self.strain_names):
            kmer_count = sum(1 for kmer_counts in self.kmer_to_counts_map.values()
                           if kmer_counts[strain_idx] > 0)
            # Approximate genome length: num_kmers + (k - 1)
            # This is a lower bound since we only see unique k-mers
            estimated_length = kmer_count + (self.kmer_length - 1) if self.kmer_length else kmer_count
            self.strain_genome_lengths[strain_name] = estimated_length

        # Log k-mer specificity statistics for scientific insight
        if self.kmer_specificity_map:
            specificity_values = list(self.kmer_specificity_map.values())
            unique_to_one = sum(1 for s in specificity_values if s == 1)
            shared_by_all = sum(1 for s in specificity_values if s == self.num_strains)
            logger.info(
                f"K-mer specificity analysis:\n"
                f" - Unique to single strain: {unique_to_one} ({100*unique_to_one/self.num_kmers:.1f}%)\n"
                f" - Shared by all strains: {shared_by_all} ({100*shared_by_all/self.num_kmers:.1f}%)\n"
                f" - Average strain count per k-mer: {np.mean(specificity_values):.2f}"
            )

        # Log genome length estimates
        if self.strain_genome_lengths:
            lengths = list(self.strain_genome_lengths.values())
            logger.info(
                f"Estimated genome lengths (for normalization):\n"
                f" - Mean: {np.mean(lengths):,.0f} bp\n"
                f" - Median: {np.median(lengths):,.0f} bp\n"
                f" - Range: {min(lengths):,.0f} - {max(lengths):,.0f} bp"
            )

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
            logger.warning(
                f"Skipped {skipped_kmers_count} k-mers during loading due to type, length, or encoding issues."
            )

    def _load_database(self) -> None:
        """
        Internal method to load data from the Parquet database file.
        Orchestrates calls to helper methods for each step of the loading process.
        Also reads k-mer length and skip_n_kmers from Parquet metadata if available.
        """
        logger.info(f"Loading k-mer database from {self.database_path} (Parquet format)...")

        if not self.database_path.is_file(): # Check file existence early
            raise FileNotFoundError(f"Database file not found: {self.database_path}")

        # Read Parquet schema metadata first
        try:
            schema = pq.read_schema(self.database_path)
            metadata = schema.metadata
            if metadata:
                kmerlen_bytes = metadata.get(b"strainr_kmerlen")
                if kmerlen_bytes:
                    self.db_kmer_length = int(kmerlen_bytes.decode('utf-8'))
                    logger.info(f"Read 'strainr_kmerlen' from Parquet metadata: {self.db_kmer_length}")

                skip_n_bytes = metadata.get(b"strainr_skip_n_kmers")
                if skip_n_bytes:
                    self.db_skip_n_kmers = skip_n_bytes.decode('utf-8').lower() == 'true'
                    logger.info(f"Read 'strainr_skip_n_kmers' from Parquet metadata: {self.db_skip_n_kmers}")
        except Exception as e:
            logger.warning(f"Could not read or parse metadata from Parquet file {self.database_path}: {e}")
            # Proceed without metadata, it might be an older DB or non-Arrow Parquet.

        kmer_strain_df = self._read_and_validate_parquet_file()

        # kmer_length logic is now more complex:
        # This method is called by _load_database.
        # It uses self.db_kmer_length (from metadata) or self.data_derived_kmer_length (inferred from data)
        # to perform its validation and setup. The final decision for self.kmer_length happens in __init__.
        kmer_type_is_str, actual_k_len_from_data = self._infer_and_set_kmer_length(kmer_strain_df)
        # actual_k_len_from_data is now stored in self.data_derived_kmer_length by the call above.

        count_matrix = self._convert_df_to_numpy_matrix(kmer_strain_df)
        # _populate_kmer_map needs the k-mer length that will be used for validation during population.
        # This should be self.db_kmer_length if available, otherwise actual_k_len_from_data.
        # The final self.kmer_length (considering _expected_k_init) isn't set yet.
        # For populating the map, the k-mer length from the file (metadata or inferred from its data) is the ground truth.
        validation_k_len = self.db_kmer_length if self.db_kmer_length is not None else actual_k_len_from_data
        if validation_k_len is None:
             raise RuntimeError("Could not determine k-mer length for map population.") # Should not happen
        self._populate_kmer_map(kmer_strain_df, count_matrix, kmer_type_is_str, validation_k_len)


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

    def get_kmer_specificity(self, kmer: bytes) -> Optional[int]:
        """
        Get the specificity of a k-mer (how many strains it appears in).

        A k-mer with specificity=1 is unique to one strain (highly discriminative).
        A k-mer with specificity=num_strains appears in all strains (not discriminative).

        Args:
            kmer: The k-mer (bytes) to query.

        Returns:
            Number of strains this k-mer appears in, or None if k-mer not in database.
        """
        if not isinstance(kmer, bytes):
            return None
        return self.kmer_specificity_map.get(kmer)

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded k-mer database.

        Returns:
            A dictionary containing key database statistics:
                - "num_strains" (int): Number of strains in the database.
                - "num_kmers" (int): Number of unique k-mers loaded.
                - "kmer_length" (int): The validated length of k-mers in the database.
                - "database_filepath" (str): Absolute path to the database file.
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

        Args:
            test_kmer: The k-mer (string or bytes) to validate.

        Returns:
            True if the length of `test_kmer` matches `self.kmer_length`,
            False otherwise. Returns False for non-str/bytes input.
        """

        if not isinstance(test_kmer, (str, bytes)):
            return False
        return len(test_kmer) == self.kmer_length  # self.kmer_length is now an int


# Example Usage (adapted from StrainKmerDb in original kmer_database.py)
if __name__ == "__main__":
    dummy_kmers_str = [("A" * 4), ("C" * 4), ("G" * 4), ("T" * 4)]
    dummy_strains = ["ExampleStrain1", "ExampleStrain2"]
    dummy_data_np = np.array([[10, 5], [3, 12], [8, 8], [0, 15]], dtype=np.uint8)
    dummy_df_str_idx = pd.DataFrame(
        dummy_data_np, index=dummy_kmers_str, columns=dummy_strains
    )

    try:
        script_dir = pathlib.Path(__file__).parent
    except NameError:
        script_dir = pathlib.Path.cwd()
    dummy_db_output_dir = script_dir / "test_db_output_consolidated"
    dummy_db_output_dir.mkdir(exist_ok=True)

    dummy_db_path_str_parquet = (
        dummy_db_output_dir / "dummy_strain_kmer_db_str_idx.parquet"
    )
    dummy_df_str_idx.to_parquet(dummy_db_path_str_parquet, index=True)
    print(
        f"Created dummy Parquet database (string k-mers) at {dummy_db_path_str_parquet.resolve()}"
    )

    try:
        print(
            "\n--- Testing consolidated StrainKmerDatabase (inferred length) from Parquet ---"
        )
        # Use the new class name StrainKmerDatabase
        db_inferred = StrainKmerDatabase(dummy_db_path_str_parquet)
        kmer_to_find_bytes = b"AAAA"
        counts = db_inferred.get_strain_counts_for_kmer(kmer_to_find_bytes)
        print(f"Counts for {kmer_to_find_bytes.decode('utf-8', 'replace')}: {counts}")

        known_kmer_bytes = b"CCCC"
        if known_kmer_bytes in db_inferred:
            print(f"K-mer {known_kmer_bytes.decode()} is in the database.")

        print(f"Total k-mers in database: {len(db_inferred)}")
        print(f"Database stats: {db_inferred.get_database_stats()}")
        print(f"Is 'AAAA' valid length? {db_inferred.validate_kmer_length(b'AAAA')}")
        print(f"Is 'AAA' valid length? {db_inferred.validate_kmer_length(b'AAA')}")

        print("\n--- Testing with expected_kmer_length provided (from Parquet) ---")
        # Use the new class name StrainKmerDatabase
        db_expected_len = StrainKmerDatabase(
            dummy_db_path_str_parquet, expected_kmer_length=4
        )
        counts_2 = db_expected_len.get_strain_counts_for_kmer(b"GGGG")
        print(f"Counts for b'GGGG': {counts_2}")

    except Exception as e:
        print(f"An error occurred during Parquet database testing: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if dummy_db_path_str_parquet.exists():
            dummy_db_path_str_parquet.unlink()
        if dummy_db_output_dir.exists() and not any(dummy_db_output_dir.iterdir()):
            dummy_db_output_dir.rmdir()
        print("\nCleaned up dummy Parquet database files and directory (if empty).")
