"""
K-mer database management for strain classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any  # Tuple removed

import numpy as np
import pandas as pd


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
        self.database_filepath = pathlib.Path(
            database_filepath
        ).resolve()  # Resolve for absolute path
        if not self.database_filepath.is_file():
            raise FileNotFoundError(
                f"Database file not found or is not a file: {self.database_filepath}"
            )

        # Initialize attributes
        self.kmer_length: int = 0
        self.kmer_to_counts_map: Dict[bytes, np.ndarray] = {} # Explicit type
        self.strain_names: List[str] = []
        self.num_strains: int = 0
        self.num_kmers: int = 0

        self._load_database(expected_kmer_length)

        print(
            f"Successfully loaded database from {self.database_filepath}\n"
            f" - K-mer length: {self.kmer_length}\n"
            f" - Number of k-mers: {self.num_kmers}\n"
            f" - Number of strains: {self.num_strains}\n"
            f" - Strain names preview: {', '.join(self.strain_names[:5])}{'...' if len(self.strain_names) > 5 else ''}"
        )

    def _load_database(self, expected_kmer_length: Optional[int]) -> None:
        """
        Internal method to load data from the Parquet database file.
        """
        print(f"Loading k-mer database from {self.database_filepath} (Parquet format)...")
        try:
            kmer_strain_df: pd.DataFrame = pd.read_parquet(self.database_filepath)
        except (IOError, ValueError, pd.errors.EmptyDataError) as e:
            raise RuntimeError(

                f"Database file disappeared after validation: {self.database_path}"
            )
        except (IOError, ValueError, pd.errors.EmptyDataError) as e:
            raise RuntimeError(
                f"Failed to read or process Parquet database from {self.database_path}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Database file {self.database_filepath} vanished after initial check."
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while reading Parquet file {self.database_filepath}: {e}"
            ) from e

        if not isinstance(kmer_strain_df, pd.DataFrame):
            raise RuntimeError(
                f"Data loaded from {self.database_filepath} is not a pandas DataFrame (type: {type(kmer_strain_df)})."
            )

        if kmer_strain_df.empty:
            raise ValueError(f"Loaded database is empty: {self.database_filepath}")

        self.strain_names = list(kmer_strain_df.columns.astype(str))

        self.num_strains = len(self.strain_names)
        if self.num_strains == 0:
            raise ValueError("Database contains no strain information (no columns).")

        if kmer_strain_df.index.empty:
            raise ValueError("Database contains no k-mers (empty index).")

        if not kmer_strain_df.index.is_unique:
            print(
                f"Warning: K-mer index in {self.database_filepath} is not unique. Duplicates will be resolved by last occurrence when creating the lookup dictionary."
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

        else:
            raise TypeError(
                f"Unsupported k-mer type in DataFrame index: {type(first_kmer_obj)}. Expected str or bytes."
            )

        if inferred_k_len == 0:
            raise ValueError(
                "First k-mer in database has zero length, which is invalid."
            )

        if expected_kmer_length is not None:
            if expected_kmer_length != inferred_k_len:

                raise ValueError(
                    f"Provided expected_kmer_length ({expected_kmer_length}) does not match "
                    f"length of first k-mer in database ({inferred_k_len})."
                )
            self.kmer_length = expected_kmer_length
        else:
            self.kmer_length = inferred_k_len
            print(f"K-mer length inferred from first k-mer: {self.kmer_length}")

        if self.kmer_length <= 0:
            raise ValueError(
                f"Determined k-mer length ({self.kmer_length}) must be positive."
            )

        temp_kmer_map: Dict[bytes, np.ndarray] = {} # Explicit type
        try:
            count_matrix = kmer_strain_df.to_numpy(dtype=np.uint8)
        except ValueError as e:
            raise TypeError(
                f"Could not convert database values to count matrix (np.uint8). "
                f"Ensure all values are numeric and within 0-255. Error: {e}"
            ) from e

        skipped_kmers_count = 0

        for i, kmer_obj in enumerate(kmer_strain_df.index):
            kmer_bytes: bytes
            current_obj_len: int

            if kmer_type_is_str:
                if not isinstance(kmer_obj, str):
                    print(
                        f"Warning: Inconsistent k-mer type at index {i}. Expected str, got {type(kmer_obj)}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
                current_obj_len = len(kmer_obj)
                if current_obj_len != self.kmer_length:
                    print(
                        f"Warning: Inconsistent k-mer string length at index {i}. Expected {self.kmer_length}, k-mer '{kmer_obj}' has length {current_obj_len}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
                try:
                    kmer_bytes = kmer_obj.encode("utf-8")
                except UnicodeEncodeError as e:
                    print(
                        f"Warning: Failed to encode k-mer string '{kmer_obj}' (index {i}) to UTF-8 bytes. Error: {e}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
            else:  # k-mer type is bytes
                if not isinstance(kmer_obj, bytes):
                    print(
                        f"Warning: Inconsistent k-mer type at index {i}. Expected bytes, got {type(kmer_obj)}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
                kmer_bytes = kmer_obj
                current_obj_len = len(kmer_bytes)
                if current_obj_len != self.kmer_length:
                    print(
                        f"Warning: Inconsistent k-mer bytes length at index {i}. Expected {self.kmer_length}, k-mer {kmer_bytes!r} has length {current_obj_len}. Skipping."
                    )
                    skipped_kmers_count += 1
                    continue
            
            temp_kmer_map[kmer_bytes] = count_matrix[i]

        self.kmer_to_counts_map = temp_kmer_map
        self.num_kmers = len(self.kmer_to_counts_map)

        if (
            self.num_kmers == 0
            and not kmer_strain_df.index.empty
            and skipped_kmers_count == len(kmer_strain_df.index)
        ):
            raise ValueError(
                f"No k-mers were successfully loaded into the database from {self.database_filepath} "
                f"(all {skipped_kmers_count} k-mers from input were skipped). "
                "This is likely due to encoding issues or consistent length mismatches. Check warnings."
            )
        if skipped_kmers_count > 0:
            print(
                f"Warning: Skipped {skipped_kmers_count} k-mers during loading due to type, length, or encoding issues."
            )

    def get_strain_counts_for_kmer(self, kmer: bytes) -> Optional[np.ndarray]:
        """
        Retrieves the strain count vector for a given k-mer.

        Args:
            kmer: The k-mer (bytes) to look up.

        Returns:
            A NumPy array (np.ndarray[np.uint8]) of counts for each strain if the
            k-mer is found, otherwise None.
        """
        if not isinstance(kmer, bytes):
            return None
        return self.kmer_to_counts_map.get(kmer)


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
            "database_filepath": str(self.database_filepath),
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
        return len(test_kmer) == self.kmer_length

    def __len__(self) -> int:
        """Returns the number of unique k-mers in the database."""
        return self.num_kmers

    def __contains__(self, kmer: bytes) -> bool:
        """Checks if a k-mer (bytes) is present in the database."""
        if not isinstance(kmer, bytes):
            return False
        return kmer in self.kmer_to_counts_map


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

    dummy_db_path_str_parquet = dummy_db_output_dir / "dummy_strain_kmer_db_str_idx.parquet"
    dummy_df_str_idx.to_parquet(dummy_db_path_str_parquet, index=True)
    print(f"Created dummy Parquet database (string k-mers) at {dummy_db_path_str_parquet.resolve()}")

    try:
        print("\n--- Testing consolidated StrainKmerDatabase (inferred length) from Parquet ---")
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
        db_expected_len = StrainKmerDatabase(dummy_db_path_str_parquet, expected_kmer_length=4)
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
