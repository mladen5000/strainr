import pathlib
import pickle # For potential future use if not pandas pickle
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

# Type Aliases
CountVector = npt.NDArray[np.uint8]  # Represents a vector of k-mer counts for each strain
Kmer = bytes  # K-mers are represented as bytes for efficiency
KmerDatabaseDict = Dict[Kmer, CountVector] # The core database structure


class KmerStrainDatabase:
    """
    Represents a database of k-mers and their corresponding strain frequency vectors.

    This class loads a k-mer database from a pickled Pandas DataFrame. The DataFrame
    is expected to have k-mers as its index (typically strings or bytes) and strain names
    as its columns. The values should be counts or frequencies (uint8).

    Attributes:
        database_filepath (pathlib.Path): The path to the database file.
        kmer_length (int): The length of k-mers in the database. Determined from data
                           if not provided, or validated if provided.
        kmer_to_counts_map (KmerDatabaseDict): A dictionary mapping each k-mer (bytes)
                                             to its count vector (np.ndarray[np.uint8]).
        strain_names (List[str]): A list of strain names present in the database.
        num_strains (int): The number of strains in the database.
        num_kmers (int): The number of unique k-mers in the database.
    """

    def __init__(
        self,
        database_filepath: Union[str, pathlib.Path],
        expected_kmer_length: Optional[int] = None,
    ) -> None:
        """
        Initializes and loads the KmerStrainDatabase from a file.

        Args:
            database_filepath: Path to the pickled Pandas DataFrame.
                               The DataFrame should have k-mers (strings or bytes) as its index
                               and strain names (strings) as its columns. Cell values should be
                               numeric and convertible to `np.uint8`.
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
            RuntimeError: For lower-level issues during file reading or unpickling, often
                          wrapping underlying exceptions like `pickle.UnpicklingError` or
                          `pd.errors.EmptyDataError`.
        """
        self.database_filepath = pathlib.Path(database_filepath).resolve()
        if not self.database_filepath.is_file(): # More specific check
            raise FileNotFoundError(f"Database file not found or is not a file: {self.database_filepath}")

        # Initialize attributes that will be set by _load_database
        self.kmer_length: int = 0 
        self.kmer_to_counts_map: KmerDatabaseDict = {}
        self.strain_names: List[str] = []
        self.num_strains: int = 0
        self.num_kmers: int = 0
        
        self._load_database(expected_kmer_length)

        print(
            f"Successfully loaded database from {self.database_filepath}\n"
            f" - K-mer length: {self.kmer_length}\n"
            f" - Number of k-mers: {self.num_kmers}\n"
            f" - Number of strains: {self.num_strains}\n"
            f" - Strain names: {', '.join(self.strain_names[:5])}{'...' if len(self.strain_names) > 5 else ''}"
        )


    def _load_database(self, expected_kmer_length: Optional[int]) -> None:
        """
        Internal method to load data from the pickled DataFrame file.

        Args:
            expected_kmer_length: Optional k-mer length for validation. If provided,
                                  it overrides any length inferred from the data and
                                  all k-mers must conform to it.

        Raises:
            RuntimeError: For issues like file not found (should be caught earlier),
                          unpickling errors, or if the loaded data is not a DataFrame.
            ValueError: If the DataFrame is empty, contains no k-mers/strains,
                        or if k-mer lengths are inconsistent or do not match
                        `expected_kmer_length` (if provided).
            TypeError: If k-mer index contains unsupported types or if DataFrame values
                       cannot be converted to `np.uint8`.
        """
        try:
            kmer_strain_df: pd.DataFrame = pd.read_pickle(self.database_filepath)
        except (pickle.UnpicklingError, pd.errors.EmptyDataError, EOFError) as e:
            raise RuntimeError(f"Could not read or unpickle database file: {self.database_filepath}. File may be corrupted or empty. Original error: {e}") from e
        except FileNotFoundError: # Should ideally be caught by __init__
             raise RuntimeError(f"Database file {self.database_filepath} vanished after initial check.") from None
        except Exception as e: # Catch-all for other pd.read_pickle issues
            raise RuntimeError(f"An unexpected error occurred while reading {self.database_filepath}: {e}") from e

        if not isinstance(kmer_strain_df, pd.DataFrame):
            raise RuntimeError(f"Data loaded from {self.database_filepath} is not a pandas DataFrame (type: {type(kmer_strain_df)}).")

        if kmer_strain_df.empty:
            raise ValueError(f"Loaded database is empty: {self.database_filepath}")

        self.strain_names = list(kmer_strain_df.columns.astype(str)) # Ensure string names
        self.num_strains = len(self.strain_names)
        if self.num_strains == 0:
            raise ValueError("Database contains no strain information (no columns).")

        if kmer_strain_df.index.empty:
            raise ValueError("Database contains no k-mers (empty index).")
        
        if not kmer_strain_df.index.is_unique:
             print(f"Warning: K-mer index in {self.database_filepath} is not unique. Duplicates will be resolved by last occurrence when creating the lookup dictionary.")

        # Determine and validate k-mer length
        first_kmer_obj = kmer_strain_df.index[0]
        if isinstance(first_kmer_obj, str):
            inferred_k_len = len(first_kmer_obj)
            kmer_type_is_str = True
        elif isinstance(first_kmer_obj, bytes):
            inferred_k_len = len(first_kmer_obj)
            kmer_type_is_str = False
        else:
            raise TypeError(f"Unsupported k-mer type in index: {type(first_kmer_obj)}. Expected str or bytes.")

        if inferred_k_len == 0:
            raise ValueError("First k-mer in database has zero length, which is invalid.")

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

        # Process k-mers and build the lookup dictionary
        temp_kmer_map: KmerDatabaseDict = {}
        
        # Convert data to NumPy array first for efficiency
        try:
            count_matrix = kmer_strain_df.to_numpy(dtype=np.uint8)
        except ValueError as e: # More specific error for conversion
            raise TypeError(
                f"Could not convert database values to count matrix (np.uint8). "
                f"Ensure all values are numeric and within 0-255. Error: {e}"
            ) from e
        
        for i, kmer_obj in enumerate(kmer_strain_df.index):
            kmer_bytes: Kmer
            current_len: int

            if kmer_type_is_str:
                if not isinstance(kmer_obj, str):
                    raise TypeError(f"Inconsistent k-mer type at index {i}. Expected str, got {type(kmer_obj)}.")
                current_len = len(kmer_obj)
                if current_len != self.kmer_length:
                    raise ValueError(
                        f"Inconsistent k-mer string length at index {i}. Expected {self.kmer_length}, "
                        f"but k-mer '{kmer_obj}' has length {current_len}."
                    )
                try:
                    kmer_bytes = kmer_obj.encode('utf-8') # Use UTF-8 for broader compatibility
                except UnicodeEncodeError as e:
                    raise ValueError(f"Failed to encode k-mer string '{kmer_obj}' (index {i}) to UTF-8 bytes. Error: {e}") from e
            else: # k-mer type is bytes
                if not isinstance(kmer_obj, bytes):
                     raise TypeError(f"Inconsistent k-mer type at index {i}. Expected bytes, got {type(kmer_obj)}.")
                kmer_bytes = kmer_obj
                current_len = len(kmer_bytes)
                if current_len != self.kmer_length:
                    raise ValueError(
                        f"Inconsistent k-mer bytes length at index {i}. Expected {self.kmer_length}, "
                        f"but k-mer {kmer_bytes!r} has length {current_len}."
                    )
            
            # Final check on byte length (especially if string encoding changed length unexpectedly, though less likely with fixed-length strings)
            if len(kmer_bytes) != self.kmer_length:
                # This should ideally be caught by string length check if kmer_type_is_str,
                # but good as a safeguard for byte representation.
                raise ValueError(
                    f"Post-encoding/cast k-mer byte length validation failed at index {i}. "
                    f"Expected {self.kmer_length}, but k-mer '{kmer_obj}' (bytes: {kmer_bytes!r}) "
                    f"has byte length {len(kmer_bytes)}."
                )
            temp_kmer_map[kmer_bytes] = count_matrix[i]

        self.kmer_to_counts_map = temp_kmer_map
        self.num_kmers = len(self.kmer_to_counts_map)

        if self.num_kmers == 0 and not kmer_strain_df.index.empty:
            # This implies all k-mers were somehow filtered or failed validation,
            # though current checks should raise errors earlier.
            raise ValueError("No k-mers were successfully processed into the database, despite non-empty input index.")


    def get_strain_counts_for_kmer(self, kmer: Kmer) -> Optional[CountVector]:
        """
        Retrieves the strain count vector for a given k-mer.

        Args:
            kmer: The k-mer (bytes) to look up. It should be the canonical
                  representation if the database stores canonical k-mers.

        Returns:
            A NumPy array (CountVector) of uint8 counts for each strain if the
            k-mer is found, otherwise None.
        """
        return self.kmer_to_counts_map.get(kmer)

    def __len__(self) -> int:
        """Returns the number of unique k-mers in the database."""
        return self.num_kmers

    def __contains__(self, kmer: Kmer) -> bool:
        """Checks if a k-mer is present in the database."""
        return kmer in self.kmer_to_counts_map

# Example Usage (optional, for testing or demonstration):
# Example Usage (optional, for testing or demonstration):
if __name__ == "__main__":
    # Create a dummy database file for testing
    # K-mers as strings in the DataFrame index
    dummy_kmers_str = ["ATGC", "CGTA", "GTAC", "TACG"] # k=4
    dummy_strains = ["Ecoli_K12", "Salmonella_enterica"]
    dummy_data_np = np.array([[10, 5], [3, 12], [8, 8], [0, 15]], dtype=np.uint8)
    dummy_df_str_idx = pd.DataFrame(dummy_data_np, index=dummy_kmers_str, columns=dummy_strains)
    
    dummy_db_dir = pathlib.Path(__file__).parent / "test_db_output"
    dummy_db_dir.mkdir(exist_ok=True)
    dummy_db_path_str = dummy_db_dir / "dummy_kmer_db_str_idx.pkl"
    dummy_df_str_idx.to_pickle(dummy_db_path_str)
    print(f"Created dummy database (string k-mers) at {dummy_db_path_str.resolve()}")

    # K-mers as bytes in the DataFrame index
    dummy_kmers_bytes = [k.encode('utf-8') for k in dummy_kmers_str]
    dummy_df_bytes_idx = pd.DataFrame(dummy_data_np, index=dummy_kmers_bytes, columns=dummy_strains)
    dummy_db_path_bytes = dummy_db_dir / "dummy_kmer_db_bytes_idx.pkl"
    dummy_df_bytes_idx.to_pickle(dummy_db_path_bytes)
    print(f"Created dummy database (byte k-mers) at {dummy_db_path_bytes.resolve()}")

    try:
        print("\n--- Testing with string k-mer database (inferred length) ---")
        db_str_inferred = KmerStrainDatabase(dummy_db_path_str) # k-mer length inferred as 4
        kmer_to_find_bytes = b"CGTA"
        counts = db_str_inferred.get_strain_counts_for_kmer(kmer_to_find_bytes)
        print(f"Counts for {kmer_to_find_bytes.decode('utf-8', 'replace')}: {counts}")
        assert b"GTAC" in db_str_inferred, "K-mer GTAC (bytes) should be in db_str_inferred"
        print(f"Total k-mers: {len(db_str_inferred)}")


        print("\n--- Testing with byte k-mer database (expected length provided) ---")
        db_bytes_expected = KmerStrainDatabase(dummy_db_path_bytes, expected_kmer_length=4)
        counts_2 = db_bytes_expected.get_strain_counts_for_kmer(b"TACG")
        print(f"Counts for b'TACG': {counts_2}")


        print("\n--- Testing expected_kmer_length mismatch (should fail) ---")
        try:
            KmerStrainDatabase(dummy_db_path_str, expected_kmer_length=5)
        except ValueError as ve:
            print(f"Caught expected error for length mismatch: {ve}")

        print("\n--- Testing with a non-existent file (should fail) ---")
        try:
            KmerStrainDatabase("non_existent_file.pkl")
        except FileNotFoundError as fnf:
            print(f"Caught expected error for non-existent file: {fnf}")

    except Exception as e:
        print(f"An error occurred during database testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files
        if dummy_db_path_str.exists():
            dummy_db_path_str.unlink()
        if dummy_db_path_bytes.exists():
            dummy_db_path_bytes.unlink()
        if dummy_db_dir.exists() and not any(dummy_db_dir.iterdir()): # Remove if empty
             dummy_db_dir.rmdir()
        print("\nCleaned up dummy database files and directory (if empty).")
