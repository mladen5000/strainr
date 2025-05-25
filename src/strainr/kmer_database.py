#!/usr/bin/env python

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
                               The DataFrame should have k-mers as index and strains as columns.
            expected_kmer_length: Optional. If provided, validates that all k-mers in the
                                  loaded database match this length. If None, the k-mer
                                  length is inferred from the first k-mer in the database.

        Raises:
            FileNotFoundError: If the database_filepath does not exist.
            ValueError: If the database is empty, k-mers have inconsistent lengths,
                        or if `expected_kmer_length` is provided and does not match
                        the k-mer lengths in the file.
            TypeError: If the data in the DataFrame is not of the expected type (e.g., counts
                       are not convertible to uint8).
        """
        self.database_filepath = pathlib.Path(database_filepath).resolve()
        if not self.database_filepath.exists():
            raise FileNotFoundError(f"Database file not found: {self.database_filepath}")

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
            expected_kmer_length: Optional k-mer length for validation.

        Raises:
            ValueError: For issues with k-mer lengths or empty database.
            TypeError: For data type issues in the DataFrame.
        """
        try:
            kmer_strain_df: pd.DataFrame = pd.read_pickle(self.database_filepath)
        except Exception as e:
            # Catch pandas-specific errors or general unpickling errors
            raise ValueError(f"Could not read or unpickle database file: {self.database_filepath}. Original error: {e}") from e

        if kmer_strain_df.empty:
            raise ValueError(f"Loaded database is empty: {self.database_filepath}")

        self.strain_names = list(kmer_strain_df.columns)
        self.num_strains = len(self.strain_names)
        if self.num_strains == 0:
            raise ValueError("Database contains no strain information (no columns).")

        # Process k-mers and determine k-mer length
        if kmer_strain_df.index.empty:
            raise ValueError("Database contains no k-mers (empty index).")
        
        # Ensure k-mers are bytes
        kmers_as_bytes: List[Kmer] = []
        inferred_k_len_from_first: Optional[int] = None

        for i, kmer_obj in enumerate(kmer_strain_df.index):
            kmer_bytes: Kmer
            if isinstance(kmer_obj, bytes):
                kmer_bytes = kmer_obj
            elif isinstance(kmer_obj, str):
                kmer_bytes = kmer_obj.encode('ascii') # Assuming ASCII k-mers
            else:
                raise TypeError(
                    f"Unsupported k-mer type in index: {type(kmer_obj)}. Expected str or bytes."
                )
            
            if i == 0:
                inferred_k_len_from_first = len(kmer_bytes)
                if expected_kmer_length is None:
                    self.kmer_length = inferred_k_len_from_first
                elif expected_kmer_length != inferred_k_len_from_first:
                    raise ValueError(
                        f"Expected k-mer length {expected_kmer_length} does not match "
                        f"length of first k-mer in database ({inferred_k_len_from_first})."
                    )
                else:
                    self.kmer_length = expected_kmer_length # Consistent

            if len(kmer_bytes) != self.kmer_length:
                raise ValueError(
                    f"Inconsistent k-mer lengths found. Expected {self.kmer_length}, "
                    f"but k-mer '{kmer_bytes.decode('ascii', errors='replace')}' (index {i}) "
                    f"has length {len(kmer_bytes)}."
                )
            kmers_as_bytes.append(kmer_bytes)

        if not hasattr(self, 'kmer_length'): # Should be set if index was not empty
             raise ValueError("Could not determine k-mer length from database index.")


        try:
            # Convert DataFrame values to numpy array of specified dtype
            count_matrix = kmer_strain_df.to_numpy(dtype=np.uint8)
        except Exception as e: # Catches potential conversion errors
            raise TypeError(
                f"Could not convert database values to count matrix (np.uint8). Error: {e}"
            ) from e

        self.kmer_to_counts_map: KmerDatabaseDict = dict(zip(kmers_as_bytes, count_matrix))
        self.num_kmers = len(self.kmer_to_counts_map)


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
# if __name__ == "__main__":
#     # Create a dummy database file for testing
#     dummy_kmers = ["AAA", "AAC", "AAG", "AAT"]
#     dummy_strains = ["Strain1", "Strain2"]
#     dummy_data = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
#     dummy_df = pd.DataFrame(dummy_data, index=dummy_kmers, columns=dummy_strains)
#     
#     dummy_db_path = pathlib.Path("dummy_kmer_database.pkl")
#     dummy_df.to_pickle(dummy_db_path)
# 
#     print(f"Created dummy database at {dummy_db_path.resolve()}")
# 
#     try:
#         # Test with expected_kmer_length
#         db_instance = KmerStrainDatabase(dummy_db_path, expected_kmer_length=3)
#         kmer_to_find = b"AAC"
#         counts = db_instance.get_strain_counts_for_kmer(kmer_to_find)
#         if counts is not None:
#             print(f"Counts for {kmer_to_find.decode()}: {counts}")
#         else:
#             print(f"K-mer {kmer_to_find.decode()} not found.")
# 
#         if b"AAG" in db_instance:
#             print("K-mer AAG is in the database.")
#         
#         print(f"Total k-mers in database: {len(db_instance)}")
# 
#         # Test without expected_kmer_length (inferred)
#         db_instance_inferred = KmerStrainDatabase(dummy_db_path)
#         
#         # Test with incorrect kmer length expectation
#         # db_instance_wrong_k = KmerStrainDatabase(dummy_db_path, expected_kmer_length=4)
# 
#     except Exception as e:
#         print(f"An error occurred during database testing: {e}")
#     finally:
#         if dummy_db_path.exists():
#             dummy_db_path.unlink() # Clean up dummy file
#             print(f"Cleaned up dummy database: {dummy_db_path.resolve()}")
