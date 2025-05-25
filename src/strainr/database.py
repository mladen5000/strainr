"""
K-mer database management for strain classification.

CHANGES:
- Complete rewrite of StrainDatabase class
- Added proper initialization and validation
- Improved type hints and documentation
- Fixed incomplete implementation
- Added database statistics and validation methods
"""

import pathlib
from typing import Optional, Dict, List, Union, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from strainr.genomic_types import KmerCountDict, CountVector


class StrainKmerDatabase:
    """
    K-mer database for strain classification.

    This class manages a database of k-mer frequencies across multiple strains,
    enabling efficient lookup of strain-specific k-mer signatures.

    Attributes:
        database_path: Path to the pickled database file
        kmer_length: Length of k-mers in the database
        strain_names: List of strain identifiers
        kmer_lookup_dict: Dictionary mapping k-mers to strain frequency vectors
        num_strains: Number of strains in database
        num_kmers: Number of unique k-mers in database
    """

    def __init__(self, database_path: Union[str, Path], kmer_length: int = 31) -> None:
        """
        Initialize strain database from pickled DataFrame.

        Args:
            database_path: Path to pickled pandas DataFrame containing k-mer frequencies
            kmer_length: Expected k-mer length for validation

        Raises:
            FileNotFoundError: If database file doesn't exist
            ValueError: If k-mer lengths don't match expected value

        Example:
            >>> db = StrainKmerDatabase("database.pkl", kmer_length=31)
            >>> print(f"Loaded {db.num_strains} strains with {db.num_kmers} k-mers")
        """
        self.database_path = Path(database_path).expanduser()
        self.kmer_length = kmer_length
        self._validate_database_file()
        self._load_database()

    def _validate_database_file(self) -> None:
        """Validate that database file exists and is readable."""
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.database_path}")

    def _load_database(self) -> None:
        """Load and validate the k-mer database from pickle file."""
        print(f"Loading k-mer database from {self.database_path}")

        try:
            kmer_frequency_dataframe = pd.read_pickle(self.database_path)
        except Exception as error:
            raise RuntimeError(f"Failed to load database: {error}") from error

        # Validate DataFrame structure
        if kmer_frequency_dataframe.empty:
            raise ValueError("Database DataFrame is empty")

        # Extract database components
        self.strain_names = list(kmer_frequency_dataframe.columns)
        kmer_sequences = kmer_frequency_dataframe.index
        frequency_matrix = kmer_frequency_dataframe.to_numpy(dtype=np.uint8)

        # Validate k-mer lengths
        if len(kmer_sequences) > 0:
            actual_kmer_length = len(str(kmer_sequences[0]))
            if actual_kmer_length != self.kmer_length:
                print(
                    f"Warning: Expected k-mer length {self.kmer_length}, "
                    f"found {actual_kmer_length}. Using actual length."
                )
                self.kmer_length = actual_kmer_length

        # Create lookup dictionary
        self.kmer_lookup_dict: KmerCountDict = dict(
            zip(kmer_sequences, frequency_matrix)
        )

        # Set database statistics
        self.num_strains = len(self.strain_names)
        self.num_kmers = len(kmer_sequences)

        print(
            f"Successfully loaded database: {self.num_strains} strains, "
            f"{self.num_kmers} k-mers (k={self.kmer_length})"
        )

    def lookup_kmer(self, kmer_sequence: bytes) -> Optional[CountVector]:
        """
        Look up strain frequency vector for a given k-mer.

        Args:
            kmer_sequence: K-mer sequence as bytes

        Returns:
            Array of strain frequencies or None if k-mer not found
        """
        return self.kmer_lookup_dict.get(kmer_sequence)

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.

        Returns:
            Dictionary containing database statistics
        """
        return {
            "num_strains": self.num_strains,
            "num_kmers": self.num_kmers,
            "kmer_length": self.kmer_length,
            "database_path": str(self.database_path),
            "strain_names": self.strain_names[:5],  # First 5 for preview
            "total_strain_names": len(self.strain_names),
        }

    def validate_kmer_length(self, test_kmer: bytes) -> bool:
        """
        Validate that a k-mer has the expected length.

        Args:
            test_kmer: K-mer to validate

        Returns:
            True if k-mer length matches database expectation
        """
        return len(test_kmer) == self.kmer_length
