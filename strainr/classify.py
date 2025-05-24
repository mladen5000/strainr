"""
Core classification engine for strain identification using k-mer matching.

This module provides the `StrainClassifier` class, which is responsible for
processing FASTQ files, matching k-mers against a pre-built database,
and assigning reads to strains.
"""

import functools
import multiprocessing as mp
import pathlib
import time
from collections import Counter, defaultdict # Counter and defaultdict not explicitly used, but good to keep for potential extensions
from typing import Generator, Tuple, Dict, List, Any, Optional # Added Optional

import numpy as np
from Bio.SeqIO.QualityIO import FastqGeneralIterator

# Assuming genomic_types.py defines these or similar structures
from strainr.genomic_types import ( # Adjusted import path assuming strainr is the root package
    CountVector,
    ReadHitResults,
    StrainAbundanceDict, # Not used in this file directly, but part of a type set
    ReadId,
    KmerString, # Not used in this file directly
    KmerCountDict # Not used in this file directly
)
from strainr.kmer_database import KmerStrainDatabase # Corrected import
from strainr.utils import open_file_transparently # Adjusted for consistency, assuming this is the intended function


class StrainClassifier:
    """
    Assigns sequence reads to strains using k-mer matching against a database.

    This class encapsulates the logic for reading sequences from FASTQ files,
    extracting k-mers, querying a `KmerStrainDatabase`, and aggregating
    k-mer matches to determine strain assignments for each read. It supports
    multiprocessing for improved performance on large datasets.

    Attributes:
        database (KmerStrainDatabase): An instance of `KmerStrainDatabase` containing
            the k-mers and their strain association vectors.
        num_processes (int): The number of worker processes to use for parallel
            read classification.
        chunk_size (int): The number of reads processed by each worker in a single batch.
    """

    def __init__(
        self,
        database: KmerStrainDatabase,
        num_processes: int = 4,
        chunk_size: int = 10000,
    ) -> None:
        """
        Initializes the StrainClassifier.

        Args:
            database: An instance of `KmerStrainDatabase` containing the k-mers
                      and their associated strain frequency/count vectors.
            num_processes: The number of parallel processes to use for classifying
                           reads. Defaults to 4.
            chunk_size: The number of reads to be processed in a single chunk by
                        each worker process. Defaults to 10000.
        """
        if not isinstance(database, KmerStrainDatabase):
            raise TypeError(
                f"Database must be an instance of KmerStrainDatabase, not {type(database).__name__}"
            )
        if num_processes <= 0:
            raise ValueError("Number of processes must be a positive integer.")
        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")
            
        self.database: KmerStrainDatabase = database
        self.num_processes: int = num_processes
        self.chunk_size: int = chunk_size

    def count_kmer_matches_for_read(
        self, read_id: ReadId, sequence_bytes: bytes
    ) -> Tuple[ReadId, CountVector]:
        """
        Counts k-mer matches for a single read against the strain database.

        This method extracts all k-mers from the given sequence, queries them
        against the k-mer database, and aggregates the resulting strain count
        vectors. It uses a memory view for efficient k-mer extraction.

        Args:
            read_id: The unique identifier for the sequence read.
            sequence_bytes: The raw sequence data for the read, encoded as bytes.

        Returns:
            A tuple containing the `read_id` and a `CountVector` (NumPy array of uint8)
            representing the sum of k-mer hits across all strains. If no k-mers from
            the read are found in the database, a zero vector is returned.
        """
        matched_kmer_vectors: List[CountVector] = []
        # Ensure zero_vector matches the number of strains in the loaded database
        zero_vector: CountVector = np.zeros(self.database.num_strains, dtype=np.uint8)
        
        # Ensure kmer_length is valid for the sequence
        if len(sequence_bytes) < self.database.kmer_length:
            return read_id, zero_vector # Sequence too short for any k-mers

        max_kmer_start_index: int = len(sequence_bytes) - self.database.kmer_length + 1

        with memoryview(sequence_bytes) as sequence_view:
            for start_pos in range(max_kmer_start_index):
                # Extract k-mer as a memoryview slice first
                kmer_slice: memoryview = sequence_view[
                    start_pos : start_pos + self.database.kmer_length
                ]
                # Convert to bytes only when necessary for lookup, if the DB keys are bytes
                kmer_as_bytes: bytes = kmer_slice.tobytes()
                
                # Use the consistent method name from KmerStrainDatabase
                strain_counts_for_kmer: Optional[CountVector] = \
                    self.database.get_strain_counts_for_kmer(kmer_as_bytes)

                if strain_counts_for_kmer is not None:
                    matched_kmer_vectors.append(strain_counts_for_kmer)

        if matched_kmer_vectors:
            # Summing a list of NumPy arrays
            total_strain_counts: CountVector = np.sum(matched_kmer_vectors, axis=0, dtype=np.uint8)
            return read_id, total_strain_counts
        else:
            return read_id, zero_vector

    def _count_kmer_matches_helper(
        self, read_tuple: Tuple[ReadId, bytes]
    ) -> Tuple[ReadId, CountVector]:
        """
        Internal helper to unpack arguments for `count_kmer_matches_for_read`
        when used with `multiprocessing.Pool.imap_unordered`.

        Args:
            read_tuple: A tuple containing the `read_id` (str) and
                        `sequence_bytes` (bytes).

        Returns:
            The result of `count_kmer_matches_for_read`, which is a tuple
            of `read_id` and its corresponding `CountVector`.
        """
        return self.count_kmer_matches_for_read(*read_tuple)

    def _generate_encoded_sequences(
        self, input_fastq: pathlib.Path
    ) -> Generator[Tuple[ReadId, bytes], None, None]:
        """
        Parses a FASTQ file and yields read IDs and their sequences as bytes.

        This generator function uses `Bio.SeqIO.QualityIO.FastqGeneralIterator`
        for efficient FASTQ parsing and `open_file_transparently` to handle
        potentially gzipped files. Sequences are encoded to UTF-8 bytes.

        Args:
            input_fastq: The path to the input FASTQ file.

        Yields:
            Tuples, where each tuple contains the sequence identifier (ReadId, str)
            and the sequence data encoded as bytes.
        """
        # Assuming open_file_transparently is correctly imported from strainr.utils
        with open_file_transparently(input_fastq) as file_handle:
            for seq_id, sequence, _ in FastqGeneralIterator(file_handle):
                yield seq_id, bytes(sequence, "utf-8")

    def classify_reads_in_file(self, input_fastq: pathlib.Path) -> ReadHitResults:
        """
        Classifies all reads in a given FASTQ file against the strain database.

        This method orchestrates the read classification process:
        1. Counts the total number of reads in the FASTQ file (for progress logging).
           Note: This initial read counting step might be slow for very large files.
        2. Generates read ID and sequence byte pairs using `_generate_encoded_sequences`.
        3. Uses a multiprocessing pool (`mp.Pool`) to parallelize k-mer matching
           for reads using `_count_kmer_matches_helper`.
        4. Collects and returns the results.

        Args:
            input_fastq: The path to the input FASTQ file.

        Returns:
            A `ReadHitResults` object, which is a list of tuples, each containing
            a `read_id` (str) and its corresponding `CountVector` (NumPy array).

        Example:
            >>> # Assuming 'db_instance' is a loaded KmerStrainDatabase
            >>> classifier = StrainClassifier(database=db_instance, num_processes=2)
            >>> results = classifier.classify_reads_in_file(pathlib.Path("sample.fastq"))
            >>> print(f"Classified {len(results)} reads.")
            Classified 100 reads. # Example output
        """
        start_time: float = time.time()

        # Note: Counting total reads by iterating through the file can be
        # time-consuming for very large files. Consider removing or making optional
        # if startup time is critical for huge datasets.
        # Using 'rt' mode with open_file_transparently for text-based line counting.
        read_count: int = 0
        with open_file_transparently(input_fastq) as f_in:
            for _ in FastqGeneralIterator(f_in): # More accurate count of records
                read_count +=1
        print(f"Processing {read_count} reads from {input_fastq}...")

        sequence_iterator: Generator[Tuple[ReadId, bytes], None, None] = \
            self._generate_encoded_sequences(input_fastq)

        classification_results: ReadHitResults = []
        with mp.Pool(processes=self.num_processes) as process_pool:
            # Using imap_unordered to get results as they complete, potentially faster
            # and better for memory if results were processed one by one.
            # list() forces all results to be collected here.
            classification_results = list(
                process_pool.imap_unordered(
                    self._count_kmer_matches_helper,
                    sequence_iterator,
                    chunksize=self.chunk_size,
                )
            )

        elapsed_time: float = time.time() - start_time
        print(f"Classification completed in {elapsed_time:.2f} seconds for {read_count} reads.")

        return classification_results


# Alternative optimized classification method (commented out as requested in original file)
# def classify_reads_streaming(self, input_fastq: pathlib.Path) -> Generator[Tuple[str, CountVector], None, None]:
#     """
#     Stream-based classification that yields results as they're computed.
#
#     This alternative approach processes reads one at a time without storing
#     all results in memory, which is beneficial for very large FASTQ files.
#
#     Args:
#         input_fastq: Path to input FASTQ file
#
#     Yields:
#         Tuples of (read_id, strain_match_counts)
#     """
#     sequence_iterator = self._generate_encoded_sequences(input_fastq)
#
#     with mp.Pool(processes=self.num_processes) as process_pool:
#         for result in process_pool.imap_unordered(
#             self._count_kmer_matches_helper,
#             sequence_iterator,
#             chunksize=self.chunk_size
#         ):
#             yield result
