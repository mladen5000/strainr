"""
Statistical analysis and disambiguation of strain classification results.

CHANGES:
- Improved function naming and documentation
- Better type hints throughout
- Fixed logic issues in hit separation
- Improved error handling
- More modular design
"""

import functools
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from genomic_types import (
    CountVector,
    ReadHitResults,
    StrainIndex,
    StrainAbundanceDict,
    ReadId,
    KmerString,
    KmerCountDict
)



class ClassificationAnalyzer:
    """
    Analyzer for processing and disambiguating strain classification results.
    
    This class handles the post-classification analysis including:
    - Separating clear vs ambiguous hits
    - Statistical disambiguation
    - Abundance calculation and normalization
    """
    
    def __init__(
        self, 
        strain_names: List[str],
        disambiguation_mode: str = "max",
        abundance_threshold: float = 0.001,
        num_processes: int = 4
    ):
        """
        Initialize the classification analyzer.
        
        Args:
            strain_names: List of strain identifiers
            disambiguation_mode: Method for resolving ambiguous reads 
                               ("max", "random", "multinomial", "dirichlet")
            abundance_threshold: Minimum abundance threshold for reporting
            num_processes: Number of processes for parallel disambiguation
        """
        self.strain_names = strain_names
        self.disambiguation_mode = disambiguation_mode
        self.abundance_threshold = abundance_threshold
        self.num_processes = num_processes
        self.random_generator = np.random.default_rng()

    def separate_hit_categories(
        self, 
        classification_results: ReadHitResults
    ) -> Tuple[Dict[str, CountVector], Dict[str, CountVector], List[str]]:
        """
        Categorize classification results into clear, ambiguous, and no-hit reads.
        
        Args:
            classification_results: List of (read_id, strain_counts) from classification
            
        Returns:
            Tuple of (clear_hits, ambiguous_hits, no_hits)
            - clear_hits: Reads with single maximum strain
            - ambiguous_hits: Reads with multiple maximum strains  
            - no_hits: Reads with no strain matches
        """
        clear_hits: Dict[str, CountVector] = {}
        ambiguous_hits: Dict[str, CountVector] = {}
        no_hit_reads: List[str] = []
        
        for read_id, strain_count_vector in classification_results:
            if np.all(strain_count_vector == 0):
                # No hits to any strain
                no_hit_reads.append(read_id)
            else:
                # Find positions with maximum values
                max_value = np.max(strain_count_vector)
                max_positions = np.argwhere(strain_count_vector == max_value).flatten()
                
                if len(max_positions) == 1:
                    # Single clear maximum
                    clear_hits[read_id] = strain_count_vector
                elif len(max_positions) > 1:
                    # Multiple maxima - ambiguous
                    ambiguous_hits[read_id] = strain_count_vector
                # Note: Removed the problematic case where max_positions == len(strain_count_vector)
                # as this would mean all strains have equal (maximum) values, which should be
                # treated as ambiguous rather than as a separate "core reads" category

        print(f"Classification summary:")
        print(f"  Clear assignments: {len(clear_hits)}")
        print(f"  Ambiguous assignments: {len(ambiguous_hits)}")
        print(f"  No hits: {len(no_hit_reads)}")
        
        return clear_hits, ambiguous_hits, no_hit_reads

    def resolve_clear_hits(self, clear_hits: Dict[str, CountVector]) -> Dict[str, int]:
        """
        Extract strain indices from reads with clear strain assignments.
        
        Args:
            clear_hits: Dictionary of reads with single maximum strain
            
        Returns:
            Dictionary mapping read_id to strain_index
        """
        return {
            read_id: int(np.argmax(strain_vector)) 
            for read_id, strain_vector in clear_hits.items()
        }

    def calculate_strain_prior(self, clear_strain_assignments: Dict[str, int]) -> Counter[int]:
        """
        Calculate prior strain probabilities from clear assignments.
        
        Args:
            clear_strain_assignments: Dictionary of read_id -> strain_index
            
        Returns:
            Counter with strain_index -> count mapping
        """
        return Counter(clear_strain_assignments.values())

    def convert_prior_to_probability_vector(
        self, 
        strain_prior_counts: Counter[int]
    ) -> np.ndarray:
        """
        Convert strain prior counts to probability vector.
        
        Args:
            strain_prior_counts: Counter of strain indices to counts
            
        Returns:
            Probability vector for all strains
        """
        prior_probabilities = np.zeros(len(self.strain_names))
        total_counts = sum(strain_prior_counts.values())
        
        for strain_index, count in strain_prior_counts.items():
            prior_probabilities[strain_index] = count / total_counts
        
        # Set zero probabilities to small value to avoid division issues
        prior_probabilities[prior_probabilities == 0] = 1e-20
        
        return prior_probabilities

    def _resolve_single_ambiguous_read(
        self, 
        strain_hits: CountVector, 
        prior_probabilities: np.ndarray
    ) -> int:
        """
        Resolve a single ambiguous read using specified disambiguation method.
        
        Args:
            strain_hits: Vector of strain hit counts for the read
            prior_probabilities: Prior probability vector for strains
            
        Returns:
            Selected strain index
        """
        # Threshold to only include maximum values
        max_value = np.max(strain_hits)
        strain_hits = strain_hits.copy()
        strain_hits[strain_hits < max_value] = 0
        
        # Apply prior probabilities
        likelihood_scores = strain_hits * prior_probabilities

        if self.disambiguation_mode == "random":
            return random.choices(
                range(len(likelihood_scores)),
                weights=likelihood_scores,
                k=1
            )[0]
        
        elif self.disambiguation_mode == "max":
            return int(np.argmax(likelihood_scores))
        
        elif self.disambiguation_mode == "dirichlet":
            likelihood_scores[likelihood_scores == 0] = 1e-10
            return int(np.argmax(
                self.random_generator.dirichlet(likelihood_scores, 1)
            ))
        
        elif self.disambiguation_mode == "multinomial":
            likelihood_scores[likelihood_scores == 0] = 1e-10
            normalized_scores = likelihood_scores / np.sum(likelihood_scores)
            return int(np.argmax(
                self.random_generator.multinomial(1, normalized_scores)
            ))
        
        else:
            raise ValueError(f"Unknown disambiguation mode: {self.disambiguation_mode}")

    def resolve_ambiguous_hits(
        self, 
        ambiguous_hits: Dict[str, CountVector], 
        prior_probabilities: np.ndarray
    ) -> Dict[str, int]:
        """
        Resolve ambiguous read assignments using maximum likelihood with priors.
        
        Args:
            ambiguous_hits: Dictionary of read_id -> strain_hit_vector
            prior_probabilities: Prior probability vector for strains
            
        Returns:
            Dictionary of read_id -> resolved_strain_index
        """
        if not ambiguous_hits:
            return {}
            
        resolved_assignments = {}
        
        # Create partial function for multiprocessing
        resolve_function = functools.partial(
            self._resolve_single_ambiguous_read, 
            prior_probabilities=prior_probabilities
        )
        
        # Use fewer cores for disambiguation to avoid oversubscription
        disambiguation_processes = max(1, self.num_processes // 4)
        
        with mp.Pool(processes=disambiguation_processes) as pool:
            resolved_indices = pool.map(resolve_function, ambiguous_hits.values())
            
            for read_id, strain_index in zip(ambiguous_hits.keys(), resolved_indices):
                resolved_assignments[read_id] = strain_index

        return resolved_assignments

    def combine_all_assignments(
        self,
        clear_assignments: Dict[str, int],
        resolved_ambiguous: Dict[str, int], 
        no_hit_reads: List[str]
    ) -> Dict[str, Union[int, str]]:
        """
        Combine all read assignments into final results dictionary.
        
        Args:
            clear_assignments: Clear strain assignments
            resolved_ambiguous: Resolved ambiguous assignments
            no_hit_reads: List of reads with no hits
            
        Returns:
            Combined dictionary with all read assignments
        """
        # Create NA assignments for no-hit reads
        na_assignments = {read_id: "NA" for read_id in no_hit_reads}
        
        # Combine all assignments
        final_assignments = {**clear_assignments, **resolved_ambiguous, **na_assignments}
        
        print(f"Final assignment summary:")
        print(f"  Total reads: {len(final_assignments)}")
        print(f"  Clear: {len(clear_assignments)}")
        print(f"  Resolved ambiguous: {len(resolved_ambiguous)}")
        print(f"  No assignment: {len(na_assignments)}")
        
        return final_assignments

# Alternative streaming analysis method (commented out as requested)
# def analyze_results_streaming(
#     self, 
#     classification_results: Generator[Tuple[str, CountVector], None, None]
# ) -> Generator[Dict[str, Union[int, str]], None, None]:
#     """
#     Stream-based analysis that processes results as they arrive.
#     
#     This alternative approach processes classification results incrementally
#     without storing all results in memory, suitable for very large datasets.
#     
#     Args:
#         classification_results: Generator of classification results
#         
#     Yields:
#         Dictionaries containing processed assignments
#     """
#     clear_hits = {}
#     ambiguous_hits = {}
#     no_hits = []
#     batch_size = 10000
#     
#     for i, (read_id, strain_counts) in enumerate(classification_results):
#         if np.all(strain_counts == 0):
#             no_hits.append(read_id)
#         elif len(np.argwhere(strain_counts == np.max(strain_counts))) == 1:
#             clear_hits[read_id] = strain_counts
#         else:
#             ambiguous_hits[read_id] = strain_counts
#         
#         # Process in batches
#         if (i + 1) % batch_size == 0:
#             yield self._process_batch(clear_hits, ambiguous_hits, no_hits)
#             clear_hits, ambiguous_hits, no_hits = {}, {}, []
#     
#     # Process final batch
#     if clear_hits or ambiguous_hits or no_hits:
#         yield self._process_batch(clear_hits, ambiguous_hits, no_hits)
