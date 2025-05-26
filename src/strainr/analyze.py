"""
Statistical analysis and disambiguation of strain classification results.

This module provides the `ClassificationAnalyzer` class, which takes raw k-mer
hit results for sequence reads against a set of strains and performs several
post-processing steps. These include categorizing reads based on hit clarity
(clear, ambiguous, no hits), calculating prior probabilities from clear hits,
and using these priors to disambiguate reads with multiple likely strain matches.
Different disambiguation strategies are supported.
"""

import functools
import multiprocessing as mp
# import random # No direct use of 'random' module, np.random.default_rng() is used.
from collections import Counter
from typing import Dict, List, Tuple, Any, Union, Set # Removed Optional, pd is removed

import numpy as np
# import pandas as pd  # Not directly used in this file

# Assuming genomic_types.py is in the same package or accessible in PYTHONPATH
# Adjust import path as necessary (e.g., from strainr.genomic_types import ...)
from strainr.genomic_types import (
    CountVector,
    ReadHitResults,  # Typically List[Tuple[ReadId, CountVector]]
    StrainIndex,  # Typically int
    # StrainAbundanceDict, # Not used here
    ReadId,  # Typically str
    # KmerString, # Not used here
    # KmerCountDict # Not used here
)


class ClassificationAnalyzer:
    """
    Analyzes and disambiguates strain classification results.

    This class processes raw k-mer hit counts for reads against strains to
    categorize read assignments, calculate strain priors, and resolve
    ambiguous assignments using various statistical methods.

    Attributes:
        strain_names (List[str]): A list of strain identifiers. The order and
            length of this list define the expected structure of CountVector.
        disambiguation_mode (str): The strategy used to resolve reads that map
            ambiguously to multiple strains. Supported modes: "max", "random",
            "multinomial", "dirichlet".
        abundance_threshold (float): A threshold used in some downstream analyses
            (though not directly in disambiguation methods here) for determining
            significant strain presence.
        num_processes (int): The number of parallel processes to use for computationally
            intensive steps like ambiguous read resolution.
        random_generator (np.random.Generator): A NumPy random number generator instance
            for stochastic disambiguation methods.
    """

    SUPPORTED_DISAMBIGUATION_MODES: Set[str] = {
        "max",
        "random",
        "multinomial",
        "dirichlet",
    }

    def __init__(
        self,
        strain_names: List[str],
        disambiguation_mode: str = "max",
        abundance_threshold: float = 0.001,  # Currently not used in this class's methods
        num_processes: int = 4,
    ) -> None:
        """
        Initializes the ClassificationAnalyzer.

        Args:
            strain_names: A list of strain identifiers (names).
            disambiguation_mode: The method for resolving ambiguous reads.
                Must be one of "max", "random", "multinomial", or "dirichlet".
                Defaults to "max".
            abundance_threshold: Minimum relative abundance for a strain to be
                considered significant in downstream analyses (not used directly by
                methods in this class but stored for potential use).
                Defaults to 0.001.
            num_processes: The number of CPU processes to use for parallelizable
                operations. Defaults to 4.

        Raises:
            ValueError: If `disambiguation_mode` is not supported, or if
                        `num_processes` is not positive.
            TypeError: If `strain_names` is not a list of strings.
        """
        if not isinstance(strain_names, list) or not all(
            isinstance(s, str) for s in strain_names
        ):
            raise TypeError("strain_names must be a list of strings.")
        if not strain_names:
            raise ValueError("strain_names cannot be empty.")
        if disambiguation_mode not in self.SUPPORTED_DISAMBIGUATION_MODES:
            raise ValueError(
                f"Unsupported disambiguation_mode: {disambiguation_mode}. "
                f"Supported modes are: {self.SUPPORTED_DISAMBIGUATION_MODES}"
            )
        if not isinstance(num_processes, int) or num_processes <= 0:
            raise ValueError("num_processes must be a positive integer.")
        if not (0.0 <= abundance_threshold < 1.0):
            raise ValueError(
                "abundance_threshold must be between 0.0 and 1.0 (exclusive of 1.0)."
            )

        self.strain_names: List[str] = strain_names
        self.disambiguation_mode: str = disambiguation_mode
        self.abundance_threshold: float = (
            abundance_threshold  # Stored but not used by methods here
        )
        self.num_processes: int = num_processes
        self.random_generator: np.random.Generator = np.random.default_rng()

    def separate_hit_categories(
        self, classification_results: ReadHitResults
    ) -> Tuple[Dict[ReadId, CountVector], Dict[ReadId, CountVector], List[ReadId]]:
        """
        Categorizes reads based on their k-mer hit patterns to strains.

        Reads are classified into three groups:
        - Clear hits: Reads that have a k-mer hit profile strongly favouring a single strain.
        - Ambiguous hits: Reads whose k-mer hits map with equal maximum strength to multiple strains.
        - No hits: Reads with no k-mer matches to any strain in the database.

        Args:
            classification_results: A list of tuples, where each tuple contains a
                `ReadId` (str) and its corresponding `CountVector` (NumPy array
                of k-mer hit counts against each strain).

        Returns:
            A tuple containing three elements:
            - `clear_hits_dict`: Dict mapping `ReadId` to `CountVector` for clear hits.
            - `ambiguous_hits_dict`: Dict mapping `ReadId` to `CountVector` for ambiguous hits.
            - `no_hit_read_ids`: List of `ReadId` for reads with no hits.
        """
        if not isinstance(classification_results, list):
            raise TypeError("classification_results must be a list of tuples.")

        clear_hits_dict: Dict[ReadId, CountVector] = {}
        ambiguous_hits_dict: Dict[ReadId, CountVector] = {}
        no_hit_read_ids: List[ReadId] = []

        for read_id, strain_count_vector in classification_results:
            if not isinstance(read_id, str) or not isinstance(
                strain_count_vector, np.ndarray
            ):
                # Or log a warning and skip
                raise TypeError(
                    "Each item in classification_results must be (ReadId, CountVector)."
                )

            if np.all(strain_count_vector == 0):
                no_hit_read_ids.append(read_id)
            else:
                max_value = np.max(strain_count_vector)
                # np.argwhere returns a 2D array, flatten it for 1D count vectors
                max_positions = np.argwhere(strain_count_vector == max_value).flatten()

                if len(max_positions) == 1:
                    clear_hits_dict[read_id] = strain_count_vector
                else:  # len(max_positions) > 1 or (rarely) 0 if vector was non-numeric (should not happen)
                    ambiguous_hits_dict[read_id] = strain_count_vector

        print(
            f"Hit categorization summary:\n"
            f"  Clear assignments: {len(clear_hits_dict)}\n"
            f"  Ambiguous assignments: {len(ambiguous_hits_dict)}\n"
            f"  No hits: {len(no_hit_read_ids)}"
        )
        return clear_hits_dict, ambiguous_hits_dict, no_hit_read_ids

    def resolve_clear_hits_to_indices(
        self, clear_hits_dict: Dict[ReadId, CountVector]
    ) -> Dict[ReadId, StrainIndex]:
        """
        Converts clear hit count vectors to their corresponding strain indices.

        For each read in the `clear_hits_dict`, this method identifies the strain
        with the maximum k-mer hit count and returns its index.

        Args:
            clear_hits_dict: A dictionary mapping `ReadId` to `CountVector` for reads
                             identified as clear hits (having a single maximum score).

        Returns:
            A dictionary mapping `ReadId` to the `StrainIndex` (int) of the
            unambiguously assigned strain.
        """
        if not isinstance(clear_hits_dict, dict):
            raise TypeError("clear_hits_dict must be a dictionary.")

        return {
            read_id: int(np.argmax(strain_vector))
            for read_id, strain_vector in clear_hits_dict.items()
        }

    def calculate_strain_prior_from_assignments(
        self, clear_strain_assignments: Dict[ReadId, StrainIndex]
    ) -> Counter[StrainIndex]:
        """
        Calculates the prior probability (raw counts) of each strain based on clear assignments.

        Args:
            clear_strain_assignments: A dictionary mapping `ReadId` to the `StrainIndex`
                                      of the strain it was clearly assigned to.

        Returns:
            A `collections.Counter` where keys are `StrainIndex` (int) and values
            are the number of reads clearly assigned to that strain.
        """
        if not isinstance(clear_strain_assignments, dict):
            raise TypeError("clear_strain_assignments must be a dictionary.")

        return Counter(clear_strain_assignments.values())

    def convert_prior_counts_to_probability_vector(
        self, strain_prior_counts: Counter[StrainIndex]
    ) -> np.ndarray:
        """
        Converts raw strain prior counts into a normalized probability vector.

        The resulting vector will have probabilities for each strain, ordered by
        the original strain list index. Probabilities for strains with zero counts
        are set to a very small epsilon value (1e-20) to prevent issues in
        downstream calculations (e.g., division by zero).

        Args:
            strain_prior_counts: A `collections.Counter` mapping `StrainIndex` (int)
                                 to its raw count from clear read assignments.

        Returns:
            A NumPy array of floats representing the prior probability for each
            strain. The length of the array matches `len(self.strain_names)`.
            Probabilities for strains with zero clear assignments are set to a small
            epsilon (1e-20) to prevent issues like zero probabilities in products
            during downstream calculations.
        """
        if not isinstance(strain_prior_counts, Counter):
            raise TypeError("strain_prior_counts must be a Counter.")

        num_total_strains = len(self.strain_names)
        prior_probabilities = np.full(
            num_total_strains, 1e-20, dtype=float
        )  # Initialize with epsilon

        total_clear_counts = sum(strain_prior_counts.values())

        if total_clear_counts > 0:
            for strain_index, count in strain_prior_counts.items():
                if 0 <= strain_index < num_total_strains:  # Ensure index is valid
                    prior_probabilities[strain_index] = (
                        float(count) / total_clear_counts
                    )

        # Re-normalize if any epsilon values were not overwritten, to ensure sum is close to 1
        # (though with initial epsilon, this might not be strictly necessary if all have some counts)
        # sum_probs = np.sum(prior_probabilities)
        # if sum_probs > 0 and not np.isclose(sum_probs, 1.0): # Avoid division by zero if all were epsilon
        #    prior_probabilities /= sum_probs
        # This re-normalization might be too aggressive if many strains have zero counts.
        # The primary goal of epsilon is to avoid zero probabilities in products.

        return prior_probabilities

    def _resolve_single_ambiguous_read(
        self, strain_hit_counts: CountVector, prior_probabilities: np.ndarray
    ) -> StrainIndex:
        """
        Resolves a single ambiguously assigned read to one strain.

        This internal method applies a chosen disambiguation strategy based on
        `self.disambiguation_mode`. It first identifies strains sharing the maximum
        hit count for the read. These hits are then weighted by the `prior_probabilities`
        to calculate likelihood scores. Finally, a single strain is selected based
        on these scores and the chosen disambiguation mode.

        Args:
            strain_hit_counts: The `CountVector` (NumPy array of k-mer hit counts)
                               for the ambiguous read.
            prior_probabilities: The pre-calculated prior probability vector for all
                                 strains, used to weight the likelihood of each
                                 potential strain assignment.

        Returns:
            The `StrainIndex` (int) of the strain chosen to resolve the ambiguity.

        Raises:
            ValueError: If `self.disambiguation_mode` is not recognized.
        """
        # Filter for strains at maximum hit count for this read
        max_hit_value = np.max(strain_hit_counts)
        # Create a working copy to modify
        candidate_strain_hits = strain_hit_counts.astype(
            float
        )  # Use float for calculations
        candidate_strain_hits[candidate_strain_hits < max_hit_value] = 0.0

        # Apply prior probabilities to get likelihood scores
        # Ensure prior_probabilities has the same length as candidate_strain_hits
        if len(prior_probabilities) != len(candidate_strain_hits):
            raise ValueError(
                "Length of prior_probabilities must match strain_hit_counts."
            )

        likelihood_scores = candidate_strain_hits * prior_probabilities

        # If all likelihood scores are zero (e.g. if priors for max-hit strains were zero),
        # fall back to just the candidate hits (equal likelihood among max-hit strains)
        if np.all(likelihood_scores == 0):
            likelihood_scores = candidate_strain_hits  # Use original max hits if priors zeroed everything out

        # Normalize likelihood_scores to sum to 1 for probabilistic choices
        sum_likelihood_scores = np.sum(likelihood_scores)
        if (
            sum_likelihood_scores == 0
        ):  # Should be rare if handled above, means no valid choice
            # Fallback: if all scores are zero, pick randomly among those with max_hit_value
            max_indices = np.where(strain_hit_counts == max_hit_value)[0]
            return int(self.random_generator.choice(max_indices))

        normalized_scores = likelihood_scores / sum_likelihood_scores

        if self.disambiguation_mode == "max":
        # Np.argmax will return the first index in case of ties in likelihood_scores.
        # Uses non-normalized likelihood_scores as only the maximum matters.
        return int(np.argmax(likelihood_scores))

        # For probabilistic modes, use normalized_scores to make a weighted choice.
        strain_indices = np.arange(len(self.strain_names))

        if self.disambiguation_mode == "random":
            return int(
                self.random_generator.choice(strain_indices, p=normalized_scores)
            )

        elif self.disambiguation_mode == "multinomial":
            # Draw 1 time from a multinomial distribution
            selected_index_array = self.random_generator.multinomial(
                1, pvals=normalized_scores
            )
            return int(np.argmax(selected_index_array))  # Index where the '1' is

        elif self.disambiguation_mode == "dirichlet":
            # Dirichlet gives sample from distribution; not directly an index.
            # We need to adjust likelihood_scores for dirichlet (alpha values > 0)
            dirichlet_alpha = likelihood_scores.copy()
            dirichlet_alpha[dirichlet_alpha == 0] = 1e-10  # Ensure alpha > 0
            # Sample from dirichlet, then choose based on sampled probabilities
            sampled_probabilities = self.random_generator.dirichlet(
                alpha=dirichlet_alpha
            )
            return int(
                self.random_generator.choice(strain_indices, p=sampled_probabilities)
            )
        else:
            # This case should have been caught by __init__
            raise ValueError(f"Unknown disambiguation mode: {self.disambiguation_mode}")

    def resolve_ambiguous_hits_parallel(
        self,
        ambiguous_hits_dict: Dict[ReadId, CountVector],
        prior_probabilities: np.ndarray,
    ) -> Dict[ReadId, StrainIndex]:
        """
        Resolves ambiguous read assignments in parallel.

        This method uses a multiprocessing pool to apply the chosen disambiguation
        strategy to each read in the `ambiguous_hits_dict`.

        Args:
            ambiguous_hits_dict: A dictionary mapping `ReadId` to `CountVector` for
                                 reads identified as ambiguous hits.
            prior_probabilities: The pre-calculated prior probability vector for
                                 all strains, used by the disambiguation strategy.

        Returns:
            A dictionary mapping `ReadId` to the `StrainIndex` (int) of the
            strain chosen to resolve the ambiguity for that read.
        """
        if not ambiguous_hits_dict:
            return {}
        if not isinstance(ambiguous_hits_dict, dict):
            raise TypeError("ambiguous_hits_dict must be a dictionary.")
        if not isinstance(prior_probabilities, np.ndarray):
            raise TypeError("prior_probabilities must be a NumPy array.")

        resolved_assignments: Dict[ReadId, StrainIndex] = {}

        # Create a partial function with fixed prior_probabilities for starmap/map
        resolve_func_with_priors = functools.partial(
            self._resolve_single_ambiguous_read, prior_probabilities=prior_probabilities
        )

        # Determine number of processes for this specific task (can be less than self.num_processes)
        # Using a smaller pool for disambiguation might be reasonable if it's memory/CPU intensive per task
        # but not massively parallelizable beyond a few cores, or to leave resources.
        # Original code used num_processes // 4.
        num_disambiguation_processes = max(
            1, self.num_processes // 2
        )  # Adjusted policy

        # Items for mapping: list of CountVectors
        hit_vectors_to_process = list(ambiguous_hits_dict.values())
        read_ids_in_order = list(ambiguous_hits_dict.keys())

        if (
            not hit_vectors_to_process
        ):  # Should be caught by the first check but good for safety
            return {}

        with mp.Pool(processes=num_disambiguation_processes) as pool:
            # `map` is suitable here as we pass only one iterable (the hit_vectors)
            # to the function (resolve_func_with_priors already has the priors).
            resolved_indices_list = pool.map(
                resolve_func_with_priors, hit_vectors_to_process
            )

        for read_id, resolved_strain_index in zip(
            read_ids_in_order, resolved_indices_list
        ):
            resolved_assignments[read_id] = resolved_strain_index

        print(f"Resolved {len(resolved_assignments)} ambiguous assignments.")
        return resolved_assignments

    def combine_assignments(
        self,
        clear_assignments: Dict[ReadId, StrainIndex],
        resolved_ambiguous_assignments: Dict[ReadId, StrainIndex],
        no_hit_read_ids: List[ReadId],
        unassigned_marker: str = "NA",  # Consistent marker for unassigned
    ) -> Dict[ReadId, Union[StrainIndex, str]]:
        """
        Combines assignments from clear, resolved ambiguous, and no-hit categories.

        Args:
            clear_assignments: Dictionary mapping `ReadId` to `StrainIndex` for clear hits.
            resolved_ambiguous_assignments: Dictionary mapping `ReadId` to `StrainIndex`
                                            for ambiguously assigned reads that have been resolved.
            no_hit_read_ids: A list of `ReadId`s for reads that had no k-mer hits.
            unassigned_marker: The string to use as a value for reads in `no_hit_read_ids`.
                               Defaults to "NA".

        Returns:
            A dictionary representing the final assignments, where keys are `ReadId`s
            and values are either `StrainIndex` (int) or the `unassigned_marker` (str).
        """
        if (
            not isinstance(clear_assignments, dict)
            or not isinstance(resolved_ambiguous_assignments, dict)
            or not isinstance(no_hit_read_ids, list)
        ):
            raise TypeError("Invalid input types for assignment dictionaries or list.")

        # Create assignments for no-hit reads
        unassigned_dict: Dict[ReadId, str] = {
            read_id: unassigned_marker for read_id in no_hit_read_ids
        }

        # Combine all dictionaries. The order of merging dictates precedence if keys overlap
        # (though they shouldn't if categories are mutually exclusive).
        final_assignments: Dict[ReadId, Union[StrainIndex, str]] = {}
        final_assignments.update(clear_assignments)
        final_assignments.update(resolved_ambiguous_assignments)
        final_assignments.update(
            unassigned_dict
        )  # Type hint needs Union[StrainIndex, str]

        print(
            f"Final combined assignment summary:\n"
            f"  Total reads processed: {len(final_assignments)}\n"
            f"  Assigned (clear + resolved): {len(clear_assignments) + len(resolved_ambiguous_assignments)}\n"
            f"  Marked '{unassigned_marker}': {len(unassigned_dict)}"
        )
        return final_assignments


# The commented-out streaming analysis method is preserved below as per instructions.
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
#     batch_size = 10000 # Example batch size
#
#     # Placeholder for _process_batch which would need to be defined
#     # and would likely use methods like resolve_clear_hits_to_indices,
#     # calculate_strain_prior_from_assignments, etc., potentially on batches.
#     # def _process_batch(self, clear, ambig, no):
#     #     # ... process this batch ...
#     #     # This would involve prior calculation, disambiguation for the batch,
#     #     # and then combining. Prior might need to be global or updated.
#     #     # This is non-trivial to implement correctly for streaming priors.
#     #     return combined_batch_assignments
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
#             # yield self._process_batch(clear_hits, ambiguous_hits, no_hits) # Call to undefined method
#             clear_hits, ambiguous_hits, no_hits = {}, {}, [] # Reset for next batch
#
#     # Process final batch
#     if clear_hits or ambiguous_hits or no_hits:
#         # yield self._process_batch(clear_hits, ambiguous_hits, no_hits) # Call to undefined method
#         pass
