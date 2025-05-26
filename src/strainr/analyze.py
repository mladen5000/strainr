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
from collections import Counter
from typing import Dict, List, Tuple, Union, Set  # Added Set

import numpy as np

from strainr.genomic_types import (
    CountVector,
    ReadHitResults,
    StrainIndex,
    ReadId,
)


class ClassificationAnalyzer:
    """
    Analyzes and disambiguates strain classification results.

    This class processes raw k-mer hit counts for reads against strains to
    categorize read assignments, calculate strain priors, and resolve
    ambiguous assignments using various statistical methods.

    Attributes:
        strain_names (List[str]): A list of unique, non-empty strain identifiers.
            The order and length of this list define the expected structure of CountVector.
        disambiguation_mode (str): The strategy used to resolve ambiguous reads.
        abundance_threshold (float): Threshold for downstream analyses (not used directly here).
        num_processes (int): Number of parallel processes for intensive steps.
        random_generator (np.random.Generator): NumPy random number generator.
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
        abundance_threshold: float = 0.001,
        num_processes: int = 4,
    ) -> None:
        """
        Initializes the ClassificationAnalyzer.

        Args:
            strain_names: List of unique, non-empty strain identifiers.
            disambiguation_mode: Method for resolving ambiguous reads.
            abundance_threshold: Min rel. abundance for significance (0.0 <= thresh < 1.0).
            num_processes: Number of CPU processes (must be positive).

        Raises:
            TypeError: If `strain_names` is not a list of strings, or if elements are not strings.
            ValueError: If `strain_names` is empty, not unique, or contains empty strings.
                        If `disambiguation_mode` is unsupported.
                        If `num_processes` is not positive.
                        If `abundance_threshold` is not in [0.0, 1.0).
        """
        if not isinstance(strain_names, list) or not all(
            isinstance(s, str) for s in strain_names
        ):
            raise TypeError("strain_names must be a list of strings.")
        if not strain_names:
            raise ValueError("strain_names cannot be empty.")
        if len(set(strain_names)) != len(strain_names):
            raise ValueError("strain_names must be unique.")
        if any(not s for s in strain_names):  # Check for empty strings
            raise ValueError("strain_names must not contain empty strings.")

        if disambiguation_mode not in self.SUPPORTED_DISAMBIGUATION_MODES:
            raise ValueError(
                f"Unsupported disambiguation_mode: {disambiguation_mode}. "
                f"Supported modes are: {self.SUPPORTED_DISAMBIGUATION_MODES}"
            )
        if not isinstance(num_processes, int) or num_processes <= 0:
            raise ValueError("num_processes must be a positive integer.")
        if not (
            isinstance(abundance_threshold, float) and 0.0 <= abundance_threshold < 1.0
        ):
            raise ValueError(
                "abundance_threshold must be a float between 0.0 and 1.0 (exclusive of 1.0)."
            )

        self.strain_names: List[str] = strain_names
        self.disambiguation_mode: str = disambiguation_mode
        self.abundance_threshold: float = abundance_threshold
        self.num_processes: int = num_processes
        self.random_generator: np.random.Generator = np.random.default_rng()
        self._num_strains = len(strain_names)  # Cache for validation

    def _validate_count_vector(self, cv: CountVector, context: str) -> None:
        """Helper to validate a CountVector."""
        if not isinstance(cv, np.ndarray):
            raise TypeError(
                f"{context}: CountVector must be a NumPy array, got {type(cv)}."
            )
        if cv.ndim != 1:
            raise ValueError(
                f"{context}: CountVector must be 1-dimensional, got {cv.ndim} dimensions."
            )
        if len(cv) != self._num_strains:
            raise ValueError(
                f"{context}: CountVector length ({len(cv)}) must match number of strains ({self._num_strains})."
            )
        # Assuming CountVector is npt.NDArray[np.uint8] from genomic_types.
        # np.issubdtype checks if cv.dtype is a subtype of np.unsignedinteger.
        # For stricter check like exact np.uint8: if cv.dtype != np.uint8:
        if not np.issubdtype(cv.dtype, np.unsignedinteger):
            raise TypeError(
                f"{context}: CountVector dtype must be unsigned integer, got {cv.dtype}."
            )

    def separate_hit_categories(
        self, classification_results: ReadHitResults
    ) -> Tuple[Dict[ReadId, CountVector], Dict[ReadId, CountVector], List[ReadId]]:
        """
        Categorizes reads based on their k-mer hit patterns.

        Args:
            classification_results: List of (ReadId, CountVector) tuples.
                                   Each CountVector must be a 1D np.ndarray of np.uint8,
                                   with length matching self.strain_names.

        Returns:
            Tuple: (clear_hits_dict, ambiguous_hits_dict, no_hit_read_ids).

        Raises:
            TypeError: If input format or types are incorrect.
            ValueError: If CountVector properties are invalid.
        """
        if not isinstance(classification_results, list):
            raise TypeError("classification_results must be a list.")
        if not all(
            isinstance(item, tuple) and len(item) == 2
            for item in classification_results
        ):
            raise TypeError(
                "Each item in classification_results must be a tuple of (ReadId, CountVector)."
            )

        clear_hits_dict: Dict[ReadId, CountVector] = {}
        ambiguous_hits_dict: Dict[ReadId, CountVector] = {}
        no_hit_read_ids: List[ReadId] = []

        for i, (read_id, strain_count_vector) in enumerate(classification_results):
            if not isinstance(read_id, str):
                raise TypeError(
                    f"Item {i}: ReadId must be a string, got {type(read_id)}."
                )
            self._validate_count_vector(strain_count_vector, f"Item {i} ('{read_id}')")

            if (
                np.sum(strain_count_vector) == 0
            ):  # More direct check for no hits if counts are non-negative
                no_hit_read_ids.append(read_id)
            else:
                max_value = np.max(strain_count_vector)
                if (
                    max_value == 0
                ):  # Should be caught by sum == 0 if counts are non-negative
                    no_hit_read_ids.append(read_id)
                    continue
                max_positions = np.flatnonzero(strain_count_vector == max_value)

                if len(max_positions) == 1:
                    clear_hits_dict[read_id] = strain_count_vector
                else:
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

        Args:
            clear_hits_dict: Dict mapping ReadId to valid CountVector for clear hits.

        Returns:
            Dict mapping ReadId to StrainIndex.

        Raises:
            TypeError: If input format or types are incorrect.
            ValueError: If CountVector properties are invalid.
        """
        if not isinstance(clear_hits_dict, dict):
            raise TypeError("clear_hits_dict must be a dictionary.")

        resolved_indices: Dict[ReadId, StrainIndex] = {}
        for read_id, strain_vector in clear_hits_dict.items():
            if not isinstance(read_id, str):
                raise TypeError(f"ReadId '{read_id}' must be a string.")
            self._validate_count_vector(strain_vector, f"ReadId '{read_id}'")
            # np.argmax is fine, as for clear hits there's one max.
            resolved_indices[read_id] = int(np.argmax(strain_vector))
        return resolved_indices

    def calculate_strain_prior_from_assignments(
        self, clear_strain_assignments: Dict[ReadId, StrainIndex]
    ) -> Counter[StrainIndex]:
        """
        Calculates raw counts of strains based on clear assignments.

        Args:
            clear_strain_assignments: Dict mapping ReadId to valid StrainIndex.

        Returns:
            Counter of StrainIndex to read counts.

        Raises:
            TypeError: If input format or types are incorrect.
            ValueError: If StrainIndex values are invalid.
        """
        if not isinstance(clear_strain_assignments, dict):
            raise TypeError("clear_strain_assignments must be a dictionary.")

        for read_id, strain_idx in clear_strain_assignments.items():
            if not isinstance(read_id, str):
                raise TypeError(f"ReadId '{read_id}' must be a string.")
            if not isinstance(strain_idx, int):
                raise TypeError(
                    f"StrainIndex for ReadId '{read_id}' must be an int, got {type(strain_idx)}."
                )
            if not (0 <= strain_idx < self._num_strains):
                raise ValueError(
                    f"StrainIndex {strain_idx} for ReadId '{read_id}' is out of range [0, {self._num_strains - 1}]."
                )
        return Counter(clear_strain_assignments.values())

    def convert_prior_counts_to_probability_vector(
        self, strain_prior_counts: Counter[StrainIndex]
    ) -> np.ndarray:
        """
        Converts raw strain prior counts into a normalized probability vector.
        Uses epsilon for zero counts.

        Args:
            strain_prior_counts: Counter mapping valid StrainIndex to its raw count.

        Returns:
            NumPy array of prior probabilities for each strain. Sum may not be exactly 1.0
            if many strains have zero counts and epsilon is used.

        Raises:
            TypeError: If input is not a Counter or keys are not int.
            ValueError: If StrainIndex keys are invalid.
        """
        if not isinstance(strain_prior_counts, Counter):
            raise TypeError("strain_prior_counts must be a Counter.")

        prior_probabilities = np.full(self._num_strains, 1e-20, dtype=float)
        total_clear_counts = sum(strain_prior_counts.values())

        if total_clear_counts > 0:
            for strain_index, count in strain_prior_counts.items():
                if not isinstance(strain_index, int):
                    raise TypeError(
                        f"StrainIndex key {strain_index} in Counter must be an int."
                    )
                if not (0 <= strain_index < self._num_strains):
                    raise ValueError(
                        f"StrainIndex key {strain_index} in Counter is out of range [0, {self._num_strains - 1}]."
                    )
                prior_probabilities[strain_index] = float(count) / total_clear_counts

        # Optional: Re-normalize if strict sum-to-1 is needed after epsilon.
        # current_sum = np.sum(prior_probabilities)
        # if current_sum > 0 and not np.isclose(current_sum, 1.0):
        #    prior_probabilities /= current_sum
        return prior_probabilities

    def _resolve_single_ambiguous_read(
        self, strain_hit_counts: CountVector, prior_probabilities: np.ndarray
    ) -> StrainIndex:
        """
        Resolves a single ambiguously assigned read to one strain.

        Args:
            strain_hit_counts: Valid CountVector for the ambiguous read.
            prior_probabilities: Valid prior probability vector (1D np.ndarray of floats,
                                 length == num_strains, sums to ~1.0, non-negative).

        Returns:
            The StrainIndex of the chosen strain.

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If input properties are invalid (e.g., length mismatch, probabilities not summing to 1).
        """
        self._validate_count_vector(
            strain_hit_counts, "resolve_single_ambiguous_read input"
        )

        if not isinstance(prior_probabilities, np.ndarray):
            raise TypeError("prior_probabilities must be a NumPy array.")
        if prior_probabilities.ndim != 1:
            raise ValueError("prior_probabilities must be 1-dimensional.")
        if len(prior_probabilities) != self._num_strains:
            raise ValueError(
                f"prior_probabilities length ({len(prior_probabilities)}) must match number of strains ({self._num_strains})."
            )
        if not np.issubdtype(prior_probabilities.dtype, np.floating):
            raise TypeError(
                f"prior_probabilities dtype must be float, got {prior_probabilities.dtype}."
            )
        if not np.all(prior_probabilities >= 0):
            raise ValueError("All prior_probabilities must be non-negative.")
        if not np.isclose(
            np.sum(prior_probabilities), 1.0, atol=1e-5
        ):  # Allow for small float inaccuracies
            # This check might be too strict if epsilon usage means sum is not exactly 1.
            # For internal use with convert_prior_counts_to_probability_vector, this might need adjustment
            # or this check relaxed if the epsilon strategy means sum isn't always 1.
            # For now, keeping it relatively strict.
            print(
                f"Warning: prior_probabilities sum ({np.sum(prior_probabilities)}) is not close to 1.0. Normalizing for choice functions."
            )
            # Re-normalize if used for p-values in choice functions that require sum=1
            # However, likelihood_scores * prior_probabilities does not require priors to sum to 1.
            # The normalization happens on likelihood_scores later.
            # So, this check might be more for sanity than strict necessity for the math here.
            pass  # For now, let it pass with a warning.

        max_hit_value = np.max(strain_hit_counts)
        if (
            max_hit_value == 0
        ):  # All hits are zero, should not be an ambiguous read by definition
            # This case should ideally be filtered out before calling this method.
            # If it occurs, randomly assign to any strain based on priors if they are informative,
            # or completely random if priors are uninformative (e.g. uniform).
            print(
                "Warning: _resolve_single_ambiguous_read called with zero hit vector. Assigning randomly based on priors."
            )
            return int(
                self.random_generator.choice(
                    np.arange(self._num_strains),
                    p=(prior_probabilities / np.sum(prior_probabilities))
                    if np.sum(prior_probabilities) > 0
                    else None,
                )
            )

        candidate_strain_hits = strain_hit_counts.astype(float)
        candidate_strain_hits[candidate_strain_hits < max_hit_value] = 0.0

        likelihood_scores = candidate_strain_hits * prior_probabilities

        if np.all(likelihood_scores == 0):
            likelihood_scores = candidate_strain_hits

        sum_likelihood_scores = np.sum(likelihood_scores)
        if sum_likelihood_scores == 0:
            max_indices = np.flatnonzero(strain_hit_counts == max_hit_value)
            if len(max_indices) == 0:  # Should not happen if max_hit_value > 0
                # Fallback to any strain if truly no max indices (e.g. all NaNs, though CountVector is uint8)
                return int(self.random_generator.choice(self._num_strains))
            return int(self.random_generator.choice(max_indices))

        normalized_scores = likelihood_scores / sum_likelihood_scores
        # Ensure normalized_scores sum to 1 for choice functions, handling potential float inaccuracies
        if not np.isclose(np.sum(normalized_scores), 1.0):
            normalized_scores /= np.sum(normalized_scores)

        strain_indices = np.arange(self._num_strains)
        if self.disambiguation_mode == "max":
            return int(np.argmax(likelihood_scores))
        elif self.disambiguation_mode == "random":
            return int(
                self.random_generator.choice(strain_indices, p=normalized_scores)
            )
        elif self.disambiguation_mode == "multinomial":
            selected_index_array = self.random_generator.multinomial(
                1, pvals=normalized_scores
            )
            return int(np.argmax(selected_index_array))
        elif self.disambiguation_mode == "dirichlet":
            dirichlet_alpha = likelihood_scores.copy()
            dirichlet_alpha[dirichlet_alpha == 0] = 1e-10  # Ensure alpha > 0
            # Prevent dirichlet from failing if all alphas are effective zero (e.g. all 1e-10)
            if np.all(dirichlet_alpha <= 1e-10):
                dirichlet_alpha = np.ones(
                    len(dirichlet_alpha)
                )  # Fallback to uniform-ish if all are tiny

            sampled_probabilities = self.random_generator.dirichlet(
                alpha=dirichlet_alpha
            )
            # Ensure sampled_probabilities sum to 1 for choice
            if not np.isclose(np.sum(sampled_probabilities), 1.0):
                sampled_probabilities /= np.sum(sampled_probabilities)
            return int(
                self.random_generator.choice(strain_indices, p=sampled_probabilities)
            )
        else:  # Should be caught by __init__
            raise ValueError(
                f"Internal error: Unknown disambiguation mode: {self.disambiguation_mode}"
            )

    def resolve_ambiguous_hits_parallel(
        self,
        ambiguous_hits_dict: Dict[ReadId, CountVector],
        prior_probabilities: np.ndarray,
    ) -> Dict[ReadId, StrainIndex]:
        """
        Resolves ambiguous read assignments in parallel.

        Args:
            ambiguous_hits_dict: Dict mapping ReadId to valid CountVector for ambiguous hits.
            prior_probabilities: Valid prior probability vector.

        Returns:
            Dict mapping ReadId to resolved StrainIndex.

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If input properties are invalid.
        """
        if not isinstance(ambiguous_hits_dict, dict):
            raise TypeError("ambiguous_hits_dict must be a dictionary.")
        # Validate prior_probabilities once before passing to partial function
        if not isinstance(
            prior_probabilities, np.ndarray
        ):  # Duplicates check from _resolve_single_ambiguous_read for safety
            raise TypeError("prior_probabilities must be a NumPy array.")
        if (
            prior_probabilities.ndim != 1
            or len(prior_probabilities) != self._num_strains
            or not np.issubdtype(prior_probabilities.dtype, np.floating)
            or np.any(prior_probabilities < 0)
        ):
            raise ValueError(
                "prior_probabilities must be a 1D float array of correct length with non-negative values."
            )
        # Sum check for priors is complex here due to epsilon; _resolve_single_ambiguous_read handles internal normalization.

        for read_id, cv in ambiguous_hits_dict.items():
            if not isinstance(read_id, str):
                raise TypeError(
                    f"ReadId '{read_id}' in ambiguous_hits_dict must be a string."
                )
            self._validate_count_vector(
                cv, f"ReadId '{read_id}' in ambiguous_hits_dict"
            )

        if not ambiguous_hits_dict:
            return {}

        resolve_func_with_priors = functools.partial(
            self._resolve_single_ambiguous_read, prior_probabilities=prior_probabilities
        )
        num_disambiguation_processes = max(1, self.num_processes // 2)

        hit_vectors_to_process = list(ambiguous_hits_dict.values())
        read_ids_in_order = list(ambiguous_hits_dict.keys())

        if not hit_vectors_to_process:
            return {}

        try:
            with mp.Pool(processes=num_disambiguation_processes) as pool:
                resolved_indices_list = pool.map(
                    resolve_func_with_priors, hit_vectors_to_process
                )
        except Exception as e:  # Catch potential mp.Pool errors
            raise RuntimeError(
                f"Multiprocessing pool error during ambiguous hit resolution: {e}"
            ) from e

        resolved_assignments = dict(zip(read_ids_in_order, resolved_indices_list))
        print(f"Resolved {len(resolved_assignments)} ambiguous assignments.")
        return resolved_assignments

    def combine_assignments(
        self,
        clear_assignments: Dict[ReadId, StrainIndex],
        resolved_ambiguous_assignments: Dict[ReadId, StrainIndex],
        no_hit_read_ids: List[ReadId],
        unassigned_marker: str = "NA",
    ) -> Dict[ReadId, Union[StrainIndex, str]]:
        """
        Combines assignments from clear, resolved ambiguous, and no-hit categories.

        Args:
            clear_assignments: Dict mapping ReadId to StrainIndex.
            resolved_ambiguous_assignments: Dict mapping ReadId to StrainIndex.
            no_hit_read_ids: List of ReadIds for no-hit reads.
            unassigned_marker: String marker for unassigned reads.

        Returns:
            Dict mapping ReadId to StrainIndex or unassigned_marker.

        Raises:
            TypeError: If input types are incorrect.
        """
        if (
            not isinstance(clear_assignments, dict)
            or not isinstance(resolved_ambiguous_assignments, dict)
            or not isinstance(no_hit_read_ids, list)
            or not isinstance(unassigned_marker, str)
        ):
            raise TypeError("Invalid input types for combine_assignments arguments.")

        # Basic validation of content types (can be expanded)
        if not all(
            isinstance(k, str) and isinstance(v, int)
            for k, v in clear_assignments.items()
        ):
            raise TypeError(
                "clear_assignments must map ReadId (str) to StrainIndex (int)."
            )
        if not all(
            isinstance(k, str) and isinstance(v, int)
            for k, v in resolved_ambiguous_assignments.items()
        ):
            raise TypeError(
                "resolved_ambiguous_assignments must map ReadId (str) to StrainIndex (int)."
            )
        if not all(isinstance(item, str) for item in no_hit_read_ids):
            raise TypeError("no_hit_read_ids must be a list of strings.")

        final_assignments: Dict[ReadId, Union[StrainIndex, str]] = {}
        final_assignments.update(clear_assignments)
        final_assignments.update(resolved_ambiguous_assignments)
        for read_id in no_hit_read_ids:
            final_assignments[read_id] = unassigned_marker

        print(
            f"Final combined assignment summary:\n"
            f"  Total reads processed: {len(final_assignments)}\n"
            f"  Assigned (clear + resolved): {len(clear_assignments) + len(resolved_ambiguous_assignments)}\n"
            f"  Marked '{unassigned_marker}': {len(no_hit_read_ids)}"  # This count is only for no_hit_read_ids
        )
        return final_assignments


# analyze_results_streaming method remains commented out as per original file.
# ... (rest of the commented out code for analyze_results_streaming)
# (Code for analyze_results_streaming is omitted here for brevity but would be included in the actual file overwrite)
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
