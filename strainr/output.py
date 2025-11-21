# =============================================================================
# strainr/output.py - Refactored output and reporting
# =============================================================================
"""
Output formatting and abundance calculation functionality.
"""

from collections import Counter
from typing import (  # Any for potential complex dicts, though striving for specifics
    Dict,
    List,
    Union,
)

# Assuming these types are available from a central genomic_types module
from .genomic_types import ReadId, StrainIndex
# For now, defining them locally if not, or assuming basic types.


class AbundanceCalculator:
    """
    Calculator for strain abundance analysis and reporting.

    Handles conversion of read assignments to relative abundances,

    normalization, thresholding, and output formatting.
    """

    def __init__(self, strain_names: List[str], abundance_threshold: float = 0.001):
        """
        Initialize the abundance calculator.

        Args:
            strain_names: List of unique, non-empty strain identifiers.
            abundance_threshold: Minimum relative abundance (0.0 to <1.0) for reporting.

        Raises:
            TypeError: If `strain_names` is not a list of strings or if `abundance_threshold` is not a float.
            ValueError: If `strain_names` is empty, contains duplicates, or has empty strings.
                        If `abundance_threshold` is not in the range [0.0, 1.0).
        """
        if not isinstance(strain_names, list) or not all(
            isinstance(s, str) for s in strain_names
        ):
            raise TypeError("strain_names must be a list of strings.")
        if not strain_names:
            raise ValueError("strain_names cannot be empty.")
        if len(set(strain_names)) != len(strain_names):
            raise ValueError("strain_names must be unique.")
        if any(not s for s in strain_names):
            raise ValueError("strain_names must not contain empty strings.")

        if not isinstance(abundance_threshold, float):
            raise TypeError("abundance_threshold must be a float.")
        if not (0.0 <= abundance_threshold < 1.0):
            raise ValueError(
                "abundance_threshold must be between 0.0 (inclusive) and 1.0 (exclusive)."
            )

        self.strain_names: List[str] = strain_names
        self.abundance_threshold: float = abundance_threshold
        self._num_strains: int = len(strain_names)

    def convert_assignments_to_strain_names(
        self,
        read_assignments: Dict[
            Union[ReadId, int], Union[StrainIndex, str]
        ],  # Keys can be int initially
        unassigned_marker: str = "NA",
    ) -> Dict[ReadId, str]:
        """
        Converts read assignments (potentially with integer keys or strain indices)
        to a dictionary mapping string ReadIds to strain names or an unassigned marker.

        Args:
            read_assignments: Dictionary mapping read IDs (str or int) to strain IDs
                              (StrainIndex as int, strain name as str, or unassigned_marker as str).
            unassigned_marker: Marker string for unassigned reads. Must be non-empty.

        Returns:
            Dictionary mapping string ReadIds to strain names or the unassigned_marker.

        Raises:
            TypeError: If input types are invalid (e.g., `unassigned_marker` not str,
                       assignment values of unexpected types).
            ValueError: If `read_assignments` keys are empty strings after conversion,
                        if `unassigned_marker` is empty, if a StrainIndex is out of bounds,
                        or if a string assignment value is not in `self.strain_names` and
                        not equal to `unassigned_marker`.
        """
        if not isinstance(read_assignments, dict):
            raise TypeError("read_assignments must be a dictionary.")
        if not isinstance(unassigned_marker, str) or not unassigned_marker:
            raise ValueError("unassigned_marker must be a non-empty string.")

        named_assignments: Dict[ReadId, str] = {}
        for read_id_orig, strain_id_or_idx in read_assignments.items():
            read_id_str = str(read_id_orig)
            if not read_id_str:
                raise ValueError(
                    "ReadId keys, after string conversion, must not be empty."
                )

            if isinstance(strain_id_or_idx, int):  # StrainIndex
                if 0 <= strain_id_or_idx < self._num_strains:
                    named_assignments[read_id_str] = self.strain_names[strain_id_or_idx]

                else:
                    raise ValueError(
                        f"StrainIndex {strain_id_or_idx} for ReadId '{read_id_str}' is out of bounds "
                        f"[0, {self._num_strains - 1}]."
                    )
            elif isinstance(strain_id_or_idx, str):
                if strain_id_or_idx in self.strain_names:
                    named_assignments[read_id_str] = strain_id_or_idx
                elif strain_id_or_idx == unassigned_marker:
                    named_assignments[read_id_str] = unassigned_marker
                else:  # Unknown string assignment
                    raise ValueError(
                        f"String assignment '{strain_id_or_idx}' for ReadId '{read_id_str}' is not a "
                        f"recognized strain name or the unassigned_marker ('{unassigned_marker}')."
                    )
            else:
                raise TypeError(
                    f"Invalid assignment type for ReadId '{read_id_str}': {type(strain_id_or_idx)}. "
                    "Expected int (StrainIndex) or str (strain name/unassigned_marker)."
                )

        return named_assignments

    def calculate_raw_abundances(
        self,
        named_assignments: Dict[ReadId, str],
        exclude_unassigned: bool = True,
        unassigned_marker: str = "NA",
    ) -> Counter[str]:
        """
        Calculates raw counts of reads assigned to each strain name or marker.

        Args:
            named_assignments: Dictionary mapping ReadId (str) to strain name (str) or unassigned_marker (str).
            exclude_unassigned: If True, counts for `unassigned_marker` are excluded.
            unassigned_marker: Marker string for unassigned reads. Must be non-empty.

        Returns:
            A Counter object with raw counts for each strain name/marker.

        Raises:
            TypeError: If input types are invalid.
            ValueError: If keys/values in `named_assignments` are empty strings or if `unassigned_marker` is empty.
        """
        if not isinstance(named_assignments, dict):
            raise TypeError("named_assignments must be a dictionary.")
        if (
            not isinstance(unassigned_marker, str) or not unassigned_marker
        ):  # Allow marker to be empty if not excluding
            if exclude_unassigned:  # Only critical if we need to compare against it
                raise ValueError(
                    "unassigned_marker must be a non-empty string when exclude_unassigned is True."
                )
            elif not unassigned_marker:  # if it's empty and we are NOT excluding, it can lead to issues if "" is a strain name
                pass  # Allow empty marker if not actively used for exclusion, though risky. Best to enforce non-empty.

        for read_id, name_or_marker in named_assignments.items():
            if not isinstance(read_id, str) or not read_id:
                raise ValueError(
                    "All ReadId keys in named_assignments must be non-empty strings."
                )
            if not isinstance(name_or_marker, str) or not name_or_marker:
                raise ValueError(
                    f"Assigned name/marker for ReadId '{read_id}' must be a non-empty string."
                )

        if exclude_unassigned:
            return Counter(
                name for name in named_assignments.values() if name != unassigned_marker
            )
        else:
            return Counter(named_assignments.values())

    def calculate_relative_abundances(
        self, raw_abundances: Counter[str]
    ) -> Dict[str, float]:
        """
        Converts raw counts to relative frequencies (0.0 to 1.0).
        """
        if not isinstance(raw_abundances, Counter):
            raise TypeError("raw_abundances must be a collections.Counter.")
        for strain_name, count in raw_abundances.items():
            if not isinstance(strain_name, str) or not strain_name:
                raise ValueError(
                    "Strain names in raw_abundances must be non-empty strings."
                )
            if not isinstance(count, (int, float)) or count < 0:
                raise ValueError(
                    f"Count for strain '{strain_name}' must be a non-negative number, got {count}."
                )

        total_reads = float(sum(raw_abundances.values()))  # Ensure float for division
        if total_reads == 0:
            return {}

        relative_abundances: Dict[str, float] = {}
        for strain, count in raw_abundances.items():
            relative_abundances[strain] = count / total_reads

        return relative_abundances

    def normalize_by_genome_length(
        self, raw_abundances: Counter[str], genome_lengths: Dict[str, int]
    ) -> Counter[str]:
        """
        Normalize read counts by genome length to correct for size bias.

        Longer genomes naturally have more k-mers and thus more read assignments.
        This normalization divides read count by genome length before calculating
        relative abundances, providing more accurate quantification.

        Scientific rationale:
        - A 5 Mbp genome appears 2x more abundant than a 2.5 Mbp genome at equal
          cell counts
        - RPKM-like normalization: Reads Per Kilobase per Million mapped reads
        - Essential for comparing strains with different genome sizes

        Args:
            raw_abundances: Raw read counts per strain
            genome_lengths: Estimated genome lengths (bp) per strain

        Returns:
            Length-normalized counts (pseudo-counts as floats)
        """
        if not isinstance(raw_abundances, Counter):
            raise TypeError("raw_abundances must be a collections.Counter.")
        if not isinstance(genome_lengths, dict):
            raise TypeError("genome_lengths must be a dictionary.")

        normalized = Counter()
        for strain, count in raw_abundances.items():
            if strain in genome_lengths and genome_lengths[strain] > 0:
                # Normalize to reads per kilobase
                normalized[strain] = (count / genome_lengths[strain]) * 1000
            else:
                # No length info, keep original count
                normalized[strain] = float(count)

        return normalized

    def apply_threshold_and_format(
        self, relative_abundances: Dict[str, float], sort_by_abundance: bool = True
    ) -> Dict[str, float]:
        """
        Filters strains below `self.abundance_threshold` and optionally sorts.
        """
        if not isinstance(relative_abundances, dict):
            raise TypeError("relative_abundances must be a dictionary.")

        filtered_abundances: Dict[str, float] = {}
        for strain, abundance in relative_abundances.items():
            if not isinstance(strain, str) or not strain:
                raise ValueError(
                    "Strain names in relative_abundances must be non-empty strings."
                )
            if not isinstance(abundance, float):
                raise TypeError(
                    f"Abundance for strain '{strain}' must be a float, got {type(abundance)}."
                )
            if not (
                -1e-9 <= abundance <= 1.0 + 1e-9
            ):  # Allow for minor float inaccuracies
                raise ValueError(
                    f"Abundance for strain '{strain}' ({abundance}) must be between 0.0 and 1.0."
                )

            if abundance >= self.abundance_threshold:
                filtered_abundances[strain] = abundance

        if not filtered_abundances:
            return {}

        key_sort_func = (
            (lambda item: item[1]) if sort_by_abundance else (lambda item: item[0])
        )
        reverse_sort = sort_by_abundance  # Descending for abundance, ascending for name (default for strings)

        return dict(
            sorted(filtered_abundances.items(), key=key_sort_func, reverse=reverse_sort)
        )

    def generate_report_string(self, final_abundances: Dict[str, float]) -> str:
        """
        Generates a formatted string reporting abundances as percentages.
        Example: "StrainA: 75.13%\nStrainB: 24.87%"
        """
        if not isinstance(final_abundances, dict):
            raise TypeError("final_abundances must be a dictionary.")

        report_lines: List[str] = []
        if not final_abundances:
            return "No strains found above the abundance threshold."

        for strain, abundance in final_abundances.items():
            if not isinstance(strain, str) or not strain:
                raise ValueError(
                    "Strain names in final_abundances must be non-empty strings."
                )
            if not isinstance(abundance, float):
                raise TypeError(
                    f"Abundance for strain '{strain}' must be a float, got {type(abundance)}."
                )
            # Formatting as percentage with 2 decimal places
            report_lines.append(f"{strain}: {abundance * 100:.2f}%")

        return "\n".join(report_lines)


def calculate_shannon_diversity(abundances: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate Shannon diversity index and evenness for microbial community.

    Shannon diversity (H') quantifies community diversity accounting for both
    richness (number of species) and evenness (distribution of abundances).

    H' = -Σ(p_i * ln(p_i))
    where p_i is the relative abundance of strain i

    Evenness (J') = H' / ln(S)
    where S is the number of strains (species richness)

    Args:
        abundances: Dictionary of strain names to relative abundances (must sum to ~1.0)

    Returns:
        Dictionary with keys:
        - 'shannon_index': Shannon diversity index (H')
        - 'richness': Number of strains present
        - 'evenness': Pielou's evenness index (J')
        - 'max_diversity': Maximum possible diversity (ln(S))

    Scientific interpretation:
    - H' = 0: Only one strain present (no diversity)
    - H' increases with more strains and more even distribution
    - J' = 1.0: Perfectly even community (all strains equal abundance)
    - J' → 0: Dominated by one strain
    - Higher diversity suggests healthier/more stable communities
    """
    import numpy as np

    if not abundances:
        return {
            'shannon_index': 0.0,
            'richness': 0,
            'evenness': 0.0,
            'max_diversity': 0.0
        }

    # Filter out zero abundances
    positive_abundances = {s: a for s, a in abundances.items() if a > 0}

    if not positive_abundances:
        return {
            'shannon_index': 0.0,
            'richness': 0,
            'evenness': 0.0,
            'max_diversity': 0.0
        }

    richness = len(positive_abundances)
    abundances_array = np.array(list(positive_abundances.values()))

    # Normalize to ensure sum = 1.0 (handle floating point errors)
    total = np.sum(abundances_array)
    if total > 0:
        abundances_array = abundances_array / total

    # Calculate Shannon index: H' = -Σ(p_i * ln(p_i))
    # Use np.where to avoid log(0)
    shannon_index = -np.sum(
        abundances_array * np.log(np.where(abundances_array > 0, abundances_array, 1))
    )

    # Calculate maximum possible diversity
    max_diversity = np.log(richness) if richness > 1 else 0.0

    # Calculate evenness (Pielou's J')
    evenness = shannon_index / max_diversity if max_diversity > 0 else 0.0

    return {
        'shannon_index': float(shannon_index),
        'richness': int(richness),
        'evenness': float(evenness),
        'max_diversity': float(max_diversity)
    }


def calculate_assignment_confidence(
    strain_counts: "np.ndarray", kmer_specificity_scores: List[int]
) -> float:
    """
    Calculate statistical confidence score for a strain assignment.

    Confidence is based on:
    1. Number of unique k-mers supporting the assignment
    2. Specificity of those k-mers (strain-unique k-mers = high confidence)
    3. Relative strength vs. other strains

    Args:
        strain_counts: Array of k-mer hit counts per strain
        kmer_specificity_scores: Specificity of each k-mer that was observed
                                 (1 = unique to one strain, higher = less specific)

    Returns:
        Confidence score between 0.0 (low) and 1.0 (high)

    Scientific rationale:
    - Assignments supported by many strain-unique k-mers are highly confident
    - Assignments based only on k-mers shared across many strains are less confident
    - This helps identify ambiguous/uncertain classifications
    """
    import numpy as np

    if len(strain_counts) == 0 or np.sum(strain_counts) == 0:
        return 0.0  # No evidence

    # Get the maximum strain count (winning strain)
    max_count = np.max(strain_counts)
    if max_count == 0:
        return 0.0

    # Factor 1: Relative strength (how much better than 2nd place?)
    sorted_counts = np.sort(strain_counts)[::-1]
    second_best = sorted_counts[1] if len(sorted_counts) > 1 else 0
    relative_strength = (max_count - second_best) / max_count if max_count > 0 else 0

    # Factor 2: K-mer specificity (prefer strain-unique k-mers)
    if kmer_specificity_scores:
        avg_specificity = np.mean(kmer_specificity_scores)
        # Normalize: specificity=1 (unique) gives 1.0, higher values give lower scores
        # Using inverse: 1/avg_specificity, capped at 1.0
        specificity_score = min(1.0, 1.0 / avg_specificity) if avg_specificity > 0 else 0
    else:
        specificity_score = 0.5  # Unknown, assume moderate

    # Factor 3: Number of supporting k-mers (more evidence = higher confidence)
    # Log-scale normalization: 10 k-mers = ~0.5, 100 k-mers = ~0.67, 1000 k-mers = ~0.75
    num_kmers = max_count
    evidence_score = min(1.0, np.log10(num_kmers + 1) / 3.0)

    # Combined confidence (weighted average)
    confidence = (
        0.4 * relative_strength +
        0.4 * specificity_score +
        0.2 * evidence_score
    )

    return float(np.clip(confidence, 0.0, 1.0))


def filter_by_coverage_thresholds(
    read_assignments: Dict[str, int],
    strain_hit_counts: Dict[str, int],
    strain_genome_lengths: Dict[str, int],
    strain_names: List[str],
    min_kmer_count: int = 10,
    min_genome_fraction: float = 0.01,
    unassigned_marker: int = -1
) -> Dict[str, int]:
    """
    Filter strain assignments based on minimum coverage thresholds.

    Removes strains that don't meet minimum k-mer evidence requirements.
    This prevents spurious calls from random k-mer matches or contamination.

    Args:
        read_assignments: Dictionary mapping read IDs to strain indices
        strain_hit_counts: Number of unique k-mers observed per strain
        strain_genome_lengths: Estimated genome length (in k-mers) per strain
        strain_names: List of strain names for index-to-name mapping
        min_kmer_count: Minimum number of unique k-mers (absolute threshold)
        min_genome_fraction: Minimum fraction of genome k-mers observed (0.0-1.0)
        unassigned_marker: Strain index value for unassigned reads (default: -1)

    Returns:
        Filtered read_assignments with low-coverage strains marked as unassigned

    Raises:
        TypeError: If input types are incorrect
        ValueError: If thresholds are invalid

    Scientific rationale:
    - Random k-mer matches occur at low frequency
    - True presence requires substantial k-mer evidence
    - Default: ≥10 k-mers AND ≥1% genome coverage
    - Prevents false positives from contamination or sequencing errors
    """
    # Input validation
    if not isinstance(read_assignments, dict):
        raise TypeError("read_assignments must be a dictionary")
    if not isinstance(strain_hit_counts, dict):
        raise TypeError("strain_hit_counts must be a dictionary")
    if not isinstance(strain_genome_lengths, dict):
        raise TypeError("strain_genome_lengths must be a dictionary")
    if not isinstance(strain_names, list):
        raise TypeError("strain_names must be a list")
    if min_kmer_count < 0:
        raise ValueError("min_kmer_count must be non-negative")
    if not (0.0 <= min_genome_fraction <= 1.0):
        raise ValueError("min_genome_fraction must be between 0.0 and 1.0")

    # Build set of strain indices that meet the thresholds
    valid_strain_indices = set()

    # Identify strains meeting both thresholds
    for strain_idx, strain_name in enumerate(strain_names):
        kmer_count = strain_hit_counts.get(strain_name, 0)
        genome_length = strain_genome_lengths.get(strain_name, 0)

        # Check absolute k-mer threshold
        meets_count_threshold = kmer_count >= min_kmer_count

        # Check relative genome coverage threshold
        if genome_length > 0:
            coverage_fraction = kmer_count / genome_length
            meets_coverage_threshold = coverage_fraction >= min_genome_fraction
        else:
            # If genome length unknown, only use absolute count threshold
            meets_coverage_threshold = True

        # Keep strain if it meets BOTH thresholds
        if meets_count_threshold and meets_coverage_threshold:
            valid_strain_indices.add(strain_idx)

    # Filter read assignments: reassign reads to unassigned if strain doesn't meet thresholds
    filtered_assignments = {}
    for read_id, strain_idx in read_assignments.items():
        if strain_idx in valid_strain_indices:
            # Strain meets thresholds, keep assignment
            filtered_assignments[read_id] = strain_idx
        else:
            # Strain doesn't meet thresholds, mark as unassigned
            filtered_assignments[read_id] = unassigned_marker

    return filtered_assignments
