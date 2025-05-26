# =============================================================================
# strainr/output.py - Refactored output and reporting
# =============================================================================
"""
Output formatting and abundance calculation functionality.

Output formatting and abundance calculation functionality.
"""

# import pathlib # Not used in this file
from collections import Counter  # defaultdict was not used
from typing import Dict, Union, List  # Removed Any

# import pandas as pd # Not used in this file


class AbundanceCalculator:
    """
    Calculator for strain abundance analysis and reporting.

    This class handles conversion of read assignments to relative abundances,
    normalization, thresholding, and output formatting.
    """

    def __init__(self, strain_names: List[str], abundance_threshold: float = 0.001):
        """
        Initialize the abundance calculator.

        Args:
            strain_names: List of all strain identifiers
            abundance_threshold: Minimum relative abundance for reporting
        """
        self.strain_names = strain_names
        self.abundance_threshold = abundance_threshold

    def convert_assignments_to_strain_names(
        self,
        read_assignments: Dict[Union[str, int], Union[str, int]],
        unassigned_marker: str = "NA",
    ) -> Dict[str, str]:
        """
        Converts read assignments to strain names.

        Handles read assignments that might use strain indices instead of names.
        Allows for a custom marker for unassigned reads.

        Args:
            read_assignments: Dictionary mapping read IDs to strain IDs (or indices).
            unassigned_marker: Marker to use for reads not assigned to any strain.

        Returns:
            Dictionary mapping read IDs to strain names.
        """
        named_assignments = {}
        for read_id, strain_id_or_idx in read_assignments.items():
            if isinstance(strain_id_or_idx, int):
                if 0 <= strain_id_or_idx < len(self.strain_names):
                    named_assignments[str(read_id)] = self.strain_names[
                        strain_id_or_idx
                    ]
                else:
                    # Handle potential index out of bounds if necessary,
                    # or assume valid indices based on upstream logic.
                    # For now, let's treat invalid indices as unassigned.
                    named_assignments[str(read_id)] = unassigned_marker
            elif isinstance(strain_id_or_idx, str):
                if strain_id_or_idx in self.strain_names:
                    named_assignments[str(read_id)] = strain_id_or_idx
                else:  # Handles if strain_id_or_idx is already unassigned_marker or other string
                    named_assignments[str(read_id)] = unassigned_marker
            else:  # if strain_id_or_idx is neither int nor str, or some other unexpected type
                named_assignments[str(read_id)] = unassigned_marker
        return named_assignments

    def calculate_raw_abundances(
        self,
        named_assignments: Dict[str, str],
        exclude_unassigned: bool = True,
        unassigned_marker: str = "NA",
    ) -> Counter[str]:
        """
        Calculates raw counts of reads assigned to each strain.

        Args:
            named_assignments: Dictionary mapping read IDs to strain names.
            exclude_unassigned: Whether to exclude unassigned reads from counts.
            unassigned_marker: Marker used for unassigned reads.

        Returns:
            A Counter object with raw counts for each strain.
        """
        if exclude_unassigned:
            return Counter(
                strain_name
                for strain_name in named_assignments.values()
                if strain_name != unassigned_marker
            )
        else:
            return Counter(named_assignments.values())

    def calculate_relative_abundances(
        self, raw_abundances: Counter[str]
    ) -> Dict[str, float]:
        """
        Converts raw counts to relative frequencies.

        Args:
            raw_abundances: Counter object with raw counts for each strain.

        Returns:
            Dictionary mapping strain names to their relative abundances.
            Returns an empty dict if total_reads is zero.
        """
        total_reads = sum(raw_abundances.values())
        if total_reads == 0:
            return {}

        relative_abundances = {
            strain: count / total_reads for strain, count in raw_abundances.items()
        }
        return relative_abundances

    def apply_threshold_and_format(
        self, relative_abundances: Dict[str, float], sort_by_abundance: bool = True
    ) -> Dict[str, float]:
        """
        Filters strains below the abundance threshold and optionally sorts results.

        Args:
            relative_abundances: Dictionary of strain names to relative abundances.
            sort_by_abundance: Whether to sort the output by abundance (descending).

        Returns:
            A dictionary of strain names to relative abundances, filtered and sorted.
        """
        filtered_abundances = {
            strain: abundance
            for strain, abundance in relative_abundances.items()
            if abundance >= self.abundance_threshold
        }

        if sort_by_abundance:
            return dict(
                sorted(
                    filtered_abundances.items(), key=lambda item: item[1], reverse=True
                )
            )
        else:
            # If not sorting by abundance, sort by strain name for consistent output
            return dict(sorted(filtered_abundances.items(), key=lambda item: item[0]))

    def generate_report_string(self, final_abundances: Dict[str, float]) -> str:
        """
        Generates a formatted string for reporting abundances.

        Args:
            final_abundances: Dictionary of strain names to their final abundances.

        Returns:
            A string reporting the abundances, typically one strain per line.
        """
        if not final_abundances:
            return "No strains found above the abundance threshold."

        report_lines = [
            f"{strain}: {abundance:.4f}"
            for strain, abundance in final_abundances.items()
        ]
        return "\n".join(report_lines)
