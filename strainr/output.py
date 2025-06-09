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
