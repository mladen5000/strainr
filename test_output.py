"""
Pytest unit tests for the AbundanceCalculator class from src.strainr.output.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""
import pytest
from collections import Counter
from typing import List, Dict, Union, Any # Any can be removed if not used

from src.strainr.output import AbundanceCalculator
# Assuming StrainIndex and ReadId might be used if FinalAssignmentsType was more specific
# For now, the tests align with AbundanceCalculator's direct type hints.
# from src.strainr.genomic_types import StrainIndex, ReadId 

# --- Fixtures ---

@pytest.fixture
def strain_names_fixture() -> List[str]:
    """Provides a default list of strain names."""
    return ["StrainA", "StrainB", "StrainC"]

@pytest.fixture
def default_threshold_fixture() -> float:
    """Provides a default abundance threshold for testing."""
    # The class default is 0.001, using a slightly different one for tests
    # can make some thresholding tests clearer.
    return 0.01 

@pytest.fixture
def calculator_fixture(strain_names_fixture: List[str], default_threshold_fixture: float) -> AbundanceCalculator:
    """Provides a default AbundanceCalculator instance."""
    return AbundanceCalculator(
        strain_names=strain_names_fixture,
        abundance_threshold=default_threshold_fixture
    )

@pytest.fixture
def unassigned_marker_fixture() -> str:
    """Provides a consistent unassigned marker for tests."""
    return "NOT_ASSIGNED"

# --- Tests for __init__ ---

def test_init_successful(strain_names_fixture: List[str], default_threshold_fixture: float):
    """Test successful initialization of AbundanceCalculator."""
    calculator = AbundanceCalculator(
        strain_names=strain_names_fixture,
        abundance_threshold=default_threshold_fixture
    )
    assert calculator.strain_names == strain_names_fixture
    assert calculator.abundance_threshold == default_threshold_fixture
    # Note: The current AbundanceCalculator.__init__ (from step 1) does not have
    # explicit validation for empty strain_names or invalid threshold values.
    # If these were added, corresponding tests for ValueError would be needed.

# --- Tests for convert_assignments_to_strain_names ---

def test_convert_assignments_indices_to_names(calculator_fixture: AbundanceCalculator, strain_names_fixture: List[str]):
    read_assignments: Dict[Union[str, int], Union[str, int]] = {
        "read1": 0,  # StrainA
        "read2": 1,  # StrainB
        "read3": 2,  # StrainC
    }
    expected = {"read1": "StrainA", "read2": "StrainB", "read3": "StrainC"}
    assert calculator_fixture.convert_assignments_to_strain_names(read_assignments) == expected

def test_convert_assignments_names_to_names(calculator_fixture: AbundanceCalculator):
    read_assignments: Dict[Union[str, int], Union[str, int]] = {
        "read1": "StrainA",
        "read2": "StrainB",
    }
    expected = {"read1": "StrainA", "read2": "StrainB"}
    assert calculator_fixture.convert_assignments_to_strain_names(read_assignments) == expected

def test_convert_assignments_mixed_indices_and_names(calculator_fixture: AbundanceCalculator):
    read_assignments: Dict[Union[str, int], Union[str, int]] = {
        "read1": 0,          # StrainA
        "read2": "StrainB",
        "read3": 2,          # StrainC
    }
    expected = {"read1": "StrainA", "read2": "StrainB", "read3": "StrainC"}
    assert calculator_fixture.convert_assignments_to_strain_names(read_assignments) == expected

def test_convert_assignments_custom_unassigned_marker(calculator_fixture: AbundanceCalculator, unassigned_marker_fixture: str):
    read_assignments: Dict[Union[str, int], Union[str, int]] = {
        "read1": 0, # StrainA
        "read2": "NonExistentStrain", 
        "read3": 99, # Invalid index
        "read4": unassigned_marker_fixture # Already marked
    }
    expected = {
        "read1": calculator_fixture.strain_names[0], # StrainA
        "read2": unassigned_marker_fixture,
        "read3": unassigned_marker_fixture,
        "read4": unassigned_marker_fixture,
    }
    result = calculator_fixture.convert_assignments_to_strain_names(read_assignments, unassigned_marker=unassigned_marker_fixture)
    assert result == expected

def test_convert_assignments_default_unassigned_marker_na(calculator_fixture: AbundanceCalculator):
    # Default unassigned_marker in the method is "NA"
    read_assignments: Dict[Union[str, int], Union[str, int]] = {
        "read_invalid_idx": 99, 
        "read_unknown_str": "UnknownStrainXYZ"
    }
    expected = {
        "read_invalid_idx": "NA",
        "read_unknown_str": "NA",
    }
    assert calculator_fixture.convert_assignments_to_strain_names(read_assignments) == expected


def test_convert_assignments_empty_input(calculator_fixture: AbundanceCalculator):
    assert calculator_fixture.convert_assignments_to_strain_names({}) == {}

def test_convert_assignments_non_str_read_ids_converted(calculator_fixture: AbundanceCalculator, strain_names_fixture: List[str]):
    read_assignments: Dict[Union[str, int], Union[str, int]] = {
        1001: 0, # read_id is int -> "StrainA"
        "read_str_002": strain_names_fixture[1] # "StrainB"
    }
    expected = {"1001": strain_names_fixture[0], "read_str_002": strain_names_fixture[1]}
    assert calculator_fixture.convert_assignments_to_strain_names(read_assignments) == expected

# --- Tests for calculate_raw_abundances ---

def test_calculate_raw_abundances_empty(calculator_fixture: AbundanceCalculator):
    assert calculator_fixture.calculate_raw_abundances({}) == Counter()

def test_calculate_raw_abundances_simple_counts(calculator_fixture: AbundanceCalculator):
    named_assignments = {"r1": "StrainA", "r2": "StrainB", "r3": "StrainA", "r4": "StrainC"}
    expected = Counter({"StrainA": 2, "StrainB": 1, "StrainC": 1})
    assert calculator_fixture.calculate_raw_abundances(named_assignments) == expected

def test_calculate_raw_abundances_exclude_unassigned_true_default(
    calculator_fixture: AbundanceCalculator, 
    unassigned_marker_fixture: str
):
    named_assignments = {
        "r1": "StrainA", "r2": unassigned_marker_fixture, 
        "r3": "StrainB", "r4": unassigned_marker_fixture
    }
    expected = Counter({"StrainA": 1, "StrainB": 1})
    # exclude_unassigned=True is default, using custom marker
    assert calculator_fixture.calculate_raw_abundances(named_assignments, unassigned_marker=unassigned_marker_fixture) == expected

def test_calculate_raw_abundances_exclude_unassigned_false(
    calculator_fixture: AbundanceCalculator, 
    unassigned_marker_fixture: str
):
    named_assignments = {
        "r1": "StrainA", "r2": unassigned_marker_fixture, 
        "r3": "StrainB", "r4": unassigned_marker_fixture
    }
    expected = Counter({"StrainA": 1, "StrainB": 1, unassigned_marker_fixture: 2})
    assert calculator_fixture.calculate_raw_abundances(
        named_assignments, exclude_unassigned=False, unassigned_marker=unassigned_marker_fixture
    ) == expected

# --- Tests for calculate_relative_abundances ---

def test_calculate_relative_abundances_basic(calculator_fixture: AbundanceCalculator):
    raw_abundances = Counter({"StrainA": 60, "StrainB": 30, "StrainC": 10}) # Total 100
    expected = {"StrainA": 0.6, "StrainB": 0.3, "StrainC": 0.1}
    result = calculator_fixture.calculate_relative_abundances(raw_abundances)
    assert sum(result.values()) == pytest.approx(1.0)
    for strain, rel_abun in expected.items():
        assert result[strain] == pytest.approx(rel_abun)

def test_calculate_relative_abundances_empty_input(calculator_fixture: AbundanceCalculator):
    assert calculator_fixture.calculate_relative_abundances(Counter()) == {}

def test_calculate_relative_abundances_all_zero_counts(calculator_fixture: AbundanceCalculator):
    raw_abundances = Counter({"StrainA": 0, "StrainB": 0, "StrainC": 0})
    # Implementation returns {} if total_reads is 0
    assert calculator_fixture.calculate_relative_abundances(raw_abundances) == {}

# --- Tests for apply_threshold_and_format ---

def test_apply_threshold_and_format_filtering_and_sorting(
    calculator_fixture: AbundanceCalculator # default_threshold_fixture is 0.01
):
    relative_abundances = {"StrainC": 0.3, "StrainA": 0.005, "StrainB": 0.695}
    # StrainA (0.005) is below threshold 0.01
    
    # Test sort_by_abundance=True (default)
    expected_sorted_abundance = {"StrainB": 0.695, "StrainC": 0.3} 
    result_abundance = calculator_fixture.apply_threshold_and_format(relative_abundances, sort_by_abundance=True)
    assert list(result_abundance.items()) == list(expected_sorted_abundance.items())

    # Test sort_by_abundance=False (sorts by name)
    # Expected: StrainB: 0.695, StrainC: 0.3 (already sorted by name after filtering)
    expected_sorted_name = {"StrainB": 0.695, "StrainC": 0.3} 
    result_name = calculator_fixture.apply_threshold_and_format(relative_abundances, sort_by_abundance=False)
    # The implementation sorts by name if sort_by_abundance is False
    # For {"StrainB": 0.695, "StrainC": 0.3}, sorted by name is already this order.
    assert list(result_name.items()) == list(expected_sorted_name.items())


def test_apply_threshold_and_format_all_below_threshold(calculator_fixture: AbundanceCalculator):
    # calculator_fixture has threshold 0.01
    relative_abundances = {"StrainA": 0.001, "StrainB": 0.002, "StrainC": 0.0005}
    assert calculator_fixture.apply_threshold_and_format(relative_abundances) == {}

def test_apply_threshold_and_format_empty_input(calculator_fixture: AbundanceCalculator):
    assert calculator_fixture.apply_threshold_and_format({}) == {}

def test_apply_threshold_and_format_exact_threshold_value(calculator_fixture: AbundanceCalculator):
    # calculator_fixture has threshold 0.01
    relative_abundances = {"StrainA": 0.01, "StrainB": 0.99}
    expected = {"StrainB": 0.99, "StrainA": 0.01} # StrainA is at threshold, should be included
    result = calculator_fixture.apply_threshold_and_format(relative_abundances, sort_by_abundance=True)
    assert result == expected

# --- Tests for generate_report_string ---

def test_generate_report_string_basic_formatting(calculator_fixture: AbundanceCalculator):
    final_abundances = {"StrainA": 0.75126, "StrainC": 0.1000, "StrainB": 0.14874}
    # Note: generate_report_string does not sort, it uses the order from final_abundances.
    # The apply_threshold_and_format method is responsible for sorting before this.
    expected_report_lines = [
        "StrainA: 0.7513", # .4f rounding
        "StrainC: 0.1000",
        "StrainB: 0.1487" 
    ]
    expected_report = "\n".join(expected_report_lines)
    assert calculator_fixture.generate_report_string(final_abundances) == expected_report

def test_generate_report_string_empty_input(calculator_fixture: AbundanceCalculator):
    expected_report = "No strains found above the abundance threshold."
    assert calculator_fixture.generate_report_string({}) == expected_report

def test_generate_report_string_single_strain(calculator_fixture: AbundanceCalculator):
    final_abundances = {"StrainX": 1.0}
    expected_report = "StrainX: 1.0000"
    assert calculator_fixture.generate_report_string(final_abundances) == expected_report

```
