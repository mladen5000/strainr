"""
Pytest unit tests for the AbundanceCalculator class from src.strainr.output.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""
import pytest
from collections import Counter
from typing import List, Dict, Union, Any 

from src.strainr.output import AbundanceCalculator, ReadId, StrainIndex


# --- Fixtures ---

@pytest.fixture
def strain_names_fixture() -> List[str]:
    return ["StrainA", "StrainB", "StrainC"]

@pytest.fixture
def default_threshold_fixture() -> float:
    return 0.01 

@pytest.fixture
def calculator_fixture(strain_names_fixture: List[str], default_threshold_fixture: float) -> AbundanceCalculator:
    return AbundanceCalculator(
        strain_names=strain_names_fixture,
        abundance_threshold=default_threshold_fixture
    )

@pytest.fixture
def unassigned_marker_fixture() -> str:
    return "NOT_ASSIGNED_TEST" # Using a distinct marker

# --- Tests for __init__ ---

def test_init_successful(strain_names_fixture: List[str], default_threshold_fixture: float):
    calc = AbundanceCalculator(strain_names_fixture, default_threshold_fixture)
    assert calc.strain_names == strain_names_fixture
    assert calc.abundance_threshold == default_threshold_fixture

def test_init_invalid_strain_names_type():
    with pytest.raises(TypeError, match="strain_names must be a list of strings."):
        AbundanceCalculator([1, 2, 3], 0.01) # type: ignore
    with pytest.raises(TypeError, match="strain_names must be a list of strings."):
        AbundanceCalculator(["StrainA", None], 0.01) # type: ignore

def test_init_invalid_strain_names_content():
    with pytest.raises(ValueError, match="strain_names cannot be empty."):
        AbundanceCalculator([], 0.01)
    with pytest.raises(ValueError, match="strain_names must not contain empty strings."):
        AbundanceCalculator(["StrainA", ""], 0.01)
    with pytest.raises(ValueError, match="strain_names must be unique."):
        AbundanceCalculator(["StrainA", "StrainA"], 0.01)

def test_init_invalid_abundance_threshold_type():
    with pytest.raises(TypeError, match="abundance_threshold must be a float."):
        AbundanceCalculator(["StrainA"], "0.01") # type: ignore

def test_init_invalid_abundance_threshold_range(strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="abundance_threshold must be between 0.0 .* and 1.0"):
        AbundanceCalculator(strain_names_fixture, -0.1)
    with pytest.raises(ValueError, match="abundance_threshold must be between 0.0 .* and 1.0"):
        AbundanceCalculator(strain_names_fixture, 1.0)
    with pytest.raises(ValueError, match="abundance_threshold must be between 0.0 .* and 1.0"):
        AbundanceCalculator(strain_names_fixture, 1.1)


# --- Tests for convert_assignments_to_strain_names ---
# (Existing tests for happy paths are good, adding error condition tests)

def test_convert_assignments_invalid_read_id_keys(calculator_fixture: AbundanceCalculator):
    with pytest.raises(ValueError, match="ReadId keys, after string conversion, must not be empty."):
        calculator_fixture.convert_assignments_to_strain_names({"": 0})

def test_convert_assignments_invalid_strain_index_value(calculator_fixture: AbundanceCalculator, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    with pytest.raises(ValueError, match=f"StrainIndex {n_strains} for ReadId 'read1' is out of bounds"):
        calculator_fixture.convert_assignments_to_strain_names({"read1": n_strains})
    with pytest.raises(ValueError, match="StrainIndex -1 for ReadId 'read1' is out of bounds"):
        calculator_fixture.convert_assignments_to_strain_names({"read1": -1})

def test_convert_assignments_unknown_strain_name_value(calculator_fixture: AbundanceCalculator, unassigned_marker_fixture: str):
    unknown_name = "UnknownStrain"
    with pytest.raises(ValueError, match=f"String assignment '{unknown_name}' for ReadId 'read1' is not a recognized strain name or the unassigned_marker"):
        calculator_fixture.convert_assignments_to_strain_names({"read1": unknown_name}, unassigned_marker=unassigned_marker_fixture)

def test_convert_assignments_invalid_assignment_value_type(calculator_fixture: AbundanceCalculator):
    with pytest.raises(TypeError, match="Invalid assignment type for ReadId 'read1'"):
        calculator_fixture.convert_assignments_to_strain_names({"read1": [0]}) # type: ignore

def test_convert_assignments_invalid_unassigned_marker(calculator_fixture: AbundanceCalculator):
    with pytest.raises(ValueError, match="unassigned_marker must be a non-empty string."):
        calculator_fixture.convert_assignments_to_strain_names({"read1": 0}, unassigned_marker="")

# Test to ensure custom marker works when it's also a strain name (should prioritize marker if assignment is not direct strain name)
def test_convert_assignments_marker_same_as_strain_name(strain_names_fixture: List[str], unassigned_marker_fixture: str):
    # If unassigned_marker is e.g. "StrainA"
    calc = AbundanceCalculator(strain_names_fixture + [unassigned_marker_fixture], 0.01)
    assignments = {"read1": 0, "read2": unassigned_marker_fixture, "read3": "Unknown"}
    
    # If "Unknown" is assigned, it should become unassigned_marker_fixture
    # If assignment value *is* unassigned_marker_fixture, it remains that.
    expected = {
        "read1": strain_names_fixture[0], # StrainA
        "read2": unassigned_marker_fixture, # Explicitly assigned to the marker
        "read3": unassigned_marker_fixture  # Unknown, becomes marker
    }
    # The method will treat "Unknown" as needing to become unassigned_marker_fixture.
    # If an assignment value *is* unassigned_marker_fixture, it's kept as such.
    with pytest.raises(ValueError, match=f"String assignment 'Unknown' for ReadId 'read3' is not a recognized strain name or the unassigned_marker"):
        calc.convert_assignments_to_strain_names(assignments, unassigned_marker=unassigned_marker_fixture)
    
    # Correct test: if a strain name happens to be the marker, direct assignments to it are fine.
    # Unknowns become the marker.
    assignments_corrected = {"read1": 0, "read2": unassigned_marker_fixture, "read3": "StrainThatDoesNotExist"}
    expected_corrected = {
        "read1": strain_names_fixture[0],
        "read2": unassigned_marker_fixture,
        "read3": unassigned_marker_fixture,
    }
    # This needs the unassigned_marker to NOT be in strain_names for the error to trigger as "unknown"
    # The current `convert_assignments_to_strain_names` raises ValueError if string is not in strain_names AND not unassigned_marker.
    # This behavior is now stricter. Original test logic for this case was based on silent conversion.

# --- Tests for calculate_raw_abundances ---
def test_calculate_raw_abundances_invalid_keys_values(calculator_fixture: AbundanceCalculator):
    with pytest.raises(ValueError, match="All ReadId keys in named_assignments must be non-empty strings."):
        calculator_fixture.calculate_raw_abundances({"": "StrainA"})
    with pytest.raises(ValueError, match="Assigned name/marker for ReadId 'read1' must be a non-empty string."):
        calculator_fixture.calculate_raw_abundances({"read1": ""})

def test_calculate_raw_abundances_invalid_marker(calculator_fixture: AbundanceCalculator):
    # Only raises if exclude_unassigned is True and marker is empty
    with pytest.raises(ValueError, match="unassigned_marker must be a non-empty string when exclude_unassigned is True."):
        calculator_fixture.calculate_raw_abundances({"read1": "StrainA"}, exclude_unassigned=True, unassigned_marker="")
    # Does not raise if exclude_unassigned is False and marker is empty (though risky)
    calculator_fixture.calculate_raw_abundances({"read1": "StrainA"}, exclude_unassigned=False, unassigned_marker="")


# --- Tests for calculate_relative_abundances ---
def test_calculate_relative_abundances_invalid_keys_values(calculator_fixture: AbundanceCalculator):
    with pytest.raises(ValueError, match="Strain names in raw_abundances must be non-empty strings."):
        calculator_fixture.calculate_relative_abundances(Counter({"": 10}))
    with pytest.raises(ValueError, match="Count for strain 'StrainA' must be a non-negative number"):
        calculator_fixture.calculate_relative_abundances(Counter({"StrainA": -1}))
    with pytest.raises(TypeError, match="raw_abundances must be a collections.Counter"):
        calculator_fixture.calculate_relative_abundances({"StrainA": 1}) # type: ignore


# --- Tests for apply_threshold_and_format ---
def test_apply_threshold_and_format_invalid_keys_values(calculator_fixture: AbundanceCalculator):
    with pytest.raises(ValueError, match="Strain names in relative_abundances must be non-empty strings."):
        calculator_fixture.apply_threshold_and_format({"": 0.5})
    with pytest.raises(TypeError, match="Abundance for strain 'StrainA' must be a float"):
        calculator_fixture.apply_threshold_and_format({"StrainA": "not-a-float"}) # type: ignore
    with pytest.raises(ValueError, match="Abundance for strain 'StrainA' .* must be between 0.0 and 1.0"):
        calculator_fixture.apply_threshold_and_format({"StrainA": 1.1})
    with pytest.raises(ValueError, match="Abundance for strain 'StrainA' .* must be between 0.0 and 1.0"):
        calculator_fixture.apply_threshold_and_format({"StrainA": -0.1})

# --- Tests for generate_report_string ---
def test_generate_report_string_percentage_format(calculator_fixture: AbundanceCalculator):
    # Refactored to output percentages
    final_abundances = {"StrainA": 0.75126, "StrainC": 0.1000, "StrainB": 0.14874}
    expected_report_lines = [
        "StrainA: 75.13%", 
        "StrainC: 10.00%",
        "StrainB: 14.87%" 
    ]
    expected_report = "\n".join(expected_report_lines)
    assert calculator_fixture.generate_report_string(final_abundances) == expected_report

def test_generate_report_string_invalid_keys_values(calculator_fixture: AbundanceCalculator):
    with pytest.raises(ValueError, match="Strain names in final_abundances must be non-empty strings."):
        calculator_fixture.generate_report_string({"": 0.5})
    with pytest.raises(TypeError, match="Abundance for strain 'StrainA' must be a float"):
        calculator_fixture.generate_report_string({"StrainA": "not-a-float"}) # type: ignore


# --- Re-check existing happy path tests to ensure they align with stricter validation if applicable ---
# (Most happy path tests from the previous version of test_output.py should still pass,
# as the stricter validation primarily targets invalid inputs. The core logic for valid inputs
# was largely preserved or made more robust, e.g., percentage formatting.)

# Example: test_convert_assignments_custom_unassigned_marker needs adjustment due to stricter validation
def test_convert_assignments_custom_unassigned_marker_strict(calculator_fixture: AbundanceCalculator, unassigned_marker_fixture: str):
    # "NonExistentStrain" will now raise ValueError, not be converted to marker.
    # Invalid index 99 will now raise ValueError.
    # Assignments that are already the marker are fine.
    
    # Test case for valid use of marker
    valid_assignments_with_marker: Dict[Union[str, int], Union[str, int]] = {
        "read1": 0, # StrainA
        "read_unassigned_by_marker": unassigned_marker_fixture 
    }
    expected_valid = {
        "read1": calculator_fixture.strain_names[0],
        "read_unassigned_by_marker": unassigned_marker_fixture,
    }
    assert calculator_fixture.convert_assignments_to_strain_names(valid_assignments_with_marker, unassigned_marker=unassigned_marker_fixture) == expected_valid

    # Test for ValueError with unknown strain string
    assignments_unknown_str: Dict[Union[str, int], Union[str, int]] = {"read_unknown": "UnknownStrain"}
    with pytest.raises(ValueError, match="String assignment 'UnknownStrain' for ReadId 'read_unknown' is not a recognized strain name"):
        calculator_fixture.convert_assignments_to_strain_names(assignments_unknown_str, unassigned_marker=unassigned_marker_fixture)

    # Test for ValueError with out-of-bounds index
    assignments_invalid_idx: Dict[Union[str, int], Union[str, int]] = {"read_invalid": 99}
    with pytest.raises(ValueError, match="StrainIndex 99 for ReadId 'read_invalid' is out of bounds"):
        calculator_fixture.convert_assignments_to_strain_names(assignments_invalid_idx, unassigned_marker=unassigned_marker_fixture)

```
