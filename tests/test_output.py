"""
Pytest unit tests for the AbundanceCalculator class from strainr.output.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""

from collections import Counter
from typing import Dict, List, Union

import pytest
from strainr.genomic_types import FinalAssignmentsType
from strainr.output import AbundanceCalculator

# Assuming StrainIndex and ReadId might be used if FinalAssignmentsType was more specific
# For now, the tests align with AbundanceCalculator's direct type hints.
# from strainr.genomic_types import StrainIndex, ReadId

# --- Fixtures ---


@pytest.fixture
def strain_names_fixture() -> List[str]:
    """Provides a default list of strain names for testing."""
    return ["StrainA", "StrainB", "StrainC"]


@pytest.fixture
def abundance_threshold_fixture() -> float:
    """Provides a default abundance threshold for testing."""
    return 0.01  # 1%


@pytest.fixture
def calculator_fixture(
    strain_names_fixture: List[str], abundance_threshold_fixture: float
) -> AbundanceCalculator:
    """Provides an AbundanceCalculator instance with default settings."""
    return AbundanceCalculator(
        strain_names=strain_names_fixture,
        abundance_threshold=abundance_threshold_fixture,
    )


@pytest.fixture
def unassigned_marker_fixture() -> str:
    """Provides a custom unassigned marker string."""
    return "NOT_ASSIGNED"


@pytest.fixture
def simple_final_assignments(
    strain_names_fixture: List[str], unassigned_marker_fixture: str
) -> FinalAssignmentsType:
    """Provides a sample FinalAssignmentsType dictionary."""
    # Assuming StrainA is index 0, StrainB is index 1
    return {
        "read1": 0,  # StrainA
        "read2": 1,  # StrainB
        "read3": 0,  # StrainA
        "read4": unassigned_marker_fixture,
        "read5": 0,  # StrainA
    }


# --- Test __init__ ---


def test_init_successful(strain_names_fixture: List[str]):
    calc = AbundanceCalculator(strain_names_fixture, abundance_threshold=0.05)
    assert calc.strain_names == strain_names_fixture
    assert calc.abundance_threshold == 0.05
    assert calc._num_strains == len(strain_names_fixture)


def test_init_invalid_strain_names_empty():
    with pytest.raises(ValueError, match="strain_names cannot be empty."):
        AbundanceCalculator(strain_names=[], abundance_threshold=0.01)


def test_init_invalid_strain_names_type():
    with pytest.raises(TypeError, match="strain_names must be a list of strings."):
        AbundanceCalculator(strain_names=[1, 2], abundance_threshold=0.01)  # type: ignore


def test_init_invalid_abundance_threshold_value(strain_names_fixture: List[str]):
    with pytest.raises(
        ValueError,
        match="abundance_threshold must be between 0.0 \\(inclusive\\) and 1.0 \\(exclusive\\).",
    ):
        AbundanceCalculator(strain_names_fixture, abundance_threshold=1.0)
    with pytest.raises(
        ValueError,
        match="abundance_threshold must be between 0.0 \\(inclusive\\) and 1.0 \\(exclusive\\).",
    ):
        AbundanceCalculator(strain_names_fixture, abundance_threshold=-0.1)


def test_init_invalid_abundance_threshold_type(strain_names_fixture: List[str]):
    with pytest.raises(TypeError, match="abundance_threshold must be a float."):
        AbundanceCalculator(strain_names_fixture, abundance_threshold="0.01")  # type: ignore


# --- Test convert_assignments_to_strain_names ---


def test_convert_assignments_typical(
    calculator_fixture: AbundanceCalculator, unassigned_marker_fixture: str
):
    read_assignments: Dict[Union[str, int], Union[str, int]] = {
        "read1": 0,  # StrainA
        "read2": 1,  # StrainB
        "read3": "StrainC",
        "read4": unassigned_marker_fixture,
        12345: 0,  # Test integer key
    }
    expected = {
        "read1": "StrainA",
        "read2": "StrainB",
        "read3": "StrainC",
        "read4": unassigned_marker_fixture,
        "12345": "StrainA",
    }
    result = calculator_fixture.convert_assignments_to_strain_names(
        read_assignments, unassigned_marker=unassigned_marker_fixture
    )
    assert result == expected


def test_convert_assignments_typical(
    calculator_fixture: AbundanceCalculator, unassigned_marker_fixture: str
):
    read_assignments: Dict[Union[str, int], Union[str, int]] = {
        "read1": 0,  # StrainA
        "read4": unassigned_marker_fixture,  # Already marked
    }
    # Test case for "NonExistentStrain"
    with pytest.raises(
        ValueError,
        match=f"String assignment 'NonExistentStrain' for ReadId 'read2' is not a recognized strain name or the unassigned_marker \\('{unassigned_marker_fixture}'\\).",
    ):
        calculator_fixture.convert_assignments_to_strain_names(
            {"read2": "NonExistentStrain"}, unassigned_marker=unassigned_marker_fixture
        )

    # Test case for invalid index 99
    with pytest.raises(
        ValueError,
        match=f"StrainIndex 99 for ReadId 'read3' is out of bounds \\[0, {len(calculator_fixture.strain_names) - 1}\\].",
    ):
        calculator_fixture.convert_assignments_to_strain_names(
            {"read3": 99}, unassigned_marker=unassigned_marker_fixture
        )

    # Test correct assignments
    expected_correct = {
        "read1": calculator_fixture.strain_names[0],  # StrainA
        "read4": unassigned_marker_fixture,
        "12345": "StrainA",
    }
    result_correct = calculator_fixture.convert_assignments_to_strain_names(
        read_assignments, unassigned_marker=unassigned_marker_fixture
    )
    assert result_correct == expected_correct


def test_convert_assignments_default_unassigned_marker_na(
    calculator_fixture: AbundanceCalculator,
):
    # Default unassigned_marker in the method is "NA"
    with pytest.raises(
        ValueError,
        match=f"StrainIndex 99 for ReadId 'read_invalid_idx' is out of bounds \\[0, {len(calculator_fixture.strain_names) - 1}\\].",
    ):
        calculator_fixture.convert_assignments_to_strain_names({"read_invalid_idx": 99})

    with pytest.raises(
        ValueError,
        match=f"String assignment 'UnknownStrainXYZ' for ReadId 'read_unknown_str' is not a recognized strain name or the unassigned_marker \\('NA'\\).",
    ):
        calculator_fixture.convert_assignments_to_strain_names({
            "read_unknown_str": "UnknownStrainXYZ"
        })
    with pytest.raises(
        ValueError,
        match=f"StrainIndex 99 for ReadId 'read_invalid_idx' is out of bounds \\[0, {len(calculator_fixture.strain_names) - 1}\\].",
    ):
        calculator_fixture.convert_assignments_to_strain_names({"read_invalid_idx": 99})

    with pytest.raises(
        ValueError,
        match=f"String assignment 'UnknownStrainXYZ' for ReadId 'read_unknown_str' is not a recognized strain name or the unassigned_marker \\('NA'\\).",
    ):
        calculator_fixture.convert_assignments_to_strain_names({
            "read_unknown_str": "UnknownStrainXYZ"
        })


def test_convert_assignments_empty_input(calculator_fixture: AbundanceCalculator):
    assert calculator_fixture.convert_assignments_to_strain_names({}) == {}


def test_convert_assignments_invalid_marker_type(
    calculator_fixture: AbundanceCalculator,
):
    with pytest.raises(
        ValueError, match="unassigned_marker must be a non-empty string."
    ):
        calculator_fixture.convert_assignments_to_strain_names(
            {"r1": 0}, unassigned_marker=""
        )  # type: ignore


# --- Test calculate_raw_abundances ---


def test_calculate_raw_abundances_typical(
    calculator_fixture: AbundanceCalculator, unassigned_marker_fixture: str
):
    named_assignments = {
        "r1": "StrainA",
        "r2": "StrainB",
        "r3": "StrainA",
        "r2": "StrainB",
        "r3": "StrainA",
        "r4": unassigned_marker_fixture,
    }
    raw_counts = calculator_fixture.calculate_raw_abundances(
        named_assignments,
        exclude_unassigned=True,
        unassigned_marker=unassigned_marker_fixture,
    )
    assert raw_counts == Counter({"StrainA": 2, "StrainB": 1})

    raw_counts_with_unassigned = calculator_fixture.calculate_raw_abundances(
        named_assignments,
        exclude_unassigned=False,
        unassigned_marker=unassigned_marker_fixture,
    )
    assert raw_counts_with_unassigned == Counter({
        "StrainA": 2,
        "StrainB": 1,
        unassigned_marker_fixture: 1,
    })


def test_calculate_raw_abundances_empty(calculator_fixture: AbundanceCalculator):
    assert calculator_fixture.calculate_raw_abundances({}) == Counter()

    # --- Test calculate_relative_abundances ---
    raw_counts = calculator_fixture.calculate_raw_abundances(
        named_assignments,
        exclude_unassigned=True,
        unassigned_marker=unassigned_marker_fixture,
    )
    assert raw_counts == Counter({"StrainA": 2, "StrainB": 1})

    raw_counts_with_unassigned = calculator_fixture.calculate_raw_abundances(
        named_assignments,
        exclude_unassigned=False,
        unassigned_marker=unassigned_marker_fixture,
    )
    assert raw_counts_with_unassigned == Counter({
        "StrainA": 2,
        "StrainB": 1,
        unassigned_marker_fixture: 1,
    })


def test_calculate_raw_abundances_empty(calculator_fixture: AbundanceCalculator):
    assert calculator_fixture.calculate_raw_abundances({}) == Counter()

    # --- Test calculate_relative_abundances ---
    raw_counts = calculator_fixture.calculate_raw_abundances(
        named_assignments,
        exclude_unassigned=True,
        unassigned_marker=unassigned_marker_fixture,
    )
    assert raw_counts == Counter({"StrainA": 2, "StrainB": 1})

    raw_counts_with_unassigned = calculator_fixture.calculate_raw_abundances(
        named_assignments,
        exclude_unassigned=False,
        unassigned_marker=unassigned_marker_fixture,
    )
    assert raw_counts_with_unassigned == Counter({
        "StrainA": 2,
        "StrainB": 1,
        unassigned_marker_fixture: 1,
    })


def test_calculate_raw_abundances_empty(calculator_fixture: AbundanceCalculator):
    assert calculator_fixture.calculate_raw_abundances({}) == Counter()


# --- Test calculate_relative_abundances ---


def test_calculate_relative_abundances_typical(
    calculator_fixture: AbundanceCalculator,
):
    raw_counts = Counter({"StrainA": 60, "StrainB": 30, "StrainC": 10})  # Total 100
    rel_ab = calculator_fixture.calculate_relative_abundances(raw_counts)
    assert rel_ab == {"StrainA": 0.6, "StrainB": 0.3, "StrainC": 0.1}


def test_calculate_relative_abundances_empty(calculator_fixture: AbundanceCalculator):
    raw_counts = Counter({"StrainA": 60, "StrainB": 30, "StrainC": 10})  # Total 100
    rel_ab = calculator_fixture.calculate_relative_abundances(raw_counts)
    assert rel_ab == {"StrainA": 0.6, "StrainB": 0.3, "StrainC": 0.1}


def test_calculate_relative_abundances_empty(calculator_fixture: AbundanceCalculator):
    assert calculator_fixture.calculate_relative_abundances(Counter()) == {}


# --- Test apply_threshold_and_format ---


def test_apply_threshold_and_format_basic(
    calculator_fixture: AbundanceCalculator, abundance_threshold_fixture: float
):
    # threshold is 0.01
    rel_ab = {"StrainA": 0.7, "StrainB": 0.2, "StrainC": 0.005}  # StrainC < threshold
    formatted = calculator_fixture.apply_threshold_and_format(
        rel_ab, sort_by_abundance=True
    )
    assert formatted == {"StrainA": 0.7, "StrainB": 0.2}  # Sorted by abundance
    assert formatted == {"StrainA": 0.7, "StrainB": 0.2}  # Sorted by abundance


def test_apply_threshold_and_format_all_below_threshold(
    calculator_fixture: AbundanceCalculator,
):
    rel_ab = {"StrainA": 0.005, "StrainB": 0.001}
    assert calculator_fixture.apply_threshold_and_format(rel_ab) == {}
    rel_ab = {"StrainA": 0.005, "StrainB": 0.001}
    assert calculator_fixture.apply_threshold_and_format(rel_ab) == {}


def test_apply_threshold_and_format_sort_by_name():
    rel_ab = {"StrainC": 0.3, "StrainA": 0.5, "StrainB": 0.2}
    formatted = calculator_fixture.apply_threshold_and_format(
        rel_ab, sort_by_abundance=False
    )
    # Expected order: StrainA, StrainB, StrainC (alphabetical)
    assert list(formatted.keys()) == ["StrainA", "StrainB", "StrainC"]


# --- Test generate_report_string ---


def test_generate_report_string_empty(calculator_fixture: AbundanceCalculator):
    assert (
        calculator_fixture.generate_report_string({})
        == "No strains found above the abundance threshold."
    )
    rel_ab = {"StrainC": 0.3, "StrainA": 0.5, "StrainB": 0.2}
    formatted = calculator_fixture.apply_threshold_and_format(
        rel_ab, sort_by_abundance=False
    )
    # Expected order: StrainA, StrainB, StrainC (alphabetical)
    assert list(formatted.keys()) == ["StrainA", "StrainB", "StrainC"]


# --- Test generate_report_string ---


def test_generate_report_string_empty(calculator_fixture: AbundanceCalculator):
    assert (
        calculator_fixture.generate_report_string({})
        == "No strains found above the abundance threshold."
    )


def test_generate_report_string_basic_formatting(
    calculator_fixture: AbundanceCalculator,
):
    final_abundances = {"StrainA": 0.75126, "StrainC": 0.1000, "StrainB": 0.14874}
    expected_report_lines = [
        "StrainA: 75.13%",
        "StrainC: 10.00%",
        "StrainB: 14.87%",
        "StrainA: 75.13%",
        "StrainC: 10.00%",
        "StrainB: 14.87%",
    ]
    expected_report = "\n".join(expected_report_lines)
    assert (
        calculator_fixture.generate_report_string(final_abundances) == expected_report
    )


def test_generate_report_string_single_strain(calculator_fixture: AbundanceCalculator):
    final_abundances = {"StrainX": 1.0}
    expected_report = "StrainX: 100.00%"
    assert (
        calculator_fixture.generate_report_string(final_abundances) == expected_report
    )


def test_generate_report_string_zero_abundance(
    calculator_fixture: AbundanceCalculator,
):
    final_abundances = {"StrainA": 0.0}
    expected_report = "StrainA: 0.00%"
    expected_report = "StrainX: 100.00%"
    assert (
        calculator_fixture.generate_report_string(final_abundances) == expected_report
    )


def test_generate_report_string_zero_abundance(
    calculator_fixture: AbundanceCalculator,
):
    final_abundances = {"StrainA": 0.0}
    expected_report = "StrainA: 0.00%"
    assert (
        calculator_fixture.generate_report_string(final_abundances) == expected_report
    )


# --- Test for invalid input types (more robust checks) ---
def test_convert_assignments_invalid_read_id_type(
    calculator_fixture: AbundanceCalculator,
):
    with pytest.raises(
        ValueError, match="ReadId keys, after string conversion, must not be empty."
    ):
        calculator_fixture.convert_assignments_to_strain_names({None: 0})  # type: ignore


def test_calculate_raw_abundances_invalid_key_type(
    calculator_fixture: AbundanceCalculator,
):
    with pytest.raises(
        ValueError,
        match="All ReadId keys in named_assignments must be non-empty strings.",
    ):
        calculator_fixture.calculate_raw_abundances({None: "StrainA"})  # type: ignore


def test_calculate_raw_abundances_invalid_value_type(
    calculator_fixture: AbundanceCalculator,
):
    with pytest.raises(
        ValueError,
        match="Assigned name/marker for ReadId 'read1' must be a non-empty string.",
    ):
        calculator_fixture.calculate_raw_abundances({"read1": None})  # type: ignore


def test_calculate_relative_abundances_invalid_key_type(
    calculator_fixture: AbundanceCalculator,
):
    with pytest.raises(
        ValueError, match="Strain names in raw_abundances must be non-empty strings."
    ):
        calculator_fixture.calculate_relative_abundances(Counter({None: 10}))  # type: ignore


def test_calculate_relative_abundances_invalid_value_type(
    calculator_fixture: AbundanceCalculator,
):
    with pytest.raises(
        ValueError,
        match="Count for strain 'StrainA' must be a non-negative number, got None.",
    ):
        calculator_fixture.calculate_relative_abundances(Counter({"StrainA": None}))  # type: ignore


def test_apply_threshold_invalid_key_type(calculator_fixture: AbundanceCalculator):
    with pytest.raises(
        ValueError,
        match="Strain names in relative_abundances must be non-empty strings.",
    ):
        calculator_fixture.apply_threshold_and_format({None: 0.5})  # type: ignore


def test_apply_threshold_invalid_value_type(calculator_fixture: AbundanceCalculator):
    with pytest.raises(
        TypeError,
        match="Abundance for strain 'StrainA' must be a float, got <class 'NoneType'>.",
    ):
        calculator_fixture.apply_threshold_and_format({"StrainA": None})  # type: ignore
    with pytest.raises(
        ValueError,
        match="Abundance for strain 'StrainA' .* must be between 0.0 and 1.0.",
    ):
        calculator_fixture.apply_threshold_and_format({"StrainA": 1.5})


def test_generate_report_invalid_key_type(calculator_fixture: AbundanceCalculator):
    with pytest.raises(
        ValueError, match="Strain names in final_abundances must be non-empty strings."
    ):
        calculator_fixture.generate_report_string({None: 0.5})  # type: ignore


def test_generate_report_invalid_value_type(calculator_fixture: AbundanceCalculator):
    with pytest.raises(
        TypeError,
        match="Abundance for strain 'StrainA' must be a float, got <class 'NoneType'>.",
    ):
        calculator_fixture.generate_report_string({"StrainA": None})  # type: ignore
