"""
Pytest unit tests for the ClassificationAnalyzer class from src.strainr.analyze.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""

import pytest
import numpy as np
from collections import Counter
from typing import Dict, List, Union
from unittest.mock import patch, MagicMock, call # Added call

from src.strainr.analyze import ClassificationAnalyzer
from src.strainr.genomic_types import ReadId, CountVector, StrainIndex, ReadHitResults

# --- Fixtures ---


@pytest.fixture
def strain_names_fixture() -> List[str]:
    """Provides a default list of strain names."""
    return ["StrainA", "StrainB", "StrainC", "StrainD"]


@pytest.fixture
def analyzer_fixture(strain_names_fixture: List[str]) -> ClassificationAnalyzer:
    """Provides a default ClassificationAnalyzer instance for general tests."""
    return ClassificationAnalyzer(
        strain_names=strain_names_fixture,
        disambiguation_mode="max",
        num_processes=1,
    )


# Fixtures for specific disambiguation modes
@pytest.fixture
def analyzer_random_mode(strain_names_fixture: List[str]) -> ClassificationAnalyzer:
    return ClassificationAnalyzer(
        strain_names=strain_names_fixture, disambiguation_mode="random", num_processes=1
    )


@pytest.fixture
def analyzer_multinomial_mode(
    strain_names_fixture: List[str],
) -> ClassificationAnalyzer:
    return ClassificationAnalyzer(
        strain_names=strain_names_fixture,
        disambiguation_mode="multinomial",
        num_processes=1,
    )


@pytest.fixture
def analyzer_dirichlet_mode(strain_names_fixture: List[str]) -> ClassificationAnalyzer:
    return ClassificationAnalyzer(
        strain_names=strain_names_fixture,
        disambiguation_mode="dirichlet",
        num_processes=1,
    )


# --- Test __init__ ---


def test_init_successful(strain_names_fixture: List[str]):
    analyzer = ClassificationAnalyzer(
        strain_names=strain_names_fixture,
        disambiguation_mode="max",
        abundance_threshold=0.01,
        num_processes=2,
    )
    assert analyzer.strain_names == strain_names_fixture
    assert analyzer.disambiguation_mode == "max"
    assert analyzer.abundance_threshold == 0.01
    assert analyzer.num_processes == 2
    assert isinstance(analyzer.random_generator, np.random.Generator)


def test_init_invalid_strain_names_empty():
    with pytest.raises(ValueError, match="strain_names cannot be empty."):
        ClassificationAnalyzer(strain_names=[])


def test_init_invalid_strain_names_type():
    with pytest.raises(TypeError, match="strain_names must be a list of strings."):
        ClassificationAnalyzer(strain_names=[1, 2, 3])  # type: ignore


def test_init_invalid_disambiguation_mode(strain_names_fixture: List[str]):
    with pytest.raises(
        ValueError, match="Unsupported disambiguation_mode: invalid_mode"
    ):
        ClassificationAnalyzer(
            strain_names=strain_names_fixture, disambiguation_mode="invalid_mode"
        )


def test_init_invalid_num_processes(strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="num_processes must be a positive integer."):
        ClassificationAnalyzer(strain_names=strain_names_fixture, num_processes=0)
    with pytest.raises(ValueError, match="num_processes must be a positive integer."):
        ClassificationAnalyzer(strain_names=strain_names_fixture, num_processes=-1)


def test_init_invalid_abundance_threshold(strain_names_fixture: List[str]):
    with pytest.raises(
        ValueError, match=r"abundance_threshold must be a float between 0.0 and 1.0 \(exclusive of 1.0\)."
    ):
        ClassificationAnalyzer(
            strain_names=strain_names_fixture, abundance_threshold=-0.1
        )
    with pytest.raises(
        ValueError, match=r"abundance_threshold must be a float between 0.0 and 1.0 \(exclusive of 1.0\)."
    ):
        ClassificationAnalyzer(
            strain_names=strain_names_fixture, abundance_threshold=1.0
        )


# --- Test separate_hit_categories ---


def test_separate_hit_categories_empty_input(analyzer_fixture: ClassificationAnalyzer):
    results: ReadHitResults = []
    clear, ambiguous, no_hits = analyzer_fixture.separate_hit_categories(results)
    assert not clear
    assert not ambiguous
    assert not no_hits


def test_separate_hit_categories_only_clear_hits(
    analyzer_fixture: ClassificationAnalyzer,
):
    results: ReadHitResults = [
        ("read1", np.array([10, 0, 0, 0], dtype=np.uint8)),
        ("read2", np.array([0, 5, 0, 0], dtype=np.uint8)),
    ]
    clear, ambiguous, no_hits = analyzer_fixture.separate_hit_categories(results)
    assert len(clear) == 2
    assert "read1" in clear and np.array_equal(clear["read1"], results[0][1])
    assert not ambiguous and not no_hits


def test_separate_hit_categories_only_ambiguous_hits(
    analyzer_fixture: ClassificationAnalyzer,
):
    results: ReadHitResults = [
        ("read1", np.array([10, 10, 0, 0], dtype=np.uint8)),
        ("read2", np.array([0, 5, 5, 5], dtype=np.uint8)),
    ]
    clear, ambiguous, no_hits = analyzer_fixture.separate_hit_categories(results)
    assert not clear
    assert len(ambiguous) == 2
    assert "read1" in ambiguous and np.array_equal(ambiguous["read1"], results[0][1])
    assert not no_hits


def test_separate_hit_categories_only_no_hits(analyzer_fixture: ClassificationAnalyzer):
    results: ReadHitResults = [
        ("read1", np.array([0, 0, 0, 0], dtype=np.uint8)),
        ("read2", np.array([0, 0, 0, 0], dtype=np.uint8)),
    ]
    clear, ambiguous, no_hits = analyzer_fixture.separate_hit_categories(results)
    assert not clear and not ambiguous
    assert len(no_hits) == 2 and "read1" in no_hits and "read2" in no_hits


def test_separate_hit_categories_mixed_hits(analyzer_fixture: ClassificationAnalyzer):
    results: ReadHitResults = [
        ("read_clear", np.array([10, 0, 0, 0], dtype=np.uint8)),
        ("read_amb", np.array([5, 5, 0, 0], dtype=np.uint8)),
        ("read_none", np.array([0, 0, 0, 0], dtype=np.uint8)),
    ]
    clear, ambiguous, no_hits = analyzer_fixture.separate_hit_categories(results)
    assert len(clear) == 1 and "read_clear" in clear
    assert len(ambiguous) == 1 and "read_amb" in ambiguous
    assert len(no_hits) == 1 and "read_none" in no_hits


def test_separate_hit_categories_invalid_input_type(
    analyzer_fixture: ClassificationAnalyzer,
):
    with pytest.raises(
        TypeError, match="classification_results must be a list." 
    ):
        analyzer_fixture.separate_hit_categories({"invalid": "data"})  # type: ignore

    results_invalid_content: ReadHitResults = [("read1", [0, 5, 0, 0])]  # type: ignore
    with pytest.raises(
        TypeError,
        match=r"Item 0 \('read1'\): CountVector must be a NumPy array, got <class 'list'>." # Corrected regex
    ):
        analyzer_fixture.separate_hit_categories(results_invalid_content)


# --- Test resolve_clear_hits_to_indices ---


def test_resolve_clear_hits_to_indices_typical(
    analyzer_fixture: ClassificationAnalyzer,
):
    clear_hits_dict: Dict[ReadId, CountVector] = {
        "read1": np.array([10, 0, 0, 0], dtype=np.uint8),  # StrainA (idx 0)
        "read2": np.array([0, 0, 5, 0], dtype=np.uint8),  # StrainC (idx 2)
    }
    resolved = analyzer_fixture.resolve_clear_hits_to_indices(clear_hits_dict)
    assert resolved == {"read1": 0, "read2": 2}


def test_resolve_clear_hits_to_indices_empty(analyzer_fixture: ClassificationAnalyzer):
    assert analyzer_fixture.resolve_clear_hits_to_indices({}) == {}


def test_resolve_clear_hits_to_indices_invalid_input(
    analyzer_fixture: ClassificationAnalyzer,
):
    with pytest.raises(TypeError, match="clear_hits_dict must be a dictionary."):
        analyzer_fixture.resolve_clear_hits_to_indices([])  # type: ignore


# --- Test calculate_strain_prior_from_assignments ---


def test_calculate_strain_prior_from_assignments_typical(
    analyzer_fixture: ClassificationAnalyzer,
):
    assignments: Dict[ReadId, StrainIndex] = {"r1": 0, "r2": 2, "r3": 0, "r4": 1}
    priors = analyzer_fixture.calculate_strain_prior_from_assignments(assignments)
    assert priors == Counter({0: 2, 2: 1, 1: 1})


def test_calculate_strain_prior_from_assignments_empty(
    analyzer_fixture: ClassificationAnalyzer,
):
    assert analyzer_fixture.calculate_strain_prior_from_assignments({}) == Counter()


def test_calculate_strain_prior_from_assignments_invalid_input(
    analyzer_fixture: ClassificationAnalyzer,
):
    with pytest.raises(
        TypeError, match="clear_strain_assignments must be a dictionary."
    ):
        analyzer_fixture.calculate_strain_prior_from_assignments([])  # type: ignore


# --- Test convert_prior_counts_to_probability_vector ---


def test_convert_prior_counts_to_probability_vector_basic(
    analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]
):
    counts = Counter({0: 6, 1: 3, 2: 1})  # Total 10
    probs = analyzer_fixture.convert_prior_counts_to_probability_vector(counts)
    expected = np.array([0.6, 0.3, 0.1, 1e-20])  # StrainD gets epsilon
    np.testing.assert_array_almost_equal(probs, expected)


def test_convert_prior_counts_to_probability_vector_all_zeros(
    analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]
):
    counts = Counter()
    probs = analyzer_fixture.convert_prior_counts_to_probability_vector(counts)
    expected = np.full(len(strain_names_fixture), 1e-20)
    np.testing.assert_array_almost_equal(probs, expected)


def test_convert_prior_counts_to_probability_vector_invalid_input(
    analyzer_fixture: ClassificationAnalyzer,
):
    with pytest.raises(TypeError, match="strain_prior_counts must be a Counter."):
        analyzer_fixture.convert_prior_counts_to_probability_vector({})  # type: ignore


# --- Test _resolve_single_ambiguous_read ---

@patch('src.strainr.analyze.np.random.default_rng')
def test_resolve_single_ambiguous_read_max_mode(
    mock_default_rng: MagicMock,
    strain_names_fixture: List[str],
):
    mock_rng_instance = MagicMock()
    mock_rng_instance.choice = MagicMock(return_value=0) 
    mock_default_rng.return_value = mock_rng_instance

    analyzer = ClassificationAnalyzer(strain_names=strain_names_fixture, disambiguation_mode="max")

    hits = np.array([10, 10, 5, 0], dtype=np.uint8)
    # Scenario 1: Equal priors for max hits -> chooses first index
    priors_equal = np.array([0.25, 0.25, 0.25, 0.25])
    assert analyzer._resolve_single_ambiguous_read(hits.copy(), priors_equal) == 0

    # Scenario 2: Higher prior for second max hit -> chooses second index
    priors_favor_second = np.array([0.1, 0.7, 0.1, 0.1])
    assert analyzer._resolve_single_ambiguous_read(hits.copy(), priors_favor_second) == 1

    # Scenario 3: Priors zero out all max hits, fallback to original max hits
    hits_2 = np.array([10, 10, 0, 0], dtype=np.uint8)
    priors_zero_max = np.array([0.0, 0.0, 0.5, 0.5])
    
    resolved_idx = analyzer._resolve_single_ambiguous_read(hits_2.copy(), priors_zero_max)
    assert resolved_idx == 0 
    # In this path (max mode, non-zero sum_likelihood_scores), 'choice' is not called.
    # So, no assertion on mock_rng_instance.choice call count.


@patch('src.strainr.analyze.np.random.default_rng')
def test_resolve_single_ambiguous_read_random_mode(
    mock_default_rng: MagicMock,
    strain_names_fixture: List[str],
):
    mock_rng_instance = MagicMock()
    mock_rng_instance.choice = MagicMock(return_value=0)
    mock_default_rng.return_value = mock_rng_instance
    
    analyzer_random_mode = ClassificationAnalyzer(strain_names=strain_names_fixture, disambiguation_mode="random")
    
    hits = np.array([10, 10, 0, 0], dtype=np.uint8)
    priors = np.array([0.6, 0.4, 0.0, 0.0])
    expected_probs = np.array([0.6, 0.4, 0.0, 0.0])

    resolved_idx = analyzer_random_mode._resolve_single_ambiguous_read(hits.copy(), priors)
    assert resolved_idx == 0
    
    mock_rng_instance.choice.assert_called_once()
    call_args = mock_rng_instance.choice.call_args
    np.testing.assert_array_equal(call_args[0][0], np.arange(len(analyzer_random_mode.strain_names)))
    np.testing.assert_array_almost_equal(call_args[1]["p"], expected_probs)


@patch('src.strainr.analyze.np.random.default_rng')
def test_resolve_single_ambiguous_read_multinomial_mode(
    mock_default_rng: MagicMock,
    strain_names_fixture: List[str],
):
    mock_rng_instance = MagicMock()
    mock_multinomial_output = np.array([0, 0, 1, 0]) 
    mock_rng_instance.multinomial = MagicMock(return_value=mock_multinomial_output)
    mock_default_rng.return_value = mock_rng_instance

    analyzer_multinomial_mode = ClassificationAnalyzer(strain_names=strain_names_fixture, disambiguation_mode="multinomial")

    hits = np.array([5, 5, 5, 0], dtype=np.uint8)
    priors = np.array([0.1, 0.2, 0.7, 0.0])
    expected_probs = np.array([0.1, 0.2, 0.7, 0.0])

    resolved_idx = analyzer_multinomial_mode._resolve_single_ambiguous_read(hits.copy(), priors)
    assert resolved_idx == 2
    mock_rng_instance.multinomial.assert_called_once_with(1, pvals=pytest.approx(expected_probs))


@patch('src.strainr.analyze.np.random.default_rng')
def test_resolve_single_ambiguous_read_dirichlet_mode(
    mock_default_rng: MagicMock,
    strain_names_fixture: List[str],
):
    mock_rng_instance = MagicMock()
    mock_dirichlet_sample = np.array([0.6,0.0,0.4,0.0,])
    mock_rng_instance.dirichlet = MagicMock(return_value=mock_dirichlet_sample)
    mock_rng_instance.choice = MagicMock(return_value=0)
    mock_default_rng.return_value = mock_rng_instance

    analyzer_dirichlet_mode = ClassificationAnalyzer(strain_names=strain_names_fixture, disambiguation_mode="dirichlet")

    hits = np.array([10, 0, 10, 0], dtype=np.uint8)
    priors = np.array([0.5, 0.0, 0.5, 0.0])
    expected_alpha = np.array([5.0, 1e-10, 5.0, 1e-10])

    resolved_idx = analyzer_dirichlet_mode._resolve_single_ambiguous_read(hits.copy(), priors)
    assert resolved_idx == 0

    mock_rng_instance.dirichlet.assert_called_once_with(alpha=pytest.approx(expected_alpha))
    mock_rng_instance.choice.assert_called_once()
    call_args_choice = mock_rng_instance.choice.call_args
    np.testing.assert_array_equal(call_args_choice[0][0], np.arange(len(analyzer_dirichlet_mode.strain_names)))
    np.testing.assert_array_almost_equal(call_args_choice[1]["p"], mock_dirichlet_sample)


# --- Test resolve_ambiguous_hits_parallel ---


def test_resolve_ambiguous_hits_parallel_empty(
    analyzer_fixture: ClassificationAnalyzer,
):
    num_strains = len(analyzer_fixture.strain_names)
    correct_shape_priors = np.array([1.0/num_strains if num_strains > 0 else 0.0] * num_strains)
    if num_strains == 0: 
        correct_shape_priors = np.array([])

    assert analyzer_fixture.resolve_ambiguous_hits_parallel({}, correct_shape_priors) == {}


@patch("multiprocessing.Pool")
def test_resolve_ambiguous_hits_parallel_mocked_pool(
    mock_pool_constructor: MagicMock,
    analyzer_fixture: ClassificationAnalyzer,
    strain_names_fixture: List[str],
):
    analyzer = analyzer_fixture 
    amb_hits: Dict[ReadId, CountVector] = {
        "read_amb1": np.array([10, 10, 0, 0], dtype=np.uint8),
        "read_amb2": np.array([0, 5, 5, 0], dtype=np.uint8),
    }
    priors = np.array([0.25] * len(strain_names_fixture))

    expected_resolved_indices = [0, 1]
    mock_pool_instance = MagicMock()
    mock_pool_instance.map.return_value = expected_resolved_indices
    mock_pool_constructor.return_value.__enter__.return_value = mock_pool_instance

    resolved = analyzer.resolve_ambiguous_hits_parallel(amb_hits, priors)
    
    mock_pool_constructor.assert_called_with(processes=1)

    map_call_args = mock_pool_instance.map.call_args
    assert map_call_args is not None
    partial_func = map_call_args[0][0]
    assert partial_func.func.__name__ == "_resolve_single_ambiguous_read"
    np.testing.assert_array_equal(partial_func.keywords["prior_probabilities"], priors)
    hit_vectors_arg = map_call_args[0][1]
    assert len(hit_vectors_arg) == 2
    np.testing.assert_array_equal(hit_vectors_arg[0], amb_hits["read_amb1"])

    assert resolved == {"read_amb1": 0, "read_amb2": 1}


# --- Test combine_assignments ---


def test_combine_assignments_typical(analyzer_fixture: ClassificationAnalyzer):
    clear: Dict[ReadId, StrainIndex] = {"clear1": 0, "clear2": 1}
    resolved_amb: Dict[ReadId, StrainIndex] = {"amb1": 2, "amb2": 0}
    no_hit_ids: List[ReadId] = ["no_hit1", "no_hit2"]
    marker = "UNASSIGNED"

    combined = analyzer_fixture.combine_assignments(
        clear, resolved_amb, no_hit_ids, unassigned_marker=marker
    )
    expected: Dict[ReadId, Union[StrainIndex, str]] = {
        "clear1": 0,
        "clear2": 1,
        "amb1": 2,
        "amb2": 0,
        "no_hit1": marker,
        "no_hit2": marker,
    }
    assert combined == expected


def test_combine_assignments_empty_categories(analyzer_fixture: ClassificationAnalyzer):
    clear: Dict[ReadId, StrainIndex] = {"c1": 0}
    combined = analyzer_fixture.combine_assignments(
        clear, {}, [], unassigned_marker="NA"
    )
    assert combined == {"c1": 0}


def test_combine_assignments_all_empty(analyzer_fixture: ClassificationAnalyzer):
    assert analyzer_fixture.combine_assignments({}, {}, []) == {}


def test_combine_assignments_invalid_types(analyzer_fixture: ClassificationAnalyzer):
    with pytest.raises(TypeError, match="Invalid input types"):
        analyzer_fixture.combine_assignments("not_dict", {}, [])  # type: ignore
