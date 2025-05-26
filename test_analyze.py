"""
Pytest unit tests for the ClassificationAnalyzer class from src.strainr.analyze.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""
import pytest
import numpy as np
from collections import Counter
from typing import Dict, List, Union # Removed Generator as streaming not tested here
from unittest.mock import patch, MagicMock, call

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
        num_processes=1 
    )

@pytest.fixture
def analyzer_random_mode(strain_names_fixture: List[str]) -> ClassificationAnalyzer:
    return ClassificationAnalyzer(strain_names=strain_names_fixture, disambiguation_mode="random", num_processes=1)

@pytest.fixture
def analyzer_multinomial_mode(strain_names_fixture: List[str]) -> ClassificationAnalyzer:
    return ClassificationAnalyzer(strain_names=strain_names_fixture, disambiguation_mode="multinomial", num_processes=1)

@pytest.fixture
def analyzer_dirichlet_mode(strain_names_fixture: List[str]) -> ClassificationAnalyzer:
    return ClassificationAnalyzer(strain_names=strain_names_fixture, disambiguation_mode="dirichlet", num_processes=1)

# --- Helper to create CountVectors ---
def create_cv(values: List[Union[int, float]], dtype=np.uint8) -> CountVector:
    return np.array(values, dtype=dtype)

# --- Test __init__ ---

def test_init_successful(strain_names_fixture: List[str]):
    analyzer = ClassificationAnalyzer(
        strain_names=strain_names_fixture,
        disambiguation_mode="max",
        abundance_threshold=0.01,
        num_processes=2
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
        ClassificationAnalyzer(strain_names=[1, 2, 3]) # type: ignore
    with pytest.raises(TypeError, match="strain_names must be a list of strings."):
        ClassificationAnalyzer(strain_names=["StrainA", 123]) # type: ignore


def test_init_duplicate_strain_names(strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="strain_names must be unique."):
        ClassificationAnalyzer(strain_names=["StrainA", "StrainB", "StrainA"])

def test_init_empty_string_in_strain_names(strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="strain_names must not contain empty strings."):
        ClassificationAnalyzer(strain_names=["StrainA", "", "StrainC"])


def test_init_invalid_disambiguation_mode(strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="Unsupported disambiguation_mode: invalid_mode"):
        ClassificationAnalyzer(strain_names=strain_names_fixture, disambiguation_mode="invalid_mode")

def test_init_invalid_num_processes(strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="num_processes must be a positive integer."):
        ClassificationAnalyzer(strain_names=strain_names_fixture, num_processes=0)
    with pytest.raises(ValueError, match="num_processes must be a positive integer."):
        ClassificationAnalyzer(strain_names=strain_names_fixture, num_processes=-1)
    with pytest.raises(ValueError, match="num_processes must be a positive integer."):
        ClassificationAnalyzer(strain_names=strain_names_fixture, num_processes=1.5) # type: ignore

def test_init_invalid_abundance_threshold(strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="abundance_threshold must be a float between 0.0 and 1.0"):
        ClassificationAnalyzer(strain_names=strain_names_fixture, abundance_threshold=-0.1)
    with pytest.raises(ValueError, match="abundance_threshold must be a float between 0.0 and 1.0"):
        ClassificationAnalyzer(strain_names=strain_names_fixture, abundance_threshold=1.0)
    with pytest.raises(ValueError, match="abundance_threshold must be a float between 0.0 and 1.0"):
         ClassificationAnalyzer(strain_names=strain_names_fixture, abundance_threshold="0.1") # type: ignore


# --- Test separate_hit_categories ---

def test_separate_hit_categories_empty_input(analyzer_fixture: ClassificationAnalyzer):
    results: ReadHitResults = []
    clear, ambiguous, no_hits = analyzer_fixture.separate_hit_categories(results)
    assert not clear and not ambiguous and not no_hits

def test_separate_hit_categories_valid_data(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    results: ReadHitResults = [
        ("read1", create_cv([10] + [0]*(n_strains-1))),      # Clear
        ("read2", create_cv([5, 5] + [0]*(n_strains-2))),    # Ambiguous
        ("read3", create_cv([0]*n_strains)),                 # No Hit
    ]
    clear, ambiguous, no_hits = analyzer_fixture.separate_hit_categories(results)
    assert len(clear) == 1 and "read1" in clear
    assert len(ambiguous) == 1 and "read2" in ambiguous
    assert len(no_hits) == 1 and "read3" in no_hits

def test_separate_hit_categories_invalid_cv_structure(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    # Wrong type for CV
    with pytest.raises(TypeError, match="Each item in classification_results must be \\(ReadId, CountVector\\)."):
        analyzer_fixture.separate_hit_categories([("read1", [1,2,3,4])]) # type: ignore
    # Wrong nDim for CV
    with pytest.raises(ValueError, match="CountVector must be 1-dimensional"):
        analyzer_fixture.separate_hit_categories([("read1", np.array([[1]*n_strains]))])
    # Wrong length for CV
    with pytest.raises(ValueError, match="CountVector length .* must match number of strains"):
        analyzer_fixture.separate_hit_categories([("read1", create_cv([1]*(n_strains-1)))])
    # Wrong dtype for CV
    with pytest.raises(TypeError, match="CountVector dtype must be unsigned integer"):
        analyzer_fixture.separate_hit_categories([("read1", create_cv([1.0]*n_strains, dtype=float))])


# --- Test resolve_clear_hits_to_indices ---
def test_resolve_clear_hits_to_indices_invalid_cv(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    invalid_cv_dict: Dict[ReadId, CountVector] = {"read1": create_cv([1.0]*n_strains, dtype=float)}
    with pytest.raises(TypeError, match="CountVector dtype must be unsigned integer"):
        analyzer_fixture.resolve_clear_hits_to_indices(invalid_cv_dict)


# --- Test calculate_strain_prior_from_assignments ---
def test_calculate_strain_prior_invalid_index(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    invalid_assignments: Dict[ReadId, StrainIndex] = {"read1": n_strains} # Index out of bounds
    with pytest.raises(ValueError, match=f"StrainIndex {n_strains} for ReadId 'read1' is out of range"):
        analyzer_fixture.calculate_strain_prior_from_assignments(invalid_assignments)
    
    invalid_assignments_neg: Dict[ReadId, StrainIndex] = {"read1": -1}
    with pytest.raises(ValueError, match=f"StrainIndex -1 for ReadId 'read1' is out of range"):
        analyzer_fixture.calculate_strain_prior_from_assignments(invalid_assignments_neg)


# --- Test convert_prior_counts_to_probability_vector ---
def test_convert_prior_counts_invalid_index(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    invalid_counts = Counter({n_strains: 5}) # Index out of bounds
    with pytest.raises(ValueError, match=f"StrainIndex key {n_strains} in Counter is out of range"):
        analyzer_fixture.convert_prior_counts_to_probability_vector(invalid_counts)

    invalid_counts_neg = Counter({-1: 5})
    with pytest.raises(ValueError, match=f"StrainIndex key -1 in Counter is out of range"):
        analyzer_fixture.convert_prior_counts_to_probability_vector(invalid_counts_neg)


# --- Test _resolve_single_ambiguous_read ---
# (Existing tests for different modes are good. Adding validation tests)

def test_resolve_single_ambiguous_read_invalid_strain_hits(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    valid_priors = np.array([1.0/n_strains]*n_strains)
    
    with pytest.raises(ValueError, match="CountVector length .* must match number of strains"):
        analyzer_fixture._resolve_single_ambiguous_read(create_cv([1]*(n_strains-1)), valid_priors)
    with pytest.raises(TypeError, match="CountVector dtype must be unsigned integer"):
        analyzer_fixture._resolve_single_ambiguous_read(create_cv([1.0]*n_strains, dtype=float), valid_priors)

def test_resolve_single_ambiguous_read_invalid_priors(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    valid_hits = create_cv([1]*n_strains)

    with pytest.raises(TypeError, match="prior_probabilities must be a NumPy array"):
        analyzer_fixture._resolve_single_ambiguous_read(valid_hits, [0.25]*n_strains) # type: ignore
    with pytest.raises(ValueError, match="prior_probabilities must be 1-dimensional"):
        analyzer_fixture._resolve_single_ambiguous_read(valid_hits, np.array([[0.25]*n_strains]))
    with pytest.raises(ValueError, match="prior_probabilities length .* must match number of strains"):
        analyzer_fixture._resolve_single_ambiguous_read(valid_hits, np.array([0.25]*(n_strains-1)))
    with pytest.raises(TypeError, match="prior_probabilities dtype must be float"):
        analyzer_fixture._resolve_single_ambiguous_read(valid_hits, np.array([1]*n_strains, dtype=int))
    with pytest.raises(ValueError, match="All prior_probabilities must be non-negative"):
        analyzer_fixture._resolve_single_ambiguous_read(valid_hits, np.array([-0.1, 1.1] + [0.0]*(n_strains-2)))

def test_resolve_single_ambiguous_read_priors_sum_not_one_warning(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str], capsys):
    n_strains = len(strain_names_fixture)
    valid_hits = create_cv([1]*n_strains)
    priors_sum_not_one = np.array([0.1, 0.1, 0.1, 0.1]) # Sums to 0.4
    analyzer_fixture._resolve_single_ambiguous_read(valid_hits, priors_sum_not_one)
    captured = capsys.readouterr()
    assert "Warning: prior_probabilities sum (0.4) is not close to 1.0." in captured.out

def test_resolve_single_ambiguous_read_zero_hit_vector_warning(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str], capsys):
    n_strains = len(strain_names_fixture)
    zero_hits = create_cv([0]*n_strains)
    priors = np.array([1.0/n_strains]*n_strains)
    # This case should now trigger a warning and random choice based on priors
    with patch.object(analyzer_fixture.random_generator, 'choice', return_value=0) as mock_choice:
        idx = analyzer_fixture._resolve_single_ambiguous_read(zero_hits, priors)
        assert idx == 0 # Mocked
        mock_choice.assert_called_once() # Check if it used random choice
    captured = capsys.readouterr()
    assert "Warning: _resolve_single_ambiguous_read called with zero hit vector." in captured.out


# --- Test resolve_ambiguous_hits_parallel ---
def test_resolve_ambiguous_hits_parallel_invalid_priors(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    amb_hits: Dict[ReadId, CountVector] = {"read1": create_cv([1]*n_strains)}
    invalid_priors = np.array([-0.1] + [0.0]*(n_strains-1)) # Negative value
    with pytest.raises(ValueError, match="prior_probabilities must be a 1D float array .* non-negative values"):
        analyzer_fixture.resolve_ambiguous_hits_parallel(amb_hits, invalid_priors)

def test_resolve_ambiguous_hits_parallel_invalid_cv_in_dict(analyzer_fixture: ClassificationAnalyzer, strain_names_fixture: List[str]):
    n_strains = len(strain_names_fixture)
    # CV with wrong length
    amb_hits_bad_cv: Dict[ReadId, CountVector] = {"read1": create_cv([1]*(n_strains-1))} 
    valid_priors = np.array([1.0/n_strains]*n_strains)
    with pytest.raises(ValueError, match="CountVector length .* must match number of strains"):
        analyzer_fixture.resolve_ambiguous_hits_parallel(amb_hits_bad_cv, valid_priors)

# --- Test combine_assignments ---
# (Existing tests for combine_assignments seem to cover type checks, will verify messages if changed)
# The type checks in combine_assignments were already quite specific.
# Adding more granular content checks for the dicts/lists:

def test_combine_assignments_invalid_content_types(analyzer_fixture: ClassificationAnalyzer):
    # Valid structure, but wrong content types
    clear_wrong_val: Dict[ReadId, Any] = {"c1": "not_an_int_index"}
    resolved_wrong_key: Dict[Any, StrainIndex] = {123: 0}
    no_hits_wrong_type: List[Any] = [12345]

    with pytest.raises(TypeError, match="clear_assignments must map ReadId \\(str\\) to StrainIndex \\(int\\)"):
        analyzer_fixture.combine_assignments(clear_wrong_val, {}, [])
    with pytest.raises(TypeError, match="resolved_ambiguous_assignments must map ReadId \\(str\\) to StrainIndex \\(int\\)"):
        analyzer_fixture.combine_assignments({}, resolved_wrong_key, [])
    with pytest.raises(TypeError, match="no_hit_read_ids must be a list of strings"):
        analyzer_fixture.combine_assignments({}, {}, no_hits_wrong_type)

# Ensure all existing tests are still valid and pass after refactoring.
# (This would be confirmed by running pytest locally)
# The mocks for random generation in probabilistic modes should still be fine.
# Fallback mechanisms in _resolve_single_ambiguous_read were made more robust,
# existing tests for these might need slight adjustments if behavior changed subtly,
# but the core idea (e.g., choice among max hits if likelihoods are zero) remains.
```
