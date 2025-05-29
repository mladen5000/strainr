"""
Pytest unit tests for utility functions in src.strainr.utils.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""

import pytest
import pandas as pd
import numpy as np
import pathlib
import pickle
import gzip
import io
from typing import List, Dict, Union, Tuple  # TextIO for type hint
from unittest.mock import patch, MagicMock, mock_open

from Bio.Seq import Seq

# Assuming src.strainr.* is in PYTHONPATH or tests are run from a suitable root
from src.strainr.utils import (
    open_file_transparently,
    get_canonical_kmer,
    pickle_intermediate_results,
    save_classification_results_to_dataframe,
)
# Assuming these types might be used in dummy data for pickle_intermediate_results
# from src.strainr.genomic_types import ReadId, CountVector, StrainName, StrainIndex

# --- Fixtures ---


@pytest.fixture
def sample_text_content_fixture() -> str:  # Renamed for clarity
    return "Hello, StrainR!\nThis is a test file.\nLine 3."


@pytest.fixture
def plain_text_file_fixture(
    tmp_path: pathlib.Path, sample_text_content_fixture: str
) -> pathlib.Path:  # Renamed
    file_path = tmp_path / "test_plain_utils.txt"
    with open(file_path, "w") as f:
        f.write(sample_text_content_fixture)
    return file_path


@pytest.fixture
def gzipped_text_file_fixture(
    tmp_path: pathlib.Path, sample_text_content_fixture: str
) -> pathlib.Path:  # Renamed
    file_path = tmp_path / "test_gzipped_utils.txt.gz"
    with gzip.open(file_path, "wt") as f:  # wt for text mode with gzip
        f.write(sample_text_content_fixture)
    return file_path


@pytest.fixture
def mock_output_dir_utils(tmp_path: pathlib.Path) -> pathlib.Path:  # Renamed
    """Provides a mock output directory within the temporary path for utils tests."""
    out_dir = tmp_path / "output_utils"
    # No need to mkdir here, functions should do it with exist_ok=True
    return out_dir


# --- Tests for open_file_transparently ---


def test_open_plain_text_file(
    plain_text_file_fixture: pathlib.Path, sample_text_content_fixture: str
):
    with open_file_transparently(plain_text_file_fixture, mode="rt") as f:
        content = f.read()
        assert content == sample_text_content_fixture
        assert isinstance(f, io.TextIOWrapper)


def test_open_gzipped_text_file(
    gzipped_text_file_fixture: pathlib.Path, sample_text_content_fixture: str
):
    with open_file_transparently(gzipped_text_file_fixture, mode="rt") as f:
        content = f.read()
        assert content == sample_text_content_fixture
        # The object returned by gzip.open in text mode is a GzipFile object that behaves like TextIO
        assert hasattr(f, "read") and hasattr(
            f, "readline"
        )  # Check for TextIO attributes


def test_open_file_not_found_error():  # Renamed for clarity
    with pytest.raises(FileNotFoundError, match="File not found:"):
        open_file_transparently("non_existent_utils_file.txt")


@patch("builtins.open", side_effect=IOError("Permission denied for test"))
def test_open_plain_io_error_mocked(
    mock_builtin_open: MagicMock, plain_text_file_fixture: pathlib.Path
):  # Renamed
    # Ensure mimetypes does not guess "gzip" for this plain file
    with patch("mimetypes.guess_type", return_value=("text/plain", None)):
        with pytest.raises(
            IOError,
            match=f"Error opening file {plain_text_file_fixture} with mode 'rt': Permission denied for test",
        ):
            open_file_transparently(plain_text_file_fixture)


@patch("gzip.open", side_effect=IOError("Gzip processing error for test"))
def test_open_gzipped_io_error_mocked(
    mock_gzip_open: MagicMock, gzipped_text_file_fixture: pathlib.Path
):  # Renamed
    # Ensure mimetypes correctly guesses "gzip"
    with patch("mimetypes.guess_type", return_value=("application/gzip", "gzip")):
        with pytest.raises(
            IOError,
            match=f"Error opening file {gzipped_text_file_fixture} with mode 'rt': Gzip processing error for test",
        ):
            open_file_transparently(gzipped_text_file_fixture)


def test_open_file_invalid_path_type_error():  # Renamed
    with pytest.raises(TypeError, match="file_path must be a string or pathlib.Path"):
        open_file_transparently(12345)  # type: ignore


# --- Tests for get_canonical_kmer ---


@pytest.mark.parametrize(
    "kmer_str, expected_canonical_str",
    [
        ("AGCT", "AGCT"),  # RevComp is TCGATAGC -> AGCT. Original: AGCT vs TCGA -> AGCT
        (
            "TCGA",
            "TCGA",
        ),  # RevComp is TCGATAGC -> AGCT. Original: TCGA vs AGCT -> AGCT. *Corrected expectation*
        ("AAAA", "AAAA"),
        ("TTTT", "AAAA"),
        ("ATAT", "ATAT"),
        ("GATC", "GATC"),
        ("ACGT", "ACGT"),
        ("TGCA", "TGCA"),  # RevComp is TGCA. Original: TGCA vs TGCA -> TGCA
        ("AGGC", "AGGC"),  # RevComp is GCCT. Original: AGGC vs GCCT -> AGGC
        (
            "GCCT",
            "GCCT",
        ),  # RevComp is AGGC. Original: GCCT vs AGGC -> AGGC. *Corrected expectation*
    ],
)
def test_get_canonical_kmer(kmer_str: str, expected_canonical_str: str):
    # Correction for TCGA: Seq("TCGA").reverse_complement() is Seq("TCGA"). Canonical should be "TCGA" if comparing "TCGA" and "TCGA".
    # The logic is min(kmer, rev_comp_kmer). For TCGA, rev_comp is TCGA. So TCGA is canonical.
    # For GCCT, rev_comp is AGGC. str("GCCT") > str("AGGC"), so AGGC is canonical.
    # The provided expected values were slightly off for these due to complex comparisons. Let's re-verify.
    # Seq("TCGA").reverse_complement() == Seq("TCGA")
    # Seq("GCCT").reverse_complement() == Seq("AGGC")
    # The test cases need to be accurate to the definition: min(lexicographical(kmer), lexicographical(rev_comp_kmer))

    # Re-evaluating expected values based on strict lexicographical comparison of kmer and its reverse complement
    corrected_expectations = {
        "AGCT": "AGCT",  # rev_comp("AGCT") is "AGCT". min("AGCT", "AGCT") is "AGCT"
        "TCGA": "TCGA",  # rev_comp("TCGA") is "TCGA". min("TCGA", "TCGA") is "TCGA"
        "AAAA": "AAAA",
        "TTTT": "AAAA",  # rev_comp("TTTT") is "AAAA". min("TTTT", "AAAA") is "AAAA"
        "ATAT": "ATAT",
        "GATC": "GATC",
        "ACGT": "ACGT",
        "TGCA": "TGCA",
        "AGGC": "AGGC",  # rev_comp("AGGC") is "GCCT". min("AGGC", "GCCT") is "AGGC"
        "GCCT": "AGGC",  # rev_comp("GCCT") is "AGGC". min("GCCT", "AGGC") is "AGGC"
    }
    kmer_seq = Seq(kmer_str)
    canonical_kmer = get_canonical_kmer(kmer_seq)
    assert str(canonical_kmer) == corrected_expectations[kmer_str]


# --- Tests for pickle_intermediate_results ---


def test_pickle_intermediate_results_success(
    mock_output_dir_utils: pathlib.Path,
):  # Renamed fixture
    raw_scores_data: List[Tuple[str, np.ndarray]] = [
        ("read1", np.array([10, 5], dtype=np.uint8)),
        ("read2", np.array([3, 12], dtype=np.uint8)),
    ]
    final_assign_data: Dict[str, Union[str, int]] = {
        "read1": 0,
        "read2": "StrainB",
        "read3": "NA_TEST",  # Use consistent unassigned marker
    }

    pickle_intermediate_results(
        mock_output_dir_utils, raw_scores_data, final_assign_data
    )

    raw_path = mock_output_dir_utils / "raw_kmer_scores.pkl"
    final_path = mock_output_dir_utils / "final_read_assignments.pkl"

    assert raw_path.exists() and raw_path.is_file()
    assert final_path.exists() and final_path.is_file()

    with open(raw_path, "rb") as f_raw:
        loaded_raw_scores = pickle.load(f_raw)
    assert len(loaded_raw_scores) == len(raw_scores_data)
    for (orig_id, orig_arr), (load_id, load_arr) in zip(
        raw_scores_data, loaded_raw_scores
    ):
        assert orig_id == load_id
        np.testing.assert_array_equal(orig_arr, load_arr)

    with open(final_path, "rb") as f_final:
        loaded_final_assign = pickle.load(f_final)
    assert loaded_final_assign == final_assign_data


@patch("pickle.dump", side_effect=pickle.PicklingError("Test Pickling Error"))
def test_pickle_intermediate_results_pickling_error(
    mock_pickle_dump: MagicMock, mock_output_dir_utils: pathlib.Path
):
    with pytest.raises(
        IOError, match="Error pickling intermediate results.*Test Pickling Error"
    ):
        pickle_intermediate_results(
            mock_output_dir_utils, [("r1", np.array([1]))], {"r1": 0}
        )


@patch("builtins.open", new_callable=mock_open)
def test_pickle_intermediate_results_io_error_on_open_mocked(
    mock_open_call: MagicMock, mock_output_dir_utils: pathlib.Path
):
    mock_open_call.side_effect = IOError("Test Cannot open file")
    with pytest.raises(
        IOError, match="Error pickling intermediate results.*Test Cannot open file"
    ):
        pickle_intermediate_results(mock_output_dir_utils, [], {})


# --- Tests for save_classification_results_to_dataframe ---


@pytest.fixture
def sample_intermediate_scores_utils() -> Dict[
    str, Union[List[float], np.ndarray]
]:  # Renamed
    return {
        "readA_utils": [0.7, 0.2, 0.1],
        "readB_utils": np.array([0.1, 0.8, 0.1], dtype=float),
        "readC_utils": [0.3, 0.3, 0.4],
    }


@pytest.fixture
def sample_final_assignments_utils() -> Dict[str, str]:  # Renamed
    return {
        "readA_utils": "StrainX_utils",
        "readB_utils": "StrainY_utils",
        "readC_utils": "StrainZ_utils",
        "readD_no_score_utils": "StrainX_utils",
    }


@pytest.fixture
def sample_strain_names_utils() -> List[str]:  # Renamed
    return ["StrainX_utils", "StrainY_utils", "StrainZ_utils"]


def test_save_results_to_dataframe_success(
    mock_output_dir_utils: pathlib.Path,
    sample_intermediate_scores_utils: Dict[str, Union[List[float], np.ndarray]],
    sample_final_assignments_utils: Dict[str, str],
    sample_strain_names_utils: List[str],
):
    save_classification_results_to_dataframe(
        mock_output_dir_utils,
        sample_intermediate_scores_utils,
        sample_final_assignments_utils,
        sample_strain_names_utils,
    )

    output_file = mock_output_dir_utils / "classification_results_table.pkl"
    assert output_file.exists() and output_file.is_file()

    loaded_df = pd.read_pickle(output_file)

    expected_columns = sample_strain_names_utils + ["final_assigned_strain"]
    assert all(col in loaded_df.columns for col in expected_columns)
    assert sorted(list(loaded_df.index)) == sorted(
        list(sample_intermediate_scores_utils.keys())
    )

    for strain_name_col in sample_strain_names_utils:
        assert loaded_df[strain_name_col].dtype == float

    for read_id, assigned_strain in sample_final_assignments_utils.items():
        if read_id in loaded_df.index:
            assert loaded_df.loc[read_id, "final_assigned_strain"] == assigned_strain

    assert loaded_df.loc["readA_utils", "StrainX_utils"] == pytest.approx(0.7)
    np.testing.assert_array_almost_equal(
        loaded_df.loc["readB_utils", sample_strain_names_utils].values,
        np.array([0.1, 0.8, 0.1]),
    )


@patch(
    "pandas.DataFrame.to_pickle",
    side_effect=pickle.PicklingError("Test DF Pickle error"),
)
def test_save_results_to_dataframe_pickling_error_mocked(  # Renamed
    mock_to_pickle: MagicMock,
    mock_output_dir_utils: pathlib.Path,
    sample_intermediate_scores_utils: Dict[str, Union[List[float], np.ndarray]],
    sample_final_assignments_utils: Dict[str, str],
    sample_strain_names_utils: List[str],
):
    with pytest.raises(
        IOError,
        match="Error saving classification results to DataFrame.*Test DF Pickle error",
    ):
        save_classification_results_to_dataframe(
            mock_output_dir_utils,
            sample_intermediate_scores_utils,
            sample_final_assignments_utils,
            sample_strain_names_utils,
        )


@patch(
    "pandas.DataFrame.from_dict",
    side_effect=ValueError("Test Inconsistent data for DF"),
)
def test_save_results_to_dataframe_from_dict_error_mocked(  # Renamed
    mock_from_dict: MagicMock,
    mock_output_dir_utils: pathlib.Path,
    sample_final_assignments_utils: Dict[str, str],
    sample_strain_names_utils: List[str],
):
    dummy_scores = {"read1_utils": [0.1, 0.2, 0.3]}
    with pytest.raises(
        IOError,
        match="Error saving classification results to DataFrame.*Test Inconsistent data for DF",
    ):
        save_classification_results_to_dataframe(
            mock_output_dir_utils,
            dummy_scores,
            sample_final_assignments_utils,
            sample_strain_names_utils,
        )


def test_save_results_to_dataframe_non_float_castable_scores_error(  # Renamed
    mock_output_dir_utils: pathlib.Path,
    sample_final_assignments_utils: Dict[str, str],
    sample_strain_names_utils: List[str],
):
    bad_scores = {"read1_utils": ["not_a_float_val", 0.2, 0.3]}
    with pytest.raises(IOError, match="Intermediate scores contain non-numeric values"):
        save_classification_results_to_dataframe(
            mock_output_dir_utils,
            bad_scores,
            sample_final_assignments_utils,
            sample_strain_names_utils,
        )
