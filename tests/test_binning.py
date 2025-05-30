"""
Pytest unit tests for the binning module (strainr.binning).
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""

import pathlib
from typing import Dict, List  # Added Optional
from unittest.mock import MagicMock, mock_open, patch  # Added mock_open

import pandas as pd
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Functions and types to test
from strainr.binning import (
    FinalAssignmentsType,
)  # Assuming this is defined in binning.py or imported there
from strainr.binning import (
    _extract_reads_for_strain,
    create_binned_fastq_files,
    generate_table,
    get_top_strain_names,
    run_binning_pipeline,
)

# --- Fixtures ---


@pytest.fixture
def strain_names_fixture() -> List[str]:
    """Provides a default list of strain names."""
    return ["StrainA", "StrainB", "StrainC"]


@pytest.fixture
def unassigned_marker_fixture() -> str:
    """Provides a default unassigned marker string."""
    return "NA_TEST"  # Use a distinct marker for tests


@pytest.fixture
def simple_final_assignments(
    strain_names_fixture: List[str], unassigned_marker_fixture: str
) -> FinalAssignmentsType:
    """Simple FinalAssignmentsType data for testing."""
    # Assumes StrainIndex corresponds to index in strain_names_fixture
    return {
        "read1": 0,  # StrainA
        "read2": 1,  # StrainB
        "read3": 0,  # StrainA
        "read4": unassigned_marker_fixture,
        "read5": 2,  # StrainC
        "read6": 99,  # Invalid index, should be handled by generate_table
    }


@pytest.fixture
def mock_fastq_paths(tmp_path: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Creates dummy FASTQ file paths in a temporary directory for path existence checks."""
    r1_path = tmp_path / "dummy_R1.fastq.gz"
    r2_path = tmp_path / "dummy_R2.fastq.gz"
    # Create empty files if needed by tests that don't fully mock Path.is_file()
    # r1_path.touch()
    # r2_path.touch()
    return {"r1": r1_path, "r2": r2_path}


@pytest.fixture
def tmp_output_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Provides a temporary output directory."""
    return tmp_path_factory.mktemp("binning_test_output")


# --- Tests for generate_table ---


def test_generate_table_basic(
    simple_final_assignments: FinalAssignmentsType, strain_names_fixture: List[str]
):
    df = generate_table(simple_final_assignments, strain_names_fixture)

    assert isinstance(df, pd.DataFrame)
    # Read IDs are from simple_final_assignments, including "read6" with invalid index
    expected_index = ["read1", "read2", "read3", "read4", "read5", "read6"]
    assert sorted(list(df.index)) == sorted(expected_index)
    assert list(df.columns) == strain_names_fixture

    expected_data = {  # Based on simple_final_assignments
        "StrainA": [1, 0, 1, 0, 0, 0],  # read1, read3
        "StrainB": [0, 1, 0, 0, 0, 0],  # read2
        "StrainC": [0, 0, 0, 0, 1, 0],  # read5
    }
    # Recreate expected_df with the full index from simple_final_assignments
    expected_df = pd.DataFrame(0, index=expected_index, columns=strain_names_fixture)
    for read_id, strain_idx_or_marker in simple_final_assignments.items():
        if isinstance(strain_idx_or_marker, int) and 0 <= strain_idx_or_marker < len(
            strain_names_fixture
        ):
            strain_name = strain_names_fixture[strain_idx_or_marker]
            expected_df.loc[read_id, strain_name] = 1

    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_generate_table_empty_assignments(strain_names_fixture: List[str]):
    df = generate_table({}, strain_names_fixture)
    assert df.empty
    assert list(df.columns) == strain_names_fixture
    assert len(df.index) == 0


def test_generate_table_empty_strain_names(
    simple_final_assignments: FinalAssignmentsType,
):
    df = generate_table(simple_final_assignments, [])
    df = pd.Series(df, index=[])
    assert df.shape == (len(simple_final_assignments), 0)  # No columns
    assert sorted(list(df.index)) == sorted(list(simple_final_assignments.keys()))


def test_generate_table_all_unassigned(
    strain_names_fixture: List[str], unassigned_marker_fixture: str
):
    assignments: FinalAssignmentsType = {
        "read1": unassigned_marker_fixture,
        "read2": unassigned_marker_fixture,
    }
    df = generate_table(assignments, strain_names_fixture)
    expected_df = pd.DataFrame(
        0, index=["read1", "read2"], columns=strain_names_fixture
    )
    pd.testing.assert_frame_equal(df, expected_df)


# --- Tests for get_top_strain_names ---


def test_get_top_strain_names_basic(
    simple_final_assignments: FinalAssignmentsType,
    strain_names_fixture: List[str],
    unassigned_marker_fixture: str,
):
    # simple_final_assignments: {"read1":0(A),"read2":1(B),"read3":0(A),"read4":NA,"read5":2(C),"read6":99(Invalid)}
    # Expected counts: StrainA: 2, StrainB: 1, StrainC: 1. "NA" and invalid 99 ignored by default.
    top_strains = get_top_strain_names(
        simple_final_assignments, strain_names_fixture, unassigned_marker_fixture
    )

    # Order for ties (StrainB, StrainC) can vary, so check presence and primary order
    assert top_strains[0] == "StrainA"
    assert "StrainB" in top_strains
    assert "StrainC" in top_strains
    assert len(top_strains) == 3


def test_get_top_strain_names_exclude_unassigned_false(
    strain_names_fixture: List[str], unassigned_marker_fixture: str
):
    # Strain names: ["StrainA", "StrainB", "StrainC"]
    # Unassigned marker: "NA_TEST"
    assignments: FinalAssignmentsType = {
        "read1": 0,  # StrainA: 1
        "read2": unassigned_marker_fixture,  # NA_TEST: 2
        "read3": unassigned_marker_fixture,
        "read4": 1,  # StrainB: 1
    }

    # Scenario 1: unassigned_marker is NOT in strain_list, exclude_unassigned=False
    # 'NA_TEST' is counted as itself if not excluded.
    top_strains = get_top_strain_names(
        assignments,
        strain_names_fixture,
        unassigned_marker_fixture,
        exclude_unassigned=False,
    )
    # Expected counts: NA_TEST:2, StrainA:1, StrainB:1.
    # The function only returns names present in strain_list, or the marker if not excluded and not in strain_list.
    # The current implementation of get_top_strain_names filters results to those in strain_list OR the unassigned_marker
    # if not excluded.
    assert top_strains == [
        unassigned_marker_fixture,
        "StrainA",
        "StrainB",
    ] or top_strains == [unassigned_marker_fixture, "StrainB", "StrainA"]

    # Scenario 2: unassigned_marker IS in strain_list (e.g. strain_list = ["StrainA", "NA_TEST"])
    custom_strain_list = ["StrainA", "StrainB", unassigned_marker_fixture]
    assignments_sc2: FinalAssignmentsType = {
        "read1": 0,  # StrainA: 1
        "read2": unassigned_marker_fixture,  # NA_TEST (as a valid strain): 2
        "read3": unassigned_marker_fixture,
        "read4": 1,  # StrainB: 1
    }
    top_strains_sc2 = get_top_strain_names(
        assignments_sc2,
        custom_strain_list,
        unassigned_marker_fixture,
        exclude_unassigned=False,
    )
    assert top_strains_sc2[0] == unassigned_marker_fixture  # NA_TEST is most abundant
    assert "StrainA" in top_strains_sc2
    assert "StrainB" in top_strains_sc2
    assert len(top_strains_sc2) == 3

    top_strains_sc2_excluded = get_top_strain_names(
        assignments_sc2,
        custom_strain_list,
        unassigned_marker_fixture,
        exclude_unassigned=True,
    )
    assert unassigned_marker_fixture not in top_strains_sc2_excluded
    assert "StrainA" in top_strains_sc2_excluded
    assert "StrainB" in top_strains_sc2_excluded
    assert len(top_strains_sc2_excluded) == 2


# --- Tests for _extract_reads_for_strain ---


@patch("strainr.utils.open_file_transparently", new_callable=mock_open)
@patch("strainr.binning.SeqIO.parse")
@patch("strainr.binning.SeqIO.write")
@patch("pathlib.Path.is_file")
def test_extract_reads_for_strain_r1_only(
    mock_is_file: MagicMock,
    mock_seqio_write: MagicMock,
    mock_seqio_parse: MagicMock,
    mock_open_transparently: MagicMock,  # This is from utils, used by binning
    tmp_output_dir: pathlib.Path,
    mock_fastq_paths: Dict[str, pathlib.Path],
):
    mock_is_file.return_value = True  # Assume R1 file exists
    # mock_open_transparently is for reading the input FASTQ
    # We also need to mock builtins.open for writing the output FASTQ

    r1_path = mock_fastq_paths["r1"]

    read_ids_for_strain = {"read_A1", "read_A2"}
    all_seq_records = [
        SeqRecord(Seq("ATGC"), id="read_A1"),
        SeqRecord(Seq("CGTA"), id="read_B1"),
        SeqRecord(Seq("TTTT"), id="read_A2"),
    ]
    mock_seqio_parse.return_value = iter(all_seq_records)  # parse returns an iterator
    mock_seqio_write.return_value = 2  # Simulate 2 records written

    with patch("builtins.open", mock_open()) as mock_builtin_write_open:
        _extract_reads_for_strain(
            "StrainA", read_ids_for_strain, r1_path, None, tmp_output_dir
        )

    mock_open_transparently.assert_called_once_with(r1_path, mode="rt")
    mock_seqio_parse.assert_called_once_with(
        mock_open_transparently.return_value, "fastq"
    )

    # Check records passed to SeqIO.write
    assert mock_seqio_write.call_count == 1
    written_records_call = mock_seqio_write.call_args[0][0]
    assert len(written_records_call) == 2
    assert written_records_call[0].id == "read_A1"
    assert written_records_call[1].id == "read_A2"

    expected_output_path = tmp_output_dir / "bin.StrainA_R1.fastq"
    mock_builtin_write_open.assert_called_once_with(expected_output_path, "w")


# --- Tests for create_binned_fastq_files ---


@patch("strainr.binning.mp.Process")
@patch("pathlib.Path.mkdir")  # Mock mkdir to avoid actual directory creation
def test_create_binned_fastq_files_basic(
    mock_mkdir: MagicMock,
    mock_process: MagicMock,
    strain_names_fixture: List[str],
    tmp_output_dir: pathlib.Path,
    mock_fastq_paths: Dict[str, pathlib.Path],
    unassigned_marker_fixture: str,
):
    top_strains = ["StrainA", "StrainB"]
    read_ids = ["read1_A", "read2_A", "read1_B"]
    data = {"StrainA": [1, 1, 0], "StrainB": [0, 0, 1], "StrainC": [0, 0, 0]}
    assignment_table = pd.DataFrame(data, index=read_ids, columns=strain_names_fixture)

    binned_strains_set, processes_list = create_binned_fastq_files(
        top_strain_names=top_strains,
        read_to_strain_assignment_table=assignment_table,
        forward_fastq_path=mock_fastq_paths["r1"],
        reverse_fastq_path=mock_fastq_paths["r2"],
        output_dir=tmp_output_dir,
        num_bins_to_create=2,
        unassigned_marker=unassigned_marker_fixture,
    )

    mock_mkdir.assert_called_with(exist_ok=True, parents=True)
    assert mock_process.call_count == 2
    assert len(processes_list) == 2
    assert binned_strains_set == {"StrainA", "StrainB"}

    # Check args for StrainA process
    args_strain_a = mock_process.call_args_list[0].kwargs["args"]
    assert args_strain_a[0] == "StrainA"
    assert args_strain_a[1] == {"read1_A", "read2_A"}

    # Check args for StrainB process
    args_strain_b = mock_process.call_args_list[1].kwargs["args"]
    assert args_strain_b[0] == "StrainB"
    assert args_strain_b[1] == {"read1_B"}


# --- Tests for run_binning_pipeline ---


@patch("strainr.binning.create_binned_fastq_files", return_value=(set(), []))
@patch("strainr.binning.get_top_strain_names")
@patch("strainr.binning.generate_table")
def test_run_binning_pipeline_flow(
    mock_generate_table: MagicMock,
    mock_get_top_names: MagicMock,
    mock_create_binned_files: MagicMock,
    strain_names_fixture: List[str],
    simple_final_assignments: FinalAssignmentsType,
    mock_fastq_paths: Dict[str, pathlib.Path],
    tmp_output_dir: pathlib.Path,
    unassigned_marker_fixture: str,
):
    dummy_assignment_table = pd.DataFrame(index=["read1"], columns=strain_names_fixture)
    mock_generate_table.return_value = dummy_assignment_table

    top_strains_list = ["StrainA", "StrainB"]
    mock_get_top_names.return_value = top_strains_list

    fwd_path_str = str(mock_fastq_paths["r1"])  # Test with string path
    out_dir = tmp_output_dir

    run_binning_pipeline(
        final_assignments=simple_final_assignments,
        all_strain_names=strain_names_fixture,
        forward_reads_fastq=fwd_path_str,
        output_directory=out_dir,
        num_top_strains_to_bin=2,
        reverse_reads_fastq=None,  # Test R1 only case for pipeline
        unassigned_marker=unassigned_marker_fixture,
    )

    mock_generate_table.assert_called_once_with(
        simple_final_assignments, strain_names_fixture
    )
    mock_get_top_names.assert_called_once_with(
        read_assignments=simple_final_assignments,
        strain_list=strain_names_fixture,
        unassigned_marker=unassigned_marker_fixture,
        exclude_unassigned=True,  # Default in run_binning_pipeline
    )
    mock_create_binned_files.assert_called_once_with(
        top_strain_names=top_strains_list,
        read_to_strain_assignment_table=dummy_assignment_table,
        forward_fastq_path=pathlib.Path(fwd_path_str),
        reverse_fastq_path=None,
        output_dir=out_dir,
        num_bins_to_create=2,
        unassigned_marker=unassigned_marker_fixture,
    )
