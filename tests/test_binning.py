# Standard library imports
import pytest
import multiprocessing as mp
import pathlib
from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Union,
    Optional,
)
import gzip
import traceback
from unittest.mock import patch, MagicMock, mock_open, call

# Third-party imports
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Local application/library specific imports
from src.strainr.utils import open_file_transparently
from src.strainr.genomic_types import (
    ReadId,
    FinalAssignmentsType,
)
from src.strainr.binning import (
    generate_table,
    get_top_strain_names,
    _extract_reads_for_strain,
    create_binned_fastq_files,
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
    return "NA_TEST"

@pytest.fixture
def simple_final_assignments(
    strain_names_fixture: List[str], unassigned_marker_fixture: str
) -> FinalAssignmentsType:
    """Provides a sample FinalAssignmentsType dictionary."""
    # StrainA is index 0, StrainB is index 1, StrainC is index 2
    return {
        "read1": 0,  # StrainA
        "read2": 1,  # StrainB
        "read3": 0,  # StrainA
        "read4": unassigned_marker_fixture,
        "read5": 0,  # StrainA
        "read6": 2,  # StrainC
    }

@pytest.fixture
def tmp_output_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a temporary output directory for binning tests."""
    output_dir = tmp_path / "binning_test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

@pytest.fixture
def mock_fastq_paths(tmp_path: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Creates mock FASTQ file paths and touches the files to make them exist."""
    r1_path = tmp_path / "dummy_R1.fastq.gz"
    r2_path = tmp_path / "dummy_R2.fastq"
    r1_path.touch()
    r2_path.touch()
    return {"r1": r1_path, "r2": r2_path}


# --- Tests for generate_table ---

def test_generate_table_basic(
    simple_final_assignments: FinalAssignmentsType, strain_names_fixture: List[str]
):
    df = generate_table(simple_final_assignments, strain_names_fixture)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == strain_names_fixture
    assert len(df) == len(simple_final_assignments)
    assert df.loc["read1", "StrainA"] == 1
    assert df.loc["read2", "StrainB"] == 1
    assert df.loc["read4", "StrainA"] == 0

def test_generate_table_empty_input():
    df = generate_table({}, [])
    assert df.empty
    df_no_strains = generate_table({"read1": "NA"}, [])
    assert df_no_strains.index.tolist() == ["read1"]
    assert df_no_strains.columns.empty

def test_generate_table_empty_strain_names(
    simple_final_assignments: FinalAssignmentsType,
):
    with pytest.raises(
        ValueError,
        match="all_strain_names is empty, but final_assignments contains integer .* assignments."
    ):
        generate_table(simple_final_assignments, [])

def test_generate_table_all_unassigned(
    strain_names_fixture: List[str], unassigned_marker_fixture: str
):
    assignments: FinalAssignmentsType = {
        "read1": unassigned_marker_fixture,
        "read2": unassigned_marker_fixture,
    }
    df = generate_table(assignments, strain_names_fixture)
    expected_df = pd.DataFrame(
        0, index=["read1", "read2"], columns=strain_names_fixture, dtype=np.uint8
    )
    pd.testing.assert_frame_equal(df, expected_df)

# --- Tests for get_top_strain_names ---

def test_get_top_strain_names_basic(
    simple_final_assignments: FinalAssignmentsType, strain_names_fixture: List[str]
):
    top_strains = get_top_strain_names(
        simple_final_assignments, strain_names_fixture, exclude_unassigned=True
    )
    assert top_strains == ["StrainA", "StrainB", "StrainC"]

def test_get_top_strain_names_include_unassigned(
    simple_final_assignments: FinalAssignmentsType,
    strain_names_fixture: List[str],
    unassigned_marker_fixture: str,
):
    top_strains = get_top_strain_names(
        simple_final_assignments,
        strain_names_fixture,
        unassigned_marker=unassigned_marker_fixture,
        exclude_unassigned=False,
    )
    assert len(top_strains) == 4
    assert set(top_strains) == {"StrainA", "StrainB", "StrainC", unassigned_marker_fixture}


# --- Tests for _extract_reads_for_strain ---

@patch("src.strainr.binning.open_file_transparently", new_callable=mock_open)
@patch("src.strainr.binning.SeqIO.parse")
@patch("src.strainr.binning.SeqIO.write")
def test_extract_reads_for_strain_r1_only(
    mock_seqio_write: MagicMock,
    mock_seqio_parse: MagicMock,
    mock_open_transp: MagicMock,
    tmp_output_dir: pathlib.Path,
    mock_fastq_paths: Dict[str, pathlib.Path],
):
    r1_path = mock_fastq_paths["r1"]
    with patch.object(pathlib.Path, 'is_file', return_value=True): # Ensure is_file() passes for the dummy path
        read_ids_for_strain = {"read_A1", "read_A2"}
        all_seq_records = [
            SeqRecord(Seq("ATGC"), id="read_A1"),
            SeqRecord(Seq("CGTA"), id="read_B1"),
            SeqRecord(Seq("TTTT"), id="read_A2"),
        ]
        mock_seqio_parse.return_value = iter(all_seq_records)
        mock_seqio_write.return_value = 2

        with patch("builtins.open", new_callable=mock_open) as mock_builtin_write_open:
            _extract_reads_for_strain(
                "StrainA", read_ids_for_strain, r1_path, None, tmp_output_dir
            )

        mock_open_transp.assert_called_once_with(r1_path, mode="rt")
        mock_builtin_write_open.assert_called_once_with(
            tmp_output_dir / "bin.StrainA_R1.fastq", "w"
        )
        mock_seqio_write.assert_called_once()
        written_records = mock_seqio_write.call_args[0][0]
        assert len(written_records) == 2
        assert written_records[0].id == "read_A1"
        assert written_records[1].id == "read_A2"

# --- Tests for create_binned_fastq_files ---

@patch("src.strainr.binning.mp.Process")
@patch("pathlib.Path.mkdir")
def test_create_binned_fastq_files_basic(
    mock_mkdir: MagicMock,
    mock_process_class: MagicMock,
    strain_names_fixture: List[str],
    tmp_output_dir: pathlib.Path,
    mock_fastq_paths: Dict[str, pathlib.Path],
    unassigned_marker_fixture: str,
):
    mock_process_instance1 = MagicMock()
    mock_process_instance2 = MagicMock()
    mock_process_class.side_effect = [mock_process_instance1, mock_process_instance2]

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
    assert binned_strains_set == {"StrainA", "StrainB"}
    assert len(processes_list) == 2
    assert processes_list[0] is mock_process_instance1
    assert processes_list[1] is mock_process_instance2

    mock_process_class.assert_has_calls(
        [
            call(target=_extract_reads_for_strain, args=("StrainA", {"read1_A", "read2_A"}, mock_fastq_paths["r1"], mock_fastq_paths["r2"], tmp_output_dir / "bins")),
            call(target=_extract_reads_for_strain, args=("StrainB", {"read1_B"}, mock_fastq_paths["r1"], mock_fastq_paths["r2"], tmp_output_dir / "bins")),
        ],
        any_order=True
    )
    mock_process_instance1.start.assert_called_once()
    mock_process_instance2.start.assert_called_once()


# --- Tests for run_binning_pipeline ---

@patch("src.strainr.binning.create_binned_fastq_files", return_value=(set(), []))
@patch("src.strainr.binning.get_top_strain_names")
@patch("src.strainr.binning.generate_table")
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

    fwd_path_str = str(mock_fastq_paths["r1"])
    out_dir = tmp_output_dir

    run_binning_pipeline(
        final_assignments=simple_final_assignments,
        all_strain_names=strain_names_fixture,
        read_assignments=simple_final_assignments,
        forward_reads_fastq=fwd_path_str,
        output_directory=out_dir,
        num_top_strains_to_bin=2,
        reverse_reads_fastq=None,
        unassigned_marker=unassigned_marker_fixture,
    )

    assert mock_generate_table.call_count == 2
    mock_get_top_names.assert_called_once_with(
        read_assignments=simple_final_assignments,
        strain_list=strain_names_fixture,
        unassigned_marker=unassigned_marker_fixture,
        exclude_unassigned=True,
    )
    mock_create_binned_files.assert_called_once_with(
        top_strain_names=top_strains_list,
        read_to_strain_assignment_table=dummy_assignment_table,
        forward_fastq_path=pathlib.Path(fwd_path_str).resolve(),
        reverse_fastq_path=None,
        output_dir=out_dir.resolve(),
        num_bins_to_create=2,
        unassigned_marker=unassigned_marker_fixture,
    )
