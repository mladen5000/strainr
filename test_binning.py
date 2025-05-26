"""
Pytest unit tests for the binning module (src.strainr.binning).
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""
import pytest
import pandas as pd
import numpy as np
import pathlib
import multiprocessing as mp
from typing import List, Dict, Set, Tuple, Union, Optional 
from unittest.mock import patch, MagicMock, call, mock_open 

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Functions and types to test
from src.strainr.binning import (
    generate_table,
    get_top_strain_names,
    _extract_reads_for_strain,
    create_binned_fastq_files,
    run_binning_pipeline,
)
# Import FinalAssignmentsType from genomic_types where it's now defined
from src.strainr.genomic_types import ReadId, StrainIndex, FinalAssignmentsType 
from src.strainr.utils import open_file_transparently 

# --- Fixtures ---

@pytest.fixture
def strain_names_fixture() -> List[str]:
    return ["StrainA", "StrainB", "StrainC"]

@pytest.fixture
def unassigned_marker_fixture() -> str:
    return "NA_TEST"

@pytest.fixture
def simple_final_assignments(strain_names_fixture: List[str], unassigned_marker_fixture: str) -> FinalAssignmentsType:
    return {
        "read1": 0, "read2": 1, "read3": 0, 
        "read4": unassigned_marker_fixture, "read5": 2, 
        "read6": 99, # Invalid index
    }

@pytest.fixture
def mock_fastq_paths(tmp_path: pathlib.Path) -> Dict[str, pathlib.Path]:
    r1_path = tmp_path / "dummy_R1.fastq.gz"
    r2_path = tmp_path / "dummy_R2.fastq.gz"
    r1_path.touch() # Ensure file exists for .is_file() checks
    r2_path.touch()
    return {"r1": r1_path, "r2": r2_path}

@pytest.fixture
def tmp_output_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return tmp_path_factory.mktemp("binning_test_output")

# --- Tests for generate_table ---
# (Existing test_generate_table_basic, _empty_assignments, _all_unassigned are good)

def test_generate_table_invalid_final_assignments_content(strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="All ReadId keys in final_assignments must be non-empty strings."):
        generate_table({123: 0}, strain_names_fixture) # type: ignore
    with pytest.raises(ValueError, match="All ReadId keys in final_assignments must be non-empty strings."):
        generate_table({"": 0}, strain_names_fixture)
    with pytest.raises(TypeError, match="Assignment for ReadId 'read1' must be a non-negative integer .* or a string"):
        generate_table({"read1": -1}, strain_names_fixture)
    with pytest.raises(TypeError, match="Assignment for ReadId 'read1' must be a non-negative integer .* or a string"):
        generate_table({"read1": [0]}, strain_names_fixture) # type: ignore

def test_generate_table_invalid_all_strain_names_content(simple_final_assignments: FinalAssignmentsType):
    with pytest.raises(TypeError, match="all_strain_names must be a list of non-empty strings."):
        generate_table(simple_final_assignments, ["StrainA", 123]) # type: ignore
    with pytest.raises(TypeError, match="all_strain_names must be a list of non-empty strings."):
        generate_table(simple_final_assignments, ["StrainA", ""])
    with pytest.raises(ValueError, match="all_strain_names must contain unique names."):
        generate_table(simple_final_assignments, ["StrainA", "StrainA"])

def test_generate_table_empty_strain_names_with_indices(capsys):
    assignments: FinalAssignmentsType = {"read1": 0, "read2": 1} # Indices present
    with pytest.raises(ValueError, match="all_strain_names is empty, but final_assignments contains integer .* assignments."):
        generate_table(assignments, [])


# --- Tests for get_top_strain_names ---
# (Existing test_get_top_strain_names_basic, _exclude_unassigned_false are good)

def test_get_top_strain_names_invalid_assignments_content(strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="All ReadId keys in read_assignments must be non-empty strings."):
        get_top_strain_names({123: 0}, strain_names_fixture) # type: ignore
    with pytest.raises(TypeError, match="Assignment for ReadId 'read1' must be a non-negative int or str."):
        get_top_strain_names({"read1": -1.0}, strain_names_fixture) # type: ignore

def test_get_top_strain_names_invalid_strain_list_content(simple_final_assignments: FinalAssignmentsType):
    with pytest.raises(TypeError, match="strain_list must be a list of non-empty strings."):
        get_top_strain_names(simple_final_assignments, ["StrainA", 123]) # type: ignore
    with pytest.raises(ValueError, match="strain_list must contain unique names."):
        get_top_strain_names(simple_final_assignments, ["StrainA", "StrainA"])

def test_get_top_strain_names_invalid_unassigned_marker(simple_final_assignments: FinalAssignmentsType, strain_names_fixture: List[str]):
    with pytest.raises(ValueError, match="unassigned_marker must be a non-empty string."):
        get_top_strain_names(simple_final_assignments, strain_names_fixture, unassigned_marker="")


# --- Tests for _extract_reads_for_strain ---
# (Existing test_extract_reads_for_strain_r1_only is a good base)

def test_extract_reads_for_strain_input_file_not_found(tmp_output_dir: pathlib.Path):
    non_existent_fastq = pathlib.Path("non_existent.fastq")
    with pytest.raises(FileNotFoundError, match=f"Forward FASTQ file not found: {non_existent_fastq}"):
        _extract_reads_for_strain("StrainA", {"read1"}, non_existent_fastq, None, tmp_output_dir)

def test_extract_reads_for_strain_output_dir_not_dir(mock_fastq_paths: Dict[str, pathlib.Path], tmp_path: pathlib.Path):
    not_a_dir = tmp_path / "not_a_dir_file.txt"
    not_a_dir.touch() # Create it as a file
    with pytest.raises(FileNotFoundError, match="Output bin directory does not exist or is not a directory"):
        _extract_reads_for_strain("StrainA", {"read1"}, mock_fastq_paths["r1"], None, not_a_dir)

@patch('src.strainr.binning.SeqIO.parse', side_effect=IOError("Simulated SeqIO read error"))
def test_extract_reads_for_strain_seqio_read_error(
    mock_parse: MagicMock, mock_fastq_paths: Dict[str, pathlib.Path], tmp_output_dir: pathlib.Path, capsys
):
    _extract_reads_for_strain("StrainA", {"read1"}, mock_fastq_paths["r1"], None, tmp_output_dir)
    assert "Unexpected error processing file" in capsys.readouterr().out # Error is caught and printed

@patch('builtins.open', new_callable=mock_open) # To mock the output file writing
@patch('src.strainr.binning.SeqIO.write', side_effect=IOError("Simulated SeqIO write error"))
def test_extract_reads_for_strain_seqio_write_error(
    mock_write: MagicMock, mock_builtin_open: MagicMock,
    mock_fastq_paths: Dict[str, pathlib.Path], tmp_output_dir: pathlib.Path, capsys
):
    # Mock is_file to return True for the input fastq
    with patch('pathlib.Path.is_file', return_value=True), \
         patch('src.strainr.utils.open_file_transparently', mock_open(read_data="@read1\nACGT\n+\n!!!!\n")): # Mock reading
        
        _extract_reads_for_strain("StrainA", {"read1"}, mock_fastq_paths["r1"], None, tmp_output_dir)
    
    assert "I/O Error processing file" in capsys.readouterr().out # Error is caught and printed


# --- Tests for create_binned_fastq_files ---
# (Existing test_create_binned_fastq_files_basic is good)
def test_create_binned_fastq_files_num_bins_zero(
    tmp_output_dir: pathlib.Path, mock_fastq_paths: Dict[str, pathlib.Path], strain_names_fixture: List[str]
):
    assignment_table = pd.DataFrame(columns=strain_names_fixture) # Dummy table
    binned_strains, processes = create_binned_fastq_files(
        ["StrainA"], assignment_table, mock_fastq_paths["r1"], None, tmp_output_dir, num_bins_to_create=0
    )
    assert not binned_strains
    assert not processes

def test_create_binned_fastq_files_num_bins_negative(
    tmp_output_dir: pathlib.Path, mock_fastq_paths: Dict[str, pathlib.Path], strain_names_fixture: List[str]
):
    assignment_table = pd.DataFrame(columns=strain_names_fixture)
    with pytest.raises(ValueError, match="num_bins_to_create must be a non-negative integer."):
        create_binned_fastq_files(
            ["StrainA"], assignment_table, mock_fastq_paths["r1"], None, tmp_output_dir, num_bins_to_create=-1
        )

def test_create_binned_fastq_files_invalid_assignment_table(
    tmp_output_dir: pathlib.Path, mock_fastq_paths: Dict[str, pathlib.Path]
):
    # Table with non-string index
    bad_table_idx = pd.DataFrame({ "StrainA": [1] }, index=[123])
    with pytest.raises(ValueError, match="Read-to-strain assignment table index must consist of strings"):
        create_binned_fastq_files(["StrainA"], bad_table_idx, mock_fastq_paths["r1"], None, tmp_output_dir)

    # Table with bad column dtype
    bad_table_dtype = pd.DataFrame({ "StrainA": ["yes_assigned"] }, index=["read1"])
    with patch('src.strainr.binning.mp.Process'): # Mock process to avoid starting it
        create_binned_fastq_files(["StrainA"], bad_table_dtype, mock_fastq_paths["r1"], None, tmp_output_dir)
        # This now prints a warning and skips, test with capsys if needed.

def test_create_binned_fastq_files_invalid_fastq_paths(
    tmp_output_dir: pathlib.Path, strain_names_fixture: List[str]
):
    assignment_table = pd.DataFrame({"StrainA":[1]}, index=["r1"], columns=["StrainA","StrainB","StrainC"])
    non_existent_path = pathlib.Path("does_not_exist.fq")
    with pytest.raises(FileNotFoundError, match=f"Forward FASTQ path '{non_existent_path}' is not a valid file."):
        create_binned_fastq_files(["StrainA"], assignment_table, non_existent_path, None, tmp_output_dir)

@patch('src.strainr.binning.mp.Process')
def test_create_binned_fastq_files_process_start_error(
    mock_process_constructor: MagicMock,
    tmp_output_dir: pathlib.Path, mock_fastq_paths: Dict[str, pathlib.Path], strain_names_fixture: List[str], capsys
):
    mock_process_instance = MagicMock()
    mock_process_instance.start.side_effect = OSError("Test process start error")
    mock_process_constructor.return_value = mock_process_instance
    
    assignment_table = pd.DataFrame({"StrainA": [1]}, index=["read1"], columns=strain_names_fixture)
    
    binned_strains, processes = create_binned_fastq_files(
        ["StrainA"], assignment_table, mock_fastq_paths["r1"], None, tmp_output_dir
    )
    
    assert not processes # Process failed to start and append
    assert "Error starting binning process for strain 'StrainA'" in capsys.readouterr().out


# --- Tests for run_binning_pipeline ---
# (Existing test_run_binning_pipeline_flow is good)

def test_run_binning_pipeline_input_fastq_not_found(
    tmp_output_dir: pathlib.Path, simple_final_assignments: FinalAssignmentsType, strain_names_fixture: List[str]
):
    with pytest.raises(FileNotFoundError, match="Forward FASTQ file not found"):
        run_binning_pipeline(
            simple_final_assignments, strain_names_fixture, "nonexistent.fq", tmp_output_dir
        )

@patch('pathlib.Path.mkdir', side_effect=OSError("Test mkdir permission error"))
def test_run_binning_pipeline_output_dir_creation_error(
    mock_mkdir: MagicMock,
    mock_fastq_paths: Dict[str, pathlib.Path], 
    simple_final_assignments: FinalAssignmentsType, 
    strain_names_fixture: List[str]
):
    with pytest.raises(IOError, match="Could not create output directory"):
        run_binning_pipeline(
            simple_final_assignments, strain_names_fixture, mock_fastq_paths["r1"], "/locked_dir"
        )

@patch('src.strainr.binning.generate_table')
@patch('src.strainr.binning.get_top_strain_names')
@patch('src.strainr.binning.create_binned_fastq_files')
def test_run_binning_pipeline_subprocess_fails_exitcode(
    mock_create_binned_files: MagicMock,
    mock_get_top_names: MagicMock,
    mock_generate_table: MagicMock,
    mock_fastq_paths: Dict[str, pathlib.Path],
    tmp_output_dir: pathlib.Path,
    simple_final_assignments: FinalAssignmentsType,
    strain_names_fixture: List[str],
    capsys: pytest.CaptureFixture
):
    # Simulate create_binned_fastq_files returning a mock process that "failed"
    mock_process_instance = MagicMock(spec=mp.Process)
    mock_process_instance.pid = 1234
    mock_process_instance.exitcode = 1 # Simulate failure
    
    mock_create_binned_files.return_value = ({"StrainA"}, [mock_process_instance])
    mock_get_top_names.return_value = ["StrainA"] # Ensure some strains to process
    mock_generate_table.return_value = pd.DataFrame({"StrainA":[1]}, index=["r1"])


    run_binning_pipeline(
        simple_final_assignments, strain_names_fixture, 
        mock_fastq_paths["r1"], tmp_output_dir
    )
    
    mock_process_instance.join.assert_called_once()
    captured = capsys.readouterr()
    assert "Warning: Binning process for a strain (PID 1234) finished with exit code 1." in captured.out
```
