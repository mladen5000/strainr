"""
Pytest unit tests for strainr.classify module.
"""
import pytest
import pathlib
import numpy as np # Added numpy import
from typing import List, Tuple, Generator
from collections import Counter # Added Counter import
from unittest.mock import patch, mock_open, MagicMock

from strainr.classify import SequenceFileProcessor, DEFAULT_CHUNK_SIZE
from strainr.genomic_types import ReadId # Assuming ReadId is just str

# --- Fixtures for FASTA/FASTQ content ---

@pytest.fixture
def fasta_content_simple() -> str:
    return """>read1 description1
ACGTACGT
>read2 description2
CGTACGTA
>read3 description3
GTACGTAC
"""

@pytest.fixture
def fastq_content_simple() -> str:
    return """@read1 description1
ACGTACGT
+
FFFFFFFF
@read2 description2
CGTACGTA
+
GGGGGGGG
@read3 description3
GTACGTAC
+
HHHHHHHH
"""

@pytest.fixture
def fasta_file_simple(tmp_path: pathlib.Path, fasta_content_simple: str) -> pathlib.Path:
    file_path = tmp_path / "simple.fasta"
    file_path.write_text(fasta_content_simple)
    return file_path

@pytest.fixture
def fastq_file_simple(tmp_path: pathlib.Path, fastq_content_simple: str) -> pathlib.Path:
    file_path = tmp_path / "simple.fastq"
    file_path.write_text(fastq_content_simple)
    return file_path

# --- Tests for SequenceFileProcessor.parse_sequence_files (Chunking) ---

def run_parser_and_collect_chunks(
    parser_gen: Generator[List[Tuple[ReadId, bytes, bytes]], None, None]
) -> List[List[Tuple[ReadId, bytes, bytes]]]:
    return list(parser_gen)

def test_parse_fasta_correct_chunks(fasta_file_simple: pathlib.Path):
    """Test parsing FASTA with chunk_size smaller than total reads."""
    chunk_size = 2
    parser = SequenceFileProcessor.parse_sequence_files(fasta_file_simple, chunk_size=chunk_size)
    chunks = run_parser_and_collect_chunks(parser)

    assert len(chunks) == 2 # Expecting ceil(3/2) = 2 chunks
    assert len(chunks[0]) == chunk_size # First chunk should be full
    assert len(chunks[1]) == 1 # Last chunk has remaining

    assert chunks[0][0][0] == "read1"
    assert chunks[0][0][1] == b"ACGTACGT"
    assert chunks[0][1][0] == "read2"
    assert chunks[0][1][1] == b"CGTACGTA"
    assert chunks[1][0][0] == "read3"
    assert chunks[1][0][1] == b"GTACGTAC"

def test_parse_fastq_correct_chunks(fastq_file_simple: pathlib.Path):
    """Test parsing FASTQ with chunk_size smaller than total reads."""
    chunk_size = 2
    parser = SequenceFileProcessor.parse_sequence_files(fastq_file_simple, chunk_size=chunk_size)
    chunks = run_parser_and_collect_chunks(parser)

    assert len(chunks) == 2
    assert len(chunks[0]) == chunk_size
    assert len(chunks[1]) == 1

    assert chunks[0][0][0] == "read1"
    assert chunks[0][0][1] == b"ACGTACGT"
    assert chunks[0][1][0] == "read2"
    assert chunks[0][1][1] == b"CGTACGTA"
    assert chunks[1][0][0] == "read3"
    assert chunks[1][0][1] == b"GTACGTAC"

def test_parse_chunk_size_larger_than_reads(fasta_file_simple: pathlib.Path):
    """Test parsing when chunk_size is larger than the total number of reads."""
    chunk_size = 10
    parser = SequenceFileProcessor.parse_sequence_files(fasta_file_simple, chunk_size=chunk_size)
    chunks = run_parser_and_collect_chunks(parser)

    assert len(chunks) == 1
    assert len(chunks[0]) == 3 # All reads in one chunk

def test_parse_chunk_size_equals_reads(fasta_file_simple: pathlib.Path):
    """Test parsing when chunk_size equals the total number of reads."""
    chunk_size = 3
    parser = SequenceFileProcessor.parse_sequence_files(fasta_file_simple, chunk_size=chunk_size)
    chunks = run_parser_and_collect_chunks(parser)

    assert len(chunks) == 1
    assert len(chunks[0]) == 3

def test_parse_empty_file(tmp_path: pathlib.Path):
    """Test parsing an empty FASTA file."""
    empty_fasta = tmp_path / "empty.fasta"
    empty_fasta.write_text(">read_that_is_empty_so_no_seq_is_yielded\n") # SeqIO.parse will yield 0 records for this

    # Test with an actual empty file
    truly_empty_fasta = tmp_path / "truly_empty.fasta"
    truly_empty_fasta.write_text("")

    with pytest.raises(ValueError, match="Unknown file format for file"): # Our detection requires a line
         list(SequenceFileProcessor.parse_sequence_files(truly_empty_fasta))

    # Test with a file that has a header but no sequences
    parser_empty_seq = SequenceFileProcessor.parse_sequence_files(empty_fasta)
    chunks_empty_seq = run_parser_and_collect_chunks(parser_empty_seq)
    assert len(chunks_empty_seq) == 0 # No valid records to form a chunk


@patch("strainr.classify.SeqIO.parse")
@patch("strainr.classify.FastqGeneralIterator")
def test_parse_paired_end_reads_chunked(mock_fastq_iter, mock_fasta_iter, tmp_path: pathlib.Path):
    """Test paired-end FASTQ parsing with chunking."""
    fwd_path = tmp_path / "fwd.fastq"
    rev_path = tmp_path / "rev.fastq"
    fwd_path.write_text("@read1\nACGT\n+\nFFFF\n@read2\nTGCA\n+\nHHHH\n@read3\nAAAA\n+\nIIII")
    rev_path.write_text("@read1\nTTTT\n+\nFFFF\n@read2\nGGGG\n+\nHHHH\n@read3\nCCCC\n+\nIIII")

    # Mock FastqGeneralIterator to return specific values
    # Each call to FastqGeneralIterator (for fwd then rev) should return an iterator
    mock_fwd_iter_instance = iter([
        ("read1", "ACGT", "FFFF"), ("read2", "TGCA", "HHHH"), ("read3", "AAAA", "IIII")
    ])
    mock_rev_iter_instance = iter([
        ("read1", "TTTT", "FFFF"), ("read2", "GGGG", "HHHH"), ("read3", "CCCC", "IIII")
    ])
    # Configure the mock to return these iterators in sequence for fwd and rev
    mock_fastq_iter.side_effect = [mock_fwd_iter_instance, mock_rev_iter_instance]

    chunk_size = 2
    # Need to mock detect_file_format as well, since it's called by parse_sequence_files
    with patch.object(SequenceFileProcessor, 'detect_file_format', return_value='fastq'):
        parser = SequenceFileProcessor.parse_sequence_files(fwd_path, rev_path, chunk_size=chunk_size)
        chunks = run_parser_and_collect_chunks(parser)

    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1

    # Check content of first chunk
    assert chunks[0][0] == ("read1", b"ACGT", b"TTTT")
    assert chunks[0][1] == ("read2", b"TGCA", b"GGGG")
    # Check content of second chunk
    assert chunks[1][0] == ("read3", b"AAAA", b"CCCC")

    assert mock_fastq_iter.call_count == 2 # Called for fwd and rev


def test_parse_single_end_no_rev_file(fasta_file_simple: pathlib.Path):
    """Test single-end parsing when no reverse file is provided."""
    chunk_size = 2
    parser = SequenceFileProcessor.parse_sequence_files(fasta_file_simple, rev_reads_path=None, chunk_size=chunk_size)
    chunks = run_parser_and_collect_chunks(parser)

    assert len(chunks) == 2
    assert len(chunks[0]) == chunk_size
    assert len(chunks[1]) == 1

    assert chunks[0][0] == ("read1", b"ACGTACGT", b"") # Empty bytes for rev
    assert chunks[0][1] == ("read2", b"CGTACGTA", b"")
    assert chunks[1][0] == ("read3", b"GTACGTAC", b"")

# More tests could include:
# - Different chunk sizes.
# - Files with more reads.
# - Paired-end files with mismatched number of reads (though _process_read_pair handles this with a warning).
# - Gzipped files (transparently handled by open_file_handle, so core parsing logic should be the same).

# --- Tests for KmerClassificationWorkflow (Chunked Execution & Temp File Handling) ---
from strainr.classify import KmerClassificationWorkflow, CliArgs, ReadHitResults, DEFAULT_ABUNDANCE_THRESHOLD, DEFAULT_NUM_PROCESSES
from strainr.database import StrainKmerDatabase # Needed for database interaction
from strainr import ClassificationAnalyzer # For mocking

# Minimal valid CliArgs for workflow initialization
@pytest.fixture
def minimal_cli_args(tmp_path: pathlib.Path, fasta_file_simple: pathlib.Path) -> CliArgs:
    # Create a dummy database file for db_path validation
    dummy_db_path = tmp_path / "dummy_db.parquet"
    # Create a minimal Parquet file, or just touch if StrainKmerDatabase is fully mocked later
    # For now, just touching might not be enough if StrainKmerDatabase tries to read it.
    # Let's assume StrainKmerDatabase will be mocked in tests that don't focus on its loading.
    dummy_db_path.touch()

    return CliArgs(
        input_forward=[fasta_file_simple], # Use an existing dummy file
        db_path=dummy_db_path,
        output_dir=tmp_path / "output",
        # Other args will use defaults from Pydantic model
    )

@pytest.fixture
def mock_database_for_classify(monkeypatch) -> MagicMock:
    mock_db = MagicMock(spec=StrainKmerDatabase)
    mock_db.kmer_length = 5 # Example k-mer length
    mock_db.db_skip_n_kmers = False
    mock_db.strain_names = ["strainA", "strainB"]
    mock_db.num_strains = 2
    # _count_kmers_for_read calls get_strain_counts_for_kmer
    mock_db.get_strain_counts_for_kmer.return_value = np.array([1,0], dtype=np.uint8) # Default mock
    return mock_db


@patch("strainr.classify.mp.Pool")
@patch("strainr.classify.SequenceFileProcessor.parse_sequence_files")
@patch("strainr.classify.pickle.dump")
@patch("pathlib.Path.unlink", return_value=None) # Mock unlink to avoid errors if file not found
@patch("pathlib.Path.rmdir", return_value=None)   # Mock rmdir
def test_classify_reads_parallel_chunked_logic(
    mock_rmdir: MagicMock,
    mock_unlink: MagicMock,
    mock_pickle_dump: MagicMock,
    mock_parse_files: MagicMock,
    mock_mp_pool: MagicMock,
    minimal_cli_args: CliArgs, # Use the CliArgs fixture
    mock_database_for_classify: MagicMock, # Mocked StrainKmerDatabase
    tmp_path: pathlib.Path # For output and temp dirs
):
    workflow = KmerClassificationWorkflow(minimal_cli_args)
    workflow.database = mock_database_for_classify # Inject mocked database

    # --- Setup Mocks ---
    # Mock SequenceFileProcessor.parse_sequence_files to yield predefined chunks
    # Each read tuple: (ReadId, fwd_bytes, rev_bytes)
    read_chunk1 = [("r1", b"ACGTA", b""), ("r2", b"CGTAC", b"")]
    read_chunk2 = [("r3", b"GTACG", b"")]
    mock_parse_files.return_value = iter([read_chunk1, read_chunk2])

    # Mock mp.Pool().map() behavior
    # _count_kmers_for_read returns (ReadId, CountVector)
    # Let's say r1 is clear for strainA, r2 is ambiguous, r3 is no-hit
    mock_pool_map_results_chunk1 = [
        ("r1", np.array([1, 0], dtype=np.uint8)), # Clear hit for strainA
        ("r2", np.array([1, 1], dtype=np.uint8))  # Ambiguous
    ]
    mock_pool_map_results_chunk2 = [
        ("r3", np.array([0, 0], dtype=np.uint8))  # No hit
    ]

    mock_pool_instance = MagicMock()
    mock_pool_instance.map.side_effect = [mock_pool_map_results_chunk1, mock_pool_map_results_chunk2]
    mock_mp_pool.return_value.__enter__.return_value = mock_pool_instance

    # ClassificationAnalyzer is created inside _classify_reads_parallel
    # We need to mock its methods. Patching the class itself or its relevant methods.
    # For simplicity, let's assume the real ClassificationAnalyzer is used but its inputs/outputs are checked.
    # Or, mock specific methods of the analyzer instance created within the method.
    # The current implementation creates an analyzer instance inside _classify_reads_parallel.
    # This is harder to mock directly without refactoring _classify_reads_parallel to accept an analyzer instance.
    # Alternative: Patch 'strainr.classify.ClassificationAnalyzer' to control its instances.

    with patch('strainr.classify.ClassificationAnalyzer') as MockAnalyzerClass:
        mock_analyzer_instance = MockAnalyzerClass.return_value

        # Configure mock analyzer behavior for separate_hit_categories
        # Chunk 1 results:
        def separate_hits_side_effect_chunk1(results):
            if results == mock_pool_map_results_chunk1: # Based on input
                return {"r1": np.array([1,0], dtype=np.uint8)}, \
                       {"r2": np.array([1,1], dtype=np.uint8)}, \
                       []
            elif results == mock_pool_map_results_chunk2:
                 return {}, {}, ["r3"]
            return {}, {}, []
        mock_analyzer_instance.separate_hit_categories.side_effect = separate_hits_side_effect_chunk1

        # Configure mock analyzer behavior for resolve_clear_hits_to_indices
        def resolve_clear_side_effect(clear_hits_dict):
            if "r1" in clear_hits_dict:
                return {"r1": 0} # strainA index
            return {}
        mock_analyzer_instance.resolve_clear_hits_to_indices.side_effect = resolve_clear_side_effect

        # --- Execute ---
        # The output_dir for workflow is tmp_path / "output"
        # The temp_ambiguous_dir will be tmp_path / "output" / "temp_ambiguous_chunks"
        # Ensure the main output_dir exists as the method expects to create a subdir in it
        minimal_cli_args.output_dir.mkdir(parents=True, exist_ok=True)

        fwd_path = minimal_cli_args.input_forward[0] # From fixture

        clear_assign, no_hits, ambig_files, temp_dir = workflow._classify_reads_parallel(fwd_path, None)

    # --- Assertions ---
    mock_parse_files.assert_called_once_with(fwd_path, None, chunk_size=minimal_cli_args.chunk_size)
    assert mock_pool_instance.map.call_count == 2 # Called for each chunk

    # Check accumulated results
    assert clear_assign == {"r1": 0} # strainA
    assert no_hits == ["r3"]

    # Check ambiguous file handling
    assert len(ambig_files) == 1 # Only chunk 1 had ambiguous read r2
    assert mock_pickle_dump.call_count == 1
    # Check what was pickled: first call to pickle.dump, first arg is the data
    # args[0] is the data, args[1] is the file handle
    pickled_data = mock_pickle_dump.call_args_list[0][0][0]
    assert "r2" in pickled_data
    np.testing.assert_array_equal(pickled_data["r2"], np.array([1,1], dtype=np.uint8))

    # Check temp directory structure
    expected_temp_dir = minimal_cli_args.output_dir / "temp_ambiguous_chunks"
    assert temp_dir == expected_temp_dir
    assert (expected_temp_dir / "ambiguous_chunk_0.pkl").exists() # Pickle dump was called for this file

    # Note: Cleanup of temp files is handled in run_workflow, not _classify_reads_parallel
    mock_unlink.assert_not_called() # Not called by _classify_reads_parallel
    mock_rmdir.assert_not_called()  # Not called by _classify_reads_parallel


@patch("strainr.classify.KmerClassificationWorkflow._initialize_database") # Mock DB init
@patch("strainr.classify.KmerClassificationWorkflow._classify_reads_parallel")
@patch("strainr.classify.ClassificationAnalyzer") # Mock the class to control its instances
@patch("strainr.classify.AbundanceCalculator")
@patch("strainr.classify.pickle.load")
@patch("pathlib.Path.unlink", return_value=None)
@patch("pathlib.Path.rmdir", return_value=None)
def test_run_workflow_two_pass_logic(
    mock_rmdir: MagicMock,
    mock_unlink: MagicMock,
    mock_pickle_load: MagicMock,
    MockAbundanceCalculator: MagicMock, # Note: Class mock
    MockAnalyzerClass: MagicMock,       # Note: Class mock
    mock_classify_reads_parallel: MagicMock,
    mock_init_db: MagicMock,
    minimal_cli_args: CliArgs, # Uses the existing fixture
    tmp_path: pathlib.Path, # For dummy files
    mock_database_for_classify: MagicMock # To be returned by _initialize_database
):
    # --- Setup Workflow and Mocks ---
    # Assign the mocked database to be "loaded" by _initialize_database
    # To do this, we need the KmerClassificationWorkflow instance to have its self.database set.
    # _initialize_database is mocked, so it won't set self.database.
    # We can set it directly after workflow instantiation for the test.

    workflow = KmerClassificationWorkflow(minimal_cli_args)
    workflow.database = mock_database_for_classify # Manually set the mocked database

    # Mock ClassificationAnalyzer instance that will be created in run_workflow
    mock_analyzer_instance = MockAnalyzerClass.return_value
    # Mock AbundanceCalculator instance
    mock_abundance_calc_instance = MockAbundanceCalculator.return_value

    # Define what _classify_reads_parallel (Pass 1) returns
    temp_ambiguous_dir = minimal_cli_args.output_dir / "temp_ambiguous_chunks" # Consistent with _classify_reads_parallel
    # temp_ambiguous_dir.mkdir(parents=True, exist_ok=True) # _classify_reads_parallel would do this. run_workflow assumes it exists.

    # Create dummy pickled ambiguous files that _classify_reads_parallel would have created
    ambig_chunk_file1 = temp_ambiguous_dir / "ambig_chunk_0.pkl"
    # ambig_chunk_file1.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists for pickle.load mock later
    # For the test, these files don't need real content if pickle.load is properly mocked.
    # However, the run_workflow logic *will* try to load from these paths.

    mock_classify_reads_parallel.return_value = (
        {"r1_clear": 0, "r4_clear": 1},  # all_clear_assignments (ReadId: StrainIndex)
        ["r3_nohit"],                     # all_no_hit_ids
        [ambig_chunk_file1],              # ambiguous_hit_files_paths
        temp_ambiguous_dir                # temp_ambiguous_dir path
    )

    # Define data that pickle.load will return for ambiguous files
    mock_pickle_load.return_value = {
        "r2_ambig": np.array([1, 1], dtype=np.uint8), # Ambiguous data for chunk 1
        "r5_ambig": np.array([2, 2], dtype=np.uint8)
    }

    # Define behavior for ClassificationAnalyzer methods called in run_workflow
    mock_analyzer_instance.calculate_strain_prior_from_assignments.return_value = Counter({0: 1, 1: 1}) # strain 0: 1 read, strain 1: 1 read
    mock_analyzer_instance.convert_prior_counts_to_probability_vector.return_value = np.array([0.5, 0.5])

    # resolve_ambiguous_hits_parallel will be called with the loaded ambiguous_hits_chunk_dict
    # For this test, assume it resolves r2_ambig to strain 0, r5_ambig to strain 1
    mock_analyzer_instance.resolve_ambiguous_hits_parallel.return_value = {"r2_ambig": 0, "r5_ambig": 1}

    # Mock combine_assignments
    expected_final_assignments = {
        "r1_clear": 0, "r4_clear": 1, # Clear
        "r3_nohit": "NA",             # No-hit
        "r2_ambig": 0, "r5_ambig": 1  # Resolved
    }
    mock_analyzer_instance.combine_assignments.return_value = expected_final_assignments

    # Mock AbundanceCalculator methods (not the primary focus, but part of the flow)
    mock_abundance_calc_instance.convert_assignments_to_strain_names.return_value = {k: str(v) for k,v in expected_final_assignments.items()}
    mock_abundance_calc_instance.calculate_raw_abundances.return_value = Counter({"0":2, "1":2, "NA":1}) # Example counts

    # --- Execute run_workflow ---
    # Ensure the main output directory exists for the workflow to create subdirectories if needed
    minimal_cli_args.output_dir.mkdir(parents=True, exist_ok=True)
    # Also ensure the temp_ambiguous_dir (which _classify_reads_parallel would make) exists for run_workflow to interact with
    temp_ambiguous_dir.mkdir(parents=True, exist_ok=True)


    workflow.run_workflow()

    # --- Assertions ---
    mock_init_db.assert_called_once()
    mock_classify_reads_parallel.assert_called_once()

    # Check prior calculation
    mock_analyzer_instance.calculate_strain_prior_from_assignments.assert_called_once_with(
        {"r1_clear": 0, "r4_clear": 1}
    )
    mock_analyzer_instance.convert_prior_counts_to_probability_vector.assert_called_once()

    # Check ambiguous resolution
    assert mock_pickle_load.call_count == 1 # Called for ambig_chunk_file1
    mock_analyzer_instance.resolve_ambiguous_hits_parallel.assert_called_once_with(
        { # This is what mock_pickle_load returned
            "r2_ambig": np.array([1, 1], dtype=np.uint8),
            "r5_ambig": np.array([2, 2], dtype=np.uint8)
        },
        np.array([0.5, 0.5]) # The prior_probs
    )

    # Check combine_assignments call
    mock_analyzer_instance.combine_assignments.assert_called_once_with(
        {"r1_clear": 0, "r4_clear": 1}, # all_clear_assignments
        {"r2_ambig": 0, "r5_ambig": 1}, # resolved_ambiguous_assignments_all_chunks
        ["r3_nohit"],                   # all_no_hit_ids
        unassigned_marker="NA"
    )

    # Check cleanup
    # mock_unlink should be called for ambig_chunk_file1
    # It might be called with a pathlib.Path object.
    # Check that unlink was called on a path object that ends with "ambig_chunk_0.pkl"
    found_unlink_call = False
    for call_args in mock_unlink.call_args_list:
        path_arg = call_args[0][0] # The first positional argument to unlink
        if isinstance(path_arg, pathlib.Path) and path_arg.name == "ambig_chunk_0.pkl":
            found_unlink_call = True
            break
    assert found_unlink_call, "unlink was not called for the ambiguous chunk file"

    # Check rmdir for the temp_ambiguous_dir
    found_rmdir_call = False
    for call_args in mock_rmdir.call_args_list:
        path_arg = call_args[0][0]
        if path_arg == temp_ambiguous_dir:
            found_rmdir_call = True
            break
    assert found_rmdir_call, "rmdir was not called for the temporary ambiguous directory"

    # Check that abundance calculation and output methods were called (basic check)
    mock_abundance_calc_instance.convert_assignments_to_strain_names.assert_called_once()
    mock_abundance_calc_instance.calculate_raw_abundances.assert_called_once()
    # Further checks on pandas DataFrame creation and to_csv can be added if needed
