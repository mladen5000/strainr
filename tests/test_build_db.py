import pytest
import pandas as pd
import strainr # Import the strainr package
from unittest.mock import patch, mock_open, MagicMock
from io import StringIO
import pathlib # For creating dummy files/dirs for path validation tests

# Import the Pydantic model to be tested
from strainr.build_db import BuildDBArgs
from pydantic import ValidationError


# Store the real read_csv function before mocking
REAL_READ_CSV = pd.read_csv

# --- Pytest-style conversion for the rest of the tests ---

@patch("pandas.read_csv")
def test_filter_genomes_unique_taxid_basic(mock_read_csv, db_builder):
    builder, test_dir, _ = db_builder # Unpack from fixture
    metadata_content = """assembly_accession,taxid,species_taxid,local_filename
GCA_001,101,100,GCA_001.fna.gz
GCA_002,200,200,GCA_002.fna.gz
GCA_003,301,300,GCA_003.fna.gz
GCA_004,400,,GCA_004.fna.gz
"""
    mock_df = REAL_READ_CSV(StringIO(metadata_content))
    mock_read_csv.return_value = mock_df
    genome_dir = test_dir / "genomes_filter_test" # Use test_dir from fixture
    genome_dir.mkdir(exist_ok=True)
    (genome_dir / "GCA_001.fna.gz").touch()
    (genome_dir / "GCA_002.fna.gz").touch()
    (genome_dir / "GCA_003.fna.gz").touch()
    (genome_dir / "GCA_004.fna.gz").touch()
    mock_metadata_path = test_dir / "metadata_filter.tsv"

    def mock_path_exists_filter(path_obj):
        if path_obj == mock_metadata_path:
            return True
        if path_obj.parent == genome_dir and path_obj.name.endswith(".fna.gz"):
            return True
        # For the filtered metadata CSV path, ensure it's correctly formed based on builder's logic
        # Assuming builder's logic saves it correctly relative to metadata_table_path.parent
        if path_obj == mock_metadata_path.parent / f"filtered_{mock_metadata_path.name}":
            return True # Allow creation of filtered metadata file
        return False


    with (
        patch("pathlib.Path.exists", side_effect=mock_path_exists_filter),
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
    ):
        filtered_files = builder._filter_genomes_by_unique_taxid(
            genome_dir, mock_metadata_path
        )
    # Based on the logic: GCA_002 is NOT unique strain (taxid == species_taxid)
    # GCA_001, GCA_003 are unique. GCA_004 has no species_taxid, so it's kept (original logic implies this).
    assert len(filtered_files) == 3
    assert genome_dir / "GCA_001.fna.gz" in filtered_files
    assert genome_dir / "GCA_003.fna.gz" in filtered_files
    assert genome_dir / "GCA_004.fna.gz" in filtered_files
    mock_to_csv.assert_called_once()


@patch("pandas.read_csv")
def test_filter_genomes_metadata_missing_local_filename(mock_read_csv, db_builder):
    builder, test_dir, _ = db_builder
    metadata_content = """assembly_accession,taxid,species_taxid
GCA_001,101,100
"""
    mock_df = REAL_READ_CSV(StringIO(metadata_content))
    mock_read_csv.return_value = mock_df
    genome_dir = test_dir / "genomes_missing_col"
    genome_dir.mkdir(exist_ok=True)
    mock_metadata_path = test_dir / "metadata_missing_col.tsv"
    with (
        patch("pathlib.Path.exists", return_value=True), # Mock existence for metadata and filtered metadata
        patch("pandas.DataFrame.to_csv"),
    ):
        filtered_files = builder._filter_genomes_by_unique_taxid(
            genome_dir, mock_metadata_path
        )
    assert len(filtered_files) == 0


@patch("pandas.read_csv")
def test_filter_genomes_local_file_does_not_exist(mock_read_csv, db_builder):
    builder, test_dir, _ = db_builder
    metadata_content = """assembly_accession,taxid,species_taxid,local_filename
GCA_001,101,100,non_existent_GCA_001.fna.gz
"""
    mock_df = REAL_READ_CSV(StringIO(metadata_content))
    mock_read_csv.return_value = mock_df
    genome_dir = test_dir / "genomes_non_existent_local"
    genome_dir.mkdir(exist_ok=True)
    mock_metadata_path = test_dir / "metadata_non_existent_local.tsv"

    def mock_path_exists_specific(path_obj):
        if path_obj == mock_metadata_path: # metadata file itself
            return True
        if path_obj == (genome_dir / "non_existent_GCA_001.fna.gz"): # the missing genome file
            return False
        # For the filtered metadata CSV path
        if path_obj == mock_metadata_path.parent / f"filtered_{mock_metadata_path.name}":
            return True
        return True # For other paths, assume they exist if needed by underlying code

    with (
        patch("pathlib.Path.exists", side_effect=mock_path_exists_specific),
        patch("pandas.DataFrame.to_csv"),
    ):
        filtered_files = builder._filter_genomes_by_unique_taxid(
            genome_dir, mock_metadata_path
        )
    assert len(filtered_files) == 0 # Since the file doesn't exist, it shouldn't be in the list


@patch("strainr.build_db.open_file_transparently", new_callable=mock_open)
def test_process_single_fasta_for_kmers_writes_correct_kmers(mock_actual_open, db_builder):
    builder, test_dir, args_instance = db_builder
    # Sequence: ATGCGTAGCATGC (13 bases)
    # Kmerlen from fixture: args_instance.kmerlen (which is 5)
    # Expected kmers (len 5):
    # ATGCG, TGCGT, GCGTA, CGTAG, GTAGC, TAGCA, AGCAT, GCATG, CATGC
    # Total: 13 - 5 + 1 = 9 kmers
    mock_actual_open.return_value = StringIO(">seq1\nATGCGTAGCATGC\n")
    dummy_fasta_path = test_dir / "dummy.fna.gz" # Not actually read due to mock_open
    strain_name = "test_strain"
    strain_idx = 0
    # temp_kmer_file_path should be inside test_dir (which is tmp_path)
    temp_kmer_file_path = test_dir / "kmer_output.txt"

    genome_file_info = (dummy_fasta_path, strain_name, strain_idx, temp_kmer_file_path)

    # Call the method. builder.args.kmerlen is already set by fixture.
    # builder.args.skip_n_kmers is False by default in fixture.
    # num_total_strains is optional, pass if it affects return type or logic for this test.
    # The test expects (strain_name, written_kmer_count, output_path)
    _, written_kmer_count, out_path = builder._process_single_fasta_for_kmers(
        genome_file_info,
        kmer_length=args_instance.kmerlen, # Use kmerlen from fixture's args
        skip_n_kmers=args_instance.skip_n_kmers,
        # num_total_strains=None # Let it use default to get the 3-tuple return
    )
    assert written_kmer_count == 9
    assert out_path == temp_kmer_file_path

    written_kmers = set()
    # mock_open was used for open_file_transparently, not for reading the output file.
    # So, we need to ensure the actual file was written to by the logic inside _process_single_fasta_for_kmers
    # The current mock setup for open_file_transparently means the file reading part is mocked.
    # The kmer generation is from the string directly.
    # The writing part uses a standard open() call, so we need to check that.
    # This test is for when output_path is provided.

    # We need to check the *content* of temp_kmer_file_path
    # The mock_open for open_file_transparently doesn't affect the open() used to write to output_path
    # So, the file should have been written.

    with open(temp_kmer_file_path, "r", encoding="utf-8") as f:
        for line in f:
            written_kmers.add(line.strip())

    expected_kmers_str = {
        "ATGCG", "TGCGT", "GCGTA", "CGTAG", "GTAGC",
        "TAGCA", "AGCAT", "GCATG", "CATGC"
    }
    assert written_kmers == expected_kmers_str
    if temp_kmer_file_path.exists():
        temp_kmer_file_path.unlink()


@patch("strainr.build_db.open_file_transparently", new_callable=mock_open)
def test_process_single_fasta_kmers_seq_shorter_than_kmerlen(mock_actual_open, db_builder):
    builder, test_dir, args_instance = db_builder
    # Kmerlen from fixture: args_instance.kmerlen (which is 5)
    # Sequence: ATG (3 bases) < 5
    mock_actual_open.return_value = StringIO(">seq1\nATG\n")
    dummy_fasta_path = test_dir / "short.fna.gz"
    temp_kmer_file_path = test_dir / "short_kmer_output.txt"
    genome_file_info = (dummy_fasta_path, "short_strain", 0, temp_kmer_file_path)

    # Call with num_total_strains to get the 4-tuple return if that's what the original test structure implied
    # Or None to get 3-tuple: (strain_name, written_kmer_count, output_path)
    _, _, written_kmer_count, _ = builder._process_single_fasta_for_kmers(
        genome_file_info,
        kmer_length=args_instance.kmerlen,
        skip_n_kmers=args_instance.skip_n_kmers,
        num_total_strains=1 # Ensure 4-tuple return for consistency if previous test structure relied on it
    )
    assert written_kmer_count == 0
    assert temp_kmer_file_path.exists() # File should be created even if empty
    with open(temp_kmer_file_path, "r") as f:
        assert f.read() == ""
    if temp_kmer_file_path.exists():
        temp_kmer_file_path.unlink()


@patch("strainr.build_db.open_file_transparently", new_callable=mock_open)
def test_process_single_fasta_kmers_seq_equals_kmerlen(mock_actual_open, db_builder):
    builder, test_dir, args_instance = db_builder
    # Kmerlen from fixture: args_instance.kmerlen (which is 5)
    # Sequence: ATGCG (5 bases) == 5. Should yield 1 kmer.
    mock_actual_open.return_value = StringIO(">seq1\nATGCG\n")
    dummy_fasta_path = test_dir / "equal.fna.gz"
    temp_kmer_file_path = test_dir / "equal_kmer_output.txt"
    genome_file_info = (dummy_fasta_path, "equal_strain", 0, temp_kmer_file_path)

    _, _, written_kmer_count, _ = builder._process_single_fasta_for_kmers(
        genome_file_info,
        kmer_length=args_instance.kmerlen,
        skip_n_kmers=args_instance.skip_n_kmers,
        num_total_strains=1
    )
    assert written_kmer_count == 1
    with open(temp_kmer_file_path, "r") as f:
        assert f.read().strip() == "ATGCG"
    if temp_kmer_file_path.exists():
        temp_kmer_file_path.unlink()

# --- Tests for BuildDBArgs Pydantic Model ---

def test_builddbargs_valid_taxid():
    """Test successful validation with taxid and default values."""
    args = BuildDBArgs(taxid="9606")
    assert args.taxid == "9606"
    assert args.kmerlen == 31
    assert args.source == "refseq"

def test_builddbargs_valid_custom_dir(tmp_path: pathlib.Path):
    """Test successful validation with a custom directory."""
    custom_dir = tmp_path / "custom_genomes"
    custom_dir.mkdir()
    args = BuildDBArgs(custom=custom_dir)
    assert args.custom == custom_dir

def test_builddbargs_no_genome_source():
    """Test ValidationError if no genome source is provided."""
    with pytest.raises(ValidationError) as excinfo:
        BuildDBArgs() # No source arguments
    assert "Exactly one of --taxid, --assembly_accessions, --genus, or --custom must be specified." in str(excinfo.value)

def test_builddbargs_multiple_genome_sources(tmp_path: pathlib.Path):
    """Test ValidationError if multiple genome sources are provided."""
    custom_dir = tmp_path / "custom_genomes"
    custom_dir.mkdir()
    with pytest.raises(ValidationError) as excinfo:
        BuildDBArgs(taxid="9606", custom=custom_dir)
    assert "Exactly one of --taxid, --assembly_accessions, --genus, or --custom must be specified." in str(excinfo.value)

def test_builddbargs_assembly_accessions_file_not_found(tmp_path: pathlib.Path):
    """Test ValidationError if assembly_accessions file does not exist."""
    non_existent_file = tmp_path / "accessions.txt"
    with pytest.raises(ValidationError) as excinfo: # Pydantic v2 uses its own error, not FileNotFoundError directly from model
        BuildDBArgs(assembly_accessions=non_existent_file)
    # Check for Pydantic's specific error messages or field errors
    assert "Path" in str(excinfo.value) # Pydantic error for FilePath
    assert "Path does not point to a file" in str(excinfo.value)


def test_builddbargs_custom_dir_not_found(tmp_path: pathlib.Path):
    """Test ValidationError if custom directory does not exist."""
    non_existent_dir = tmp_path / "non_existent_custom"
    with pytest.raises(ValidationError) as excinfo:
        BuildDBArgs(custom=non_existent_dir)
    assert "Path" in str(excinfo.value)
    assert "Path does not point to a directory" in str(excinfo.value)


def test_builddbargs_valid_assembly_accessions(tmp_path: pathlib.Path):
    """Test successful validation with an existing assembly_accessions file."""
    accessions_file = tmp_path / "accessions.txt"
    accessions_file.write_text("GCA_001\nGCA_002")
    args = BuildDBArgs(assembly_accessions=accessions_file)
    assert args.assembly_accessions == accessions_file

@pytest.mark.parametrize("invalid_kmerlen", [0, -1, -31])
def test_builddbargs_invalid_kmerlen(invalid_kmerlen: int):
    """Test ValidationError for kmerlen <= 0."""
    with pytest.raises(ValidationError) as excinfo:
        BuildDBArgs(taxid="9606", kmerlen=invalid_kmerlen)
    assert "Input should be greater than 0" in str(excinfo.value) # Pydantic v2 specific message for gt=0

@pytest.mark.parametrize("invalid_procs", [0, -1])
def test_builddbargs_invalid_procs(invalid_procs: int):
    """Test ValidationError for procs <= 0."""
    with pytest.raises(ValidationError) as excinfo:
        BuildDBArgs(taxid="9606", procs=invalid_procs)
    assert "Input should be greater than 0" in str(excinfo.value)

def test_builddbargs_invalid_assembly_level():
    """Test ValidationError for invalid assembly_levels choice."""
    with pytest.raises(ValidationError) as excinfo:
        BuildDBArgs(taxid="9606", assembly_levels="bogus_level")
    assert "Input should be 'complete', 'chromosome', 'scaffold' or 'contig'" in str(excinfo.value)


def test_builddbargs_invalid_source():
    """Test ValidationError for invalid source choice."""
    with pytest.raises(ValidationError) as excinfo:
        BuildDBArgs(taxid="9606", source="bogus_source")
    assert "Input should be 'refseq' or 'genbank'" in str(excinfo.value)

def test_builddbargs_defaults_applied():
    """Test that default values are correctly applied."""
    args = BuildDBArgs(taxid="123")
    assert args.kmerlen == 31
    assert args.assembly_levels == "complete"
    assert args.source == "refseq"
    assert args.procs == 4
    assert args.out == "strainr_kmer_database"
    assert args.unique_taxid is False
    assert args.skip_n_kmers is False


# --- Tests for Parquet Metadata Writing ---

@patch("strainr.build_db.subprocess.run")
@patch("strainr.build_db.mp.Pool")
@patch("strainr.build_db.DatabaseBuilder._process_and_write_kmers_worker")
@patch("pathlib.Path.glob")
@patch("builtins.open", new_callable=mock_open)
def test_build_kmer_database_parallel_writes_metadata(
    mock_open_file: MagicMock,
    mock_glob: MagicMock,
    mock_process_worker: MagicMock,
    mock_pool: MagicMock,
    mock_subprocess_run: MagicMock,
    tmp_path: pathlib.Path
):
    """
    Test that _build_kmer_database_parallel writes kmerlen and skip_n_kmers to Parquet metadata.
    """
    # Setup BuildDBArgs
    kmerlen_val = 21
    skip_n_val = True
    args = BuildDBArgs(
        custom=tmp_path, # Provide a valid path for custom source
        kmerlen=kmerlen_val,
        skip_n_kmers=skip_n_val,
        out="test_db_metadata"
    )

    # Create a dummy custom directory as BuildDBArgs validates it
    custom_genomes_dir = tmp_path # Use tmp_path directly as the custom source for simplicity
    # custom_genomes_dir.mkdir(parents=True, exist_ok=True) # BuildDBArgs checks existence, Typer would too. tmp_path exists.

    args_for_builder = BuildDBArgs(
        custom=custom_genomes_dir, # custom expects a DirectoryPath, tmp_path is one
        kmerlen=kmerlen_val,
        skip_n_kmers=skip_n_val,
        out="test_db_metadata_build" # This will be used for output_db_dir
    )

    db_builder_instance = strainr.build_db.DatabaseBuilder(args_for_builder)
    # Override base_path to ensure outputs go into tmp_path for this test
    db_builder_instance.base_path = tmp_path

    # Mock genome files and strain names
    genome_files = [custom_genomes_dir / "genome1.fna.gz"]
    strain_names = ["strain1"]

    # Mock path interactions for the temp directory
    # db_builder_instance.output_db_dir is like 'test_db_metadata_build_files'
    temp_kmer_parts_dir = db_builder_instance.output_db_dir / "tmp_kmer_parts"

    # Mock glob to return a list of dummy part files
    # Example: tmp_path / "test_db_metadata_build_files" / "tmp_kmer_parts" / "*.part.gz"
    mock_glob.return_value = [temp_kmer_parts_dir / "0.part.gz"]

    # Mock subprocess.run to simulate success for sort and zcat
    mock_subprocess_run.return_value = MagicMock(returncode=0, stderr="")

    # Mock the worker function to avoid actual k-mer processing
    mock_process_worker.return_value = temp_kmer_parts_dir / "0.part.gz" # Simulate worker creates a file

    # Mock open for the sorted file reading part
    # Simulate a very simple sorted k-mer file content: kmer_hex\tstrain_idx
    # kmer "AAAAA" (hex for kmerlen=5, adjust if kmerlen_val changes) -> 4141414141
    # For kmerlen_val = 21, this is more complex if we need valid hex.
    # Let's use a simple kmer for simulation if actual content doesn't matter beyond format.
    # "A"*21 -> hex: 41 repeated 21 times
    simulated_kmer_hex = "41" * kmerlen_val
    mock_open_file.return_value = StringIO(f"{simulated_kmer_hex}\t0\n")

    # Mock the Pool context manager
    mock_pool_instance = MagicMock()
    mock_pool.return_value.__enter__.return_value = mock_pool_instance
    mock_pool_instance.imap_unordered.return_value = [] # Simulate no actual items from workers for simplicity in map phase

    # Ensure the temp directory and output directory are handled by tmp_path
    # The builder creates self.output_db_dir, e.g., .../test_db_metadata_build_files
    # The Parquet file will be at .../test_db_metadata_build.db.parquet

    # Call the method
    # The DatabaseBuilder's base_path is cwd. Parquet file will be tmp_path / builder.output_db_name + ".db.parquet"
    # output_parquet_path = tmp_path / (db_builder_instance.output_db_name + ".db.parquet")
    # The DatabaseBuilder uses pathlib.Path().cwd() as base_path by default.
    # We've overridden it to tmp_path above.

    output_parquet_path = db_builder_instance._build_kmer_database_parallel(genome_files, strain_names)

    assert output_parquet_path is not None
    # The path returned by _build_kmer_database_parallel is base_path / (output_db_name + ".db.parquet")
    # So it should be tmp_path / "test_db_metadata_build.db.parquet"
    assert output_parquet_path == tmp_path / (args_for_builder.out + ".db.parquet")
    assert output_parquet_path.exists()

    # Read schema and check metadata
    import pyarrow.parquet as pq
    schema = pq.read_schema(output_parquet_path)

    assert b"strainr_kmerlen" in schema.metadata
    assert schema.metadata[b"strainr_kmerlen"] == str(kmerlen_val).encode('utf-8')

    assert b"strainr_skip_n_kmers" in schema.metadata
    assert schema.metadata[b"strainr_skip_n_kmers"] == str(skip_n_val).encode('utf-8')


def test_rust_kmer_counter_usage_and_correctness(tmp_path: pathlib.Path):
    """
    Tests if the Rust k-mer counter is used when available and if its output
    is consistent with the Python k-mer extraction logic.
    """
    import strainr.build_db # For accessing module-level flags
    from strainr.build_db import DatabaseBuilder, BuildDBArgs
    from kmer_counter_rs import extract_kmers_rs as rust_extract_func

    # Args for DatabaseBuilder
    args = BuildDBArgs(
        custom=tmp_path, # Needs a valid path
        kmerlen=5,
        skip_n_kmers=False # Test with skip_n_kmers=False first
    )
    db_builder_instance = DatabaseBuilder(args=args)

    # 1. Check if Rust module is active
    assert strainr.build_db._RUST_KMER_COUNTER_AVAILABLE is True, "Rust k-mer counter should be available."
    assert strainr.build_db._extract_kmers_func == rust_extract_func, "DatabaseBuilder should be configured to use Rust k-mer extraction."

    # 2. Define sample sequence and k-mer length
    sample_sequence_str = "ATGCGTAGCATGCGT"
    sample_sequence_bytes = sample_sequence_str.encode('utf-8')
    kmer_len = 5

    # 3. Call _extract_kmers_from_bytes (should use Rust)
    # Ensure the builder's args are set for this specific test scenario if they affect kmer extraction
    db_builder_instance.args.kmerlen = kmer_len
    db_builder_instance.args.skip_n_kmers = False # Explicitly set for this part of the test

    rust_kmers_list = db_builder_instance._extract_kmers_from_bytes(sample_sequence_bytes, kmer_len)
    rust_kmers_set = set(rust_kmers_list) # Convert list of k-mers to set for comparison

    # 4. Get expected k-mers using Python logic (DatabaseBuilder._py_extract_canonical_kmers_static)
    # The _py_extract_canonical_kmers_static method returns a list of bytes.
    py_kmers_list = DatabaseBuilder._py_extract_canonical_kmers_static(
        sample_sequence_bytes, kmer_len, False # skip_n_kmers = False
    )
    py_kmers_set = set(py_kmers_list)

    # 5. Assert sets are identical
    assert rust_kmers_set == py_kmers_set, \
        f"Rust k-mer set {rust_kmers_set} differs from Python k-mer set {py_kmers_set} (skip_n_kmers=False)"

    # 6. Test with skip_n_kmers=True (and a sequence containing 'N')
    db_builder_instance.args.skip_n_kmers = True
    sample_sequence_n_str = "ATGCGNTNNCATGCGT"
    sample_sequence_n_bytes = sample_sequence_n_str.encode('utf-8')

    rust_kmers_n_list = db_builder_instance._extract_kmers_from_bytes(sample_sequence_n_bytes, kmer_len)
    rust_kmers_n_set = set(rust_kmers_n_list)

    py_kmers_n_list = DatabaseBuilder._py_extract_canonical_kmers_static(
        sample_sequence_n_bytes, kmer_len, True # skip_n_kmers = True
    )
    py_kmers_n_set = set(py_kmers_n_list)

    assert rust_kmers_n_set == py_kmers_n_set, \
        f"Rust k-mer set {rust_kmers_n_set} differs from Python k-mer set {py_kmers_n_set} (skip_n_kmers=True)"

    # Test case: what if Rust is NOT available (e.g. by temporarily overriding the flag)
    # This part of the test ensures the fallback mechanism works as expected.
    try:
        original_rust_available = strainr.build_db._RUST_KMER_COUNTER_AVAILABLE
        original_extract_func = strainr.build_db._extract_kmers_func

        strainr.build_db._RUST_KMER_COUNTER_AVAILABLE = False
        strainr.build_db._extract_kmers_func = None # Simulate Rust not being imported

        # Re-create builder instance or ensure its state reflects the change
        # For this test, simply changing the global flags should be enough as _extract_kmers_from_bytes checks them.

        # With Rust disabled, it should use the Python path.
        # We expect the result to be the same as py_kmers_set calculated earlier.
        fallback_kmers_list = db_builder_instance._extract_kmers_from_bytes(sample_sequence_bytes, kmer_len)
        fallback_kmers_set = set(fallback_kmers_list)

        assert fallback_kmers_set == py_kmers_set, \
            "Fallback to Python k-mer extraction did not produce the expected Python k-mers."
        assert strainr.build_db._extract_kmers_func is None, \
            "After disabling Rust, _extract_kmers_func should be None (Python path)."

    finally:
        # Restore original state
        strainr.build_db._RUST_KMER_COUNTER_AVAILABLE = original_rust_available
        strainr.build_db._extract_kmers_func = original_extract_func


# --- Integration test for k-mer extraction pipeline consistency ---
def test_pipeline_kmer_extraction_consistency(tmp_path: pathlib.Path):
    """
    Tests that the k-mer extraction pipeline (_process_and_write_kmers_worker)
    produces consistent results between the Rust path and Python path,
    for both skip_n_kmers=True and skip_n_kmers=False.
    """
    import gzip
    import shutil
    from unittest.mock import patch
    import strainr.build_db # Import the module to modify its globals
    from strainr.build_db import BuildDBArgs, DatabaseBuilder

    # 1. Setup: Create dummy FASTA file
    fasta_content = """>seq1
AAANTTTTCCCGGGNA
>seq2
GGNNAACCTTGGNACC
"""
    genome_file = tmp_path / "test_genome.fna"
    genome_file.write_text(fasta_content)
    kmer_len = 4

    # Helper function to get k-mers from the pipeline
    def get_kmers_from_pipeline(
        build_args: BuildDBArgs,
        genome_file_path: pathlib.Path,
        k_len: int,
        force_python: bool
    ) -> set:
        # Create a temporary directory for the worker's output
        # This dir will be inside tmp_path, managed by the test fixture
        worker_output_dir = tmp_path / f"worker_output_{'py' if force_python else 'rs'}_{build_args.skip_n_kmers}"
        if worker_output_dir.exists():
            shutil.rmtree(worker_output_dir) # Clean up from previous helper call if any
        worker_output_dir.mkdir()

        task_tuple = (
            genome_file_path,
            "test_strain", # strain_name
            0,             # strain_idx
            k_len,
            build_args.skip_n_kmers,
            worker_output_dir
        )

        original_rust_available = strainr.build_db._RUST_KMER_COUNTER_AVAILABLE
        original_extract_func = strainr.build_db._extract_kmers_func

        # Access the static method via the class
        worker_fn = DatabaseBuilder._process_and_write_kmers_worker

        if force_python:
            # Temporarily disable Rust path
            strainr.build_db._RUST_KMER_COUNTER_AVAILABLE = False
            strainr.build_db._extract_kmers_func = None
        else:
            # Ensure Rust path is enabled (it should be by default in test env)
            strainr.build_db._RUST_KMER_COUNTER_AVAILABLE = True
            # Attempt to import the actual rust function to reset _extract_kmers_func
            # This assumes kmer_counter_rs.extract_kmer_rs is the actual function
            try:
                from kmer_counter_rs import extract_kmer_rs
                strainr.build_db._extract_kmers_func = extract_kmer_rs
            except ImportError: # pragma: no cover
                # If Rust module not found at all, this path can't run as "rust" path.
                # The test running this helper would then effectively test Python vs Python, or fail.
                # This shouldn't happen if Rust is part of the test environment.
                 pytest.skip("Rust kmer_counter_rs module not available, cannot run Rust path for consistency test.")


        extracted_kmers = set()
        try:
            # Call the worker function (which is now static)
            output_part_file = worker_fn(task_tuple)

            if not output_part_file.exists(): # pragma: no cover
                raise FileNotFoundError(f"Worker did not produce output file: {output_part_file}")

            with gzip.open(output_part_file, "rt") as f_in:
                for line in f_in:
                    kmer_hex = line.strip().split("\t")[0]
                    extracted_kmers.add(bytes.fromhex(kmer_hex))
        finally:
            # Restore original state of globals
            strainr.build_db._RUST_KMER_COUNTER_AVAILABLE = original_rust_available
            strainr.build_db._extract_kmers_func = original_extract_func
            # Clean up worker_output_dir
            if worker_output_dir.exists():
                shutil.rmtree(worker_output_dir)

        return extracted_kmers

    # Scenario 1: skip_n_kmers = False (process N-containing k-mers)
    args_skip_n_false = BuildDBArgs(
        custom=tmp_path, kmerlen=kmer_len, skip_n_kmers=False, procs=1
    )

    # Ensure Rust is available for this path, otherwise skip
    if not strainr.build_db._RUST_KMER_COUNTER_AVAILABLE : # pragma: no cover
        pytest.skip("Rust kmer_counter_rs module not available for Rust path in Scenario 1.")

    kmers_rust_skip_false = get_kmers_from_pipeline(
        args_skip_n_false, genome_file, kmer_len, force_python=False
    )
    kmers_python_skip_false = get_kmers_from_pipeline(
        args_skip_n_false, genome_file, kmer_len, force_python=True
    )

    assert kmers_rust_skip_false == kmers_python_skip_false, \
        f"K-mer sets differ for skip_n_kmers=False. Rust: {len(kmers_rust_skip_false)}, Python: {len(kmers_python_skip_false)}"

    # Scenario 2: skip_n_kmers = True (do NOT process N-containing k-mers)
    args_skip_n_true = BuildDBArgs(
        custom=tmp_path, kmerlen=kmer_len, skip_n_kmers=True, procs=1
    )

    if not strainr.build_db._RUST_KMER_COUNTER_AVAILABLE: # pragma: no cover
        pytest.skip("Rust kmer_counter_rs module not available for Rust path in Scenario 2.")

    kmers_rust_skip_true = get_kmers_from_pipeline(
        args_skip_n_true, genome_file, kmer_len, force_python=False
    )
    kmers_python_skip_true = get_kmers_from_pipeline(
        args_skip_n_true, genome_file, kmer_len, force_python=True
    )

    assert kmers_rust_skip_true == kmers_python_skip_true, \
        f"K-mer sets differ for skip_n_kmers=True. Rust: {len(kmers_rust_skip_true)}, Python: {len(kmers_python_skip_true)}"

    # For sanity check, ensure the two scenarios produce different results from each other
    assert kmers_rust_skip_false != kmers_rust_skip_true, \
        "K-mer sets for skip_n_kmers=False and skip_n_kmers=True should be different with N-containing sequences (Rust path)."
    assert kmers_python_skip_false != kmers_python_skip_true, \
        "K-mer sets for skip_n_kmers=False and skip_n_kmers=True should be different with N-containing sequences (Python path)."

    # Verify that when skip_n_kmers=True, no k-mers contain 'N'
    for kmer in kmers_rust_skip_true:
        assert b"N" not in kmer.upper(), f"Found N in kmer {kmer.decode()} when skip_n_kmers=True (Rust path)"
    for kmer in kmers_python_skip_true:
        assert b"N" not in kmer.upper(), f"Found N in kmer {kmer.decode()} when skip_n_kmers=True (Python path)"

    # Verify that when skip_n_kmers=False, some k-mers *do* contain 'N' (given the input)
    assert any(b"N" in kmer.upper() for kmer in kmers_rust_skip_false), \
        "Expected N-containing kmers when skip_n_kmers=False (Rust path)"
    assert any(b"N" in kmer.upper() for kmer in kmers_python_skip_false), \
        "Expected N-containing kmers when skip_n_kmers=False (Python path)"
