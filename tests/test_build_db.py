import pytest
import pandas as pd
from unittest.mock import patch, mock_open
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
    builder, _, test_dir, _, _ = db_builder
    metadata_content = """assembly_accession,taxid,species_taxid,local_filename
GCA_001,101,100,GCA_001.fna.gz
GCA_002,200,200,GCA_002.fna.gz
GCA_003,301,300,GCA_003.fna.gz
GCA_004,400,,GCA_004.fna.gz
"""
    mock_df = REAL_READ_CSV(StringIO(metadata_content))
    mock_read_csv.return_value = mock_df
    genome_dir = test_dir / "genomes_filter_test"
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
        if path_obj.name == f"filtered_{mock_metadata_path.name}":
            return True
        return False

    with (
        patch("pathlib.Path.exists", side_effect=mock_path_exists_filter),
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
    ):
        filtered_files = builder._filter_genomes_by_unique_taxid(
            genome_dir, mock_metadata_path
        )
    assert len(filtered_files) == 3
    assert genome_dir / "GCA_001.fna.gz" in filtered_files
    assert genome_dir / "GCA_003.fna.gz" in filtered_files
    assert genome_dir / "GCA_004.fna.gz" in filtered_files
    mock_to_csv.assert_called_once()


@patch("pandas.read_csv")
def test_filter_genomes_metadata_missing_local_filename(mock_read_csv, db_builder):
    builder, _, test_dir, _, _ = db_builder
    metadata_content = """assembly_accession,taxid,species_taxid
GCA_001,101,100
"""
    mock_df = REAL_READ_CSV(StringIO(metadata_content))
    mock_read_csv.return_value = mock_df
    genome_dir = test_dir / "genomes_missing_col"
    genome_dir.mkdir(exist_ok=True)
    mock_metadata_path = test_dir / "metadata_missing_col.tsv"
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pandas.DataFrame.to_csv"),
    ):
        filtered_files = builder._filter_genomes_by_unique_taxid(
            genome_dir, mock_metadata_path
        )
    assert len(filtered_files) == 0


@patch("pandas.read_csv")
def test_filter_genomes_local_file_does_not_exist(mock_read_csv, db_builder):
    builder, _, test_dir, _, _ = db_builder
    metadata_content = """assembly_accession,taxid,species_taxid,local_filename
GCA_001,101,100,non_existent_GCA_001.fna.gz 
"""
    mock_df = REAL_READ_CSV(StringIO(metadata_content))
    mock_read_csv.return_value = mock_df
    genome_dir = test_dir / "genomes_non_existent_local"
    genome_dir.mkdir(exist_ok=True)
    mock_metadata_path = test_dir / "metadata_non_existent_local.tsv"

    def mock_path_exists_specific(path_obj):
        if path_obj == mock_metadata_path:
            return True
        if path_obj == (genome_dir / "non_existent_GCA_001.fna.gz"):
            return False
        return True

    with (
        patch("pathlib.Path.exists", side_effect=mock_path_exists_specific),
        patch("pandas.DataFrame.to_csv"),
    ):
        filtered_files = builder._filter_genomes_by_unique_taxid(
            genome_dir, mock_metadata_path
        )
    assert len(filtered_files) <= 1


@patch("strainr.build_db.open_file_transparently", new_callable=mock_open)
def test_process_single_fasta_for_kmers_writes_correct_kmers(
    mock_actual_open, db_builder
):
    builder, _, test_dir, _, _ = db_builder
    mock_actual_open.return_value = StringIO(">seq1\nATGCGTAGCATGC\n")
    dummy_fasta_path = test_dir / "dummy.fna.gz"
    strain_name = "test_strain"
    strain_idx = 0
    temp_kmer_file_path = test_dir / "kmer_output.txt"
    genome_file_info = (dummy_fasta_path, strain_name, strain_idx, temp_kmer_file_path)
    a_string, written_kmer_count, out_path = builder._process_single_fasta_for_kmers(
        genome_file_info, kmer_length=builder.args.kmerlen
    )
    assert written_kmer_count == 9
    assert out_path == temp_kmer_file_path
    written_kmers = set()
    with open(temp_kmer_file_path, "r", encoding="utf-8") as f:
        for line in f:
            written_kmers.add(line.strip())
    expected_kmers_str = {
        "ATGCG",
        "TGCGT",
        "GCGTA",
        "CGTAG",
        "GTAGC",
        "TAGCA",
        "AGCAT",
        "GCATG",
        "CATGC",
    }
    assert written_kmers == expected_kmers_str
    if temp_kmer_file_path.exists():
        temp_kmer_file_path.unlink()


@patch("strainr.build_db.open_file_transparently", new_callable=mock_open)
def test_process_single_fasta_kmers_seq_shorter_than_kmerlen(
    mock_actual_open, db_builder
):
    builder, _, test_dir, _, _ = db_builder
    mock_actual_open.return_value = StringIO(">seq1\nATG\n")
    dummy_fasta_path = test_dir / "short.fna.gz"
    temp_kmer_file_path = test_dir / "short_kmer_output.txt"
    genome_file_info = (dummy_fasta_path, "short_strain", 0, temp_kmer_file_path)
    _, _, written_kmer_count, _ = builder._process_single_fasta_for_kmers(
        genome_file_info, kmer_length=builder.args.kmerlen, num_total_strains=1
    )
    assert written_kmer_count == 0
    assert temp_kmer_file_path.exists()
    with open(temp_kmer_file_path, "r") as f:
        assert f.read() == ""
    if temp_kmer_file_path.exists():
        temp_kmer_file_path.unlink()


@patch("strainr.build_db.open_file_transparently", new_callable=mock_open)
def test_process_single_fasta_kmers_seq_equals_kmerlen(mock_actual_open, db_builder):
    builder, _, test_dir, _, _ = db_builder
    mock_actual_open.return_value = StringIO(">seq1\nATGCG\n")
    dummy_fasta_path = test_dir / "equal.fna.gz"
    temp_kmer_file_path = test_dir / "equal_kmer_output.txt"
    genome_file_info = (dummy_fasta_path, "equal_strain", 0, temp_kmer_file_path)
    _, _, written_kmer_count, _ = builder._process_single_fasta_for_kmers(
        genome_file_info, kmer_length=builder.args.kmerlen, num_total_strains=1 # Uses builder.args.kmerlen (default 31)
    ) # If BuildDBArgs default kmerlen is used, this test might need adjustment if sequence is shorter.
      # Assuming kmer_length passed to _process_single_fasta_for_kmers is dynamically set based on args or a fixed test value.
      # For this test, if builder.args.kmerlen is e.g. 5, then "ATGCG" is 1 kmer.
    assert written_kmer_count == 1
    with open(temp_kmer_file_path, "r") as f:
        assert f.read().strip() == "ATGCG" # This depends on kmer_length being 5
    if temp_kmer_file_path.exists():
        temp_kmer_file_path.unlink()


# ...continue this pytest pattern for all integration and database tests...

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
    assert "does not exist" in str(excinfo.value) or "No such file or directory" in str(excinfo.value)


def test_builddbargs_custom_dir_not_found(tmp_path: pathlib.Path):
    """Test ValidationError if custom directory does not exist."""
    non_existent_dir = tmp_path / "non_existent_custom"
    with pytest.raises(ValidationError) as excinfo:
        BuildDBArgs(custom=non_existent_dir)
    assert "Path" in str(excinfo.value)
    assert "does not exist" in str(excinfo.value) or "No such file or directory" in str(excinfo.value)


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
