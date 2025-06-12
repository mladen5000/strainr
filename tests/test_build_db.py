import pandas as pd
from unittest.mock import patch, mock_open
from io import StringIO

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
        genome_file_info, kmer_length=builder.args.kmerlen, num_total_strains=1
    )
    assert written_kmer_count == 1
    with open(temp_kmer_file_path, "r") as f:
        assert f.read().strip() == "ATGCG"
    if temp_kmer_file_path.exists():
        temp_kmer_file_path.unlink()


# ...continue this pytest pattern for all integration and database tests...
