"""
Pytest unit tests for the StrainKmerDatabase class from src.strainr.database.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""
import pytest
import pandas as pd
import numpy as np
import pathlib
# import pickle # Pickle is no longer used directly for db files
from typing import List, Dict, Union, Any, Optional
from unittest.mock import patch, MagicMock

from src.strainr.database import StrainKmerDatabase
from src.strainr.genomic_types import KmerCountDict, CountVector # Assuming these are relevant

# --- Helper Functions & Fixtures ---

KMER_LEN_FOR_TESTS = 5 # Keep it short for easier test data

def create_dummy_dataframe_for_db(
    kmer_list: List[Union[str, bytes]], 
    strain_names: List[str], 
    counts: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Creates a DataFrame suitable for StrainKmerDatabase."""
    if not kmer_list and not strain_names:
        return pd.DataFrame()
    if not kmer_list:
        return pd.DataFrame(columns=strain_names)
    if not strain_names:
        return pd.DataFrame(index=pd.Index(kmer_list, name="kmer_index")) # Use a name for index
        
    if counts is None:
        counts_shape = (len(kmer_list), len(strain_names))
        if 0 in counts_shape:
            return pd.DataFrame(index=pd.Index(kmer_list, name="kmer_index"), columns=strain_names)
        counts = np.random.randint(0, 255, size=counts_shape, dtype=np.uint8)
    
    return pd.DataFrame(counts, index=pd.Index(kmer_list, name="kmer_index"), columns=strain_names)

@pytest.fixture
def strain_names_fixture_db() -> List[str]: # Renamed to avoid conflict if other test files use similar names
    return ["StrainDbX", "StrainDbY", "StrainDbZ"]

@pytest.fixture
def valid_kmers_str_db(default_kmer_length_db: int) -> List[str]:
    return ["A" * default_kmer_length_db, "C" * default_kmer_length_db, "G" * default_kmer_length_db]

@pytest.fixture
def valid_kmers_bytes_db(valid_kmers_str_db: List[str]) -> List[bytes]:
    return [k.encode('utf-8') for k in valid_kmers_str_db]

@pytest.fixture
def default_kmer_length_db() -> int:
    return KMER_LEN_FOR_TESTS

@pytest.fixture
def parquet_db_file_db( 
    tmp_path: pathlib.Path, 
    request: pytest.FixtureRequest, 
    strain_names_fixture_db: List[str],
    default_kmer_length_db: int 
) -> pathlib.Path:
    params = getattr(request, "param", {})
    kmer_type = params.get("kmer_type", "str") 
    
    default_kmers: List[Union[str,bytes]]
    if kmer_type == "str":
        default_kmers = [char * default_kmer_length_db for char in ["A", "C", "G"]]
    else: 
        default_kmers = [char.encode('utf-8') * default_kmer_length_db for char in ["A", "C", "G"]]

    kmer_data = params.get("kmer_data", default_kmers)
    df_counts = params.get("df_counts") 
    empty_df = params.get("empty_df", False)
    no_kmers_df = params.get("no_kmers_df", False) 
    no_strains_df = params.get("no_strains_df", False)

    db_file = tmp_path / f"test_db_{kmer_type}.parquet" # Changed extension
    df: pd.DataFrame

    if empty_df:
        df = create_dummy_dataframe_for_db([], [])
    elif no_kmers_df:
        df = create_dummy_dataframe_for_db([], strain_names_fixture_db)
    elif no_strains_df:
        df = create_dummy_dataframe_for_db(kmer_data, [])
    else:
        df = create_dummy_dataframe_for_db(kmer_data, strain_names_fixture_db, df_counts)

    df.to_parquet(db_file, index=True) # Changed saving method
    return db_file

# --- Tests for __init__ and _load_database ---

@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "str"}], indirect=True) # Updated fixture name
def test_db_init_successful_kmers_as_str(
    parquet_db_file_db: pathlib.Path, # Updated fixture name
    default_kmer_length_db: int, 
    strain_names_fixture_db: List[str],
    valid_kmers_str_db: List[str] 
):
    db = StrainKmerDatabase(parquet_db_file_db, kmer_length=default_kmer_length_db) # Use updated fixture
    assert db.kmer_length == default_kmer_length_db
    assert db.num_strains == len(strain_names_fixture_db)
    assert db.num_kmers == len(valid_kmers_str_db) # Based on default kmer generation
    assert all(isinstance(k, bytes) for k in db.kmer_lookup_dict.keys())
    assert db.strain_names == strain_names_fixture_db

@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "bytes"}], indirect=True) # Updated fixture name
def test_db_init_successful_kmers_as_bytes(
    parquet_db_file_db: pathlib.Path, # Updated fixture name
    default_kmer_length_db: int, 
    strain_names_fixture_db: List[str],
    valid_kmers_bytes_db: List[bytes]
):
    db = StrainKmerDatabase(parquet_db_file_db, kmer_length=default_kmer_length_db) # Use updated fixture
    assert db.kmer_length == default_kmer_length_db
    assert db.num_strains == len(strain_names_fixture_db)
    assert db.num_kmers == len(valid_kmers_bytes_db)
    assert all(isinstance(k, bytes) for k in db.kmer_lookup_dict.keys())

@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "str"}], indirect=True) # Updated fixture name
def test_db_init_kmer_length_mismatch_warning(
    parquet_db_file_db: pathlib.Path, # Updated fixture name
    default_kmer_length_db: int, 
    capsys: pytest.CaptureFixture
):
    constructor_kmer_len = default_kmer_length_db + 2 
    db = StrainKmerDatabase(parquet_db_file_db, kmer_length=constructor_kmer_len) # Use updated fixture
    captured = capsys.readouterr()
    assert f"Warning: Initial expected k-mer length was {constructor_kmer_len}" in captured.out
    assert f"but found {default_kmer_length_db}" in captured.out
    assert f"Using actual length from database: {default_kmer_length_db}" in captured.out
    assert db.kmer_length == default_kmer_length_db

def test_db_init_file_not_found():
    with pytest.raises(FileNotFoundError, match="Database file not found or is not a file:"):
        StrainKmerDatabase("non_existent_file_for_db.parquet", kmer_length=5) # Updated extension

@patch("pandas.read_parquet") # Patched to read_parquet
def test_db_init_empty_or_corrupt_parquet_file_error(mock_read_parquet: MagicMock, tmp_path: pathlib.Path): # Renamed mock
    # Simulate an error that pd.read_parquet might raise for empty/corrupt files
    # For example, pyarrow.lib.ArrowInvalid or a general ValueError/IOError
    mock_read_parquet.side_effect = ValueError("Failed to read Parquet file") 
    db_path = tmp_path / "empty_or_corrupt_db.parquet" # Updated extension
    db_path.touch() 
    # Match the RuntimeError message from StrainKmerDatabase._load_database
    with pytest.raises(RuntimeError, match="Failed to read or process Parquet database"):
        StrainKmerDatabase(db_path, kmer_length=5)

@patch("pandas.read_parquet") # Patched to read_parquet
def test_db_init_not_a_dataframe_error(mock_read_parquet: MagicMock, tmp_path: pathlib.Path): # Renamed mock
    mock_read_parquet.return_value = {"not_a_df": True} 
    db_path = tmp_path / "not_df_db.parquet" # Updated extension
    db_path.touch()
    with pytest.raises(RuntimeError, match="is not a pandas DataFrame"):
        StrainKmerDatabase(db_path, kmer_length=5)

@pytest.mark.parametrize("parquet_db_file_db", [{"empty_df": True}], indirect=True) # Updated fixture name
def test_db_init_empty_dataframe_value_error(parquet_db_file_db: pathlib.Path): # Updated fixture name
    with pytest.raises(ValueError, match="Database DataFrame from .* is empty"):
        StrainKmerDatabase(parquet_db_file_db, kmer_length=5) # Use updated fixture

@pytest.mark.parametrize("parquet_db_file_db", [{"no_kmers_df": True}], indirect=True) # Updated fixture name
def test_db_init_dataframe_no_kmers_error(parquet_db_file_db: pathlib.Path): # Updated fixture name
    with pytest.raises(ValueError, match="has no k-mers (empty index)"):
        StrainKmerDatabase(parquet_db_file_db, kmer_length=5) # Use updated fixture

@pytest.mark.parametrize("parquet_db_file_db", [{"no_strains_df": True}], indirect=True) # Updated fixture name
def test_db_init_dataframe_no_strains_loads_no_counts(
    parquet_db_file_db: pathlib.Path, # Updated fixture name
    default_kmer_length_db: int,
    valid_kmers_str_db: List[str] 
):
    with pytest.raises(ValueError, match="Database contains no strain information"):
        StrainKmerDatabase(parquet_db_file_db, kmer_length=default_kmer_length_db) # Use updated fixture


@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_data": ["A" * KMER_LEN_FOR_TESTS, "C" * (KMER_LEN_FOR_TESTS-1)]}], indirect=True) # Updated fixture name
def test_db_init_inconsistent_kmer_lengths_error(parquet_db_file_db: pathlib.Path, default_kmer_length_db: int): # Updated fixture name
    with pytest.raises(ValueError, match="Inconsistent k-mer length found"):
        StrainKmerDatabase(parquet_db_file_db, kmer_length=default_kmer_length_db) # Use updated fixture

@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_data": [12345, "A"*KMER_LEN_FOR_TESTS ]}], indirect=True) # Updated fixture name
def test_db_init_unsupported_kmer_type_in_index_error(parquet_db_file_db: pathlib.Path, default_kmer_length_db: int): # Updated fixture name
    with pytest.raises(ValueError, match="Unsupported k-mer type in DataFrame index: <class 'int'>"):
        StrainKmerDatabase(parquet_db_file_db, kmer_length=default_kmer_length_db) # Use updated fixture
        
@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "str", "df_counts": np.array([["X", "Y", "Z"], ["U", "V", "W"], ["P", "Q", "R"]])}], indirect=True) # Updated fixture name
def test_db_init_non_numeric_data_in_df_error(parquet_db_file_db: pathlib.Path, default_kmer_length_db: int): # Updated fixture name
    with pytest.raises(RuntimeError, match="Could not convert DataFrame values to np.uint8"):
        StrainKmerDatabase(parquet_db_file_db, kmer_length=default_kmer_length_db) # Use updated fixture


# --- Tests for lookup_kmer ---
@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "str", "kmer_data": ["ATGCG", "CGTAA"]}], indirect=True) # Updated fixture name
def test_db_lookup_kmer_found_and_not_found(parquet_db_file_db: pathlib.Path, strain_names_fixture_db: List[str]): # Updated fixture name
    db = StrainKmerDatabase(parquet_db_file_db, kmer_length=5) # Use updated fixture
    
    known_kmer_str = "ATGCG"
    known_kmer_bytes = known_kmer_str.encode('utf-8')
    count_vector = db.lookup_kmer(known_kmer_bytes)
    assert count_vector is not None
    assert isinstance(count_vector, np.ndarray)
    assert len(count_vector) == len(strain_names_fixture_db)
    assert count_vector.dtype == np.uint8

    unknown_kmer_bytes = ("XXXXX").encode('utf-8')
    assert db.lookup_kmer(unknown_kmer_bytes) is None

    incorrect_length_kmer = ("ATGC").encode('utf-8') # Length 4
    assert db.lookup_kmer(incorrect_length_kmer) is None


# --- Tests for get_database_stats ---
@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "str", "kmer_data": ["AAAAA", "CCCCC", "GGGGG", "TTTTT", "ACGTA", "TGCAC"]}], indirect=True) # Updated fixture name
def test_db_get_database_stats(parquet_db_file_db: pathlib.Path, default_kmer_length_db: int, strain_names_fixture_db: List[str]): # Updated fixture name
    db = StrainKmerDatabase(parquet_db_file_db, kmer_length=default_kmer_length_db) # Use updated fixture
    stats = db.get_database_stats()

    assert stats["num_strains"] == len(strain_names_fixture_db)
    assert stats["num_kmers"] == 6 
    assert stats["kmer_length"] == default_kmer_length_db
    assert stats["database_path"] == str(parquet_db_file_db.resolve()) # Use updated fixture
    assert len(stats["strain_names_preview"]) <= 5
    assert stats["strain_names_preview"] == strain_names_fixture_db[:5] 
    assert stats["total_strain_names"] == len(strain_names_fixture_db)


# --- Tests for validate_kmer_length ---
@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "str"}], indirect=True) # Updated fixture name
def test_db_validate_kmer_length(parquet_db_file_db: pathlib.Path, default_kmer_length_db: int): # Updated fixture name
    db = StrainKmerDatabase(parquet_db_file_db, kmer_length=default_kmer_length_db) # Use updated fixture
    
    correct_len_str = "X" * default_kmer_length_db
    incorrect_len_str = "X" * (default_kmer_length_db - 1)
    correct_len_bytes = b"Y" * default_kmer_length_db
    incorrect_len_bytes = b"Y" * (default_kmer_length_db - 1)

    assert db.validate_kmer_length(correct_len_str) is True
    assert db.validate_kmer_length(incorrect_len_str) is False
    assert db.validate_kmer_length(correct_len_bytes) is True
    assert db.validate_kmer_length(incorrect_len_bytes) is False
    
    assert db.validate_kmer_length(12345) is False 
    assert db.validate_kmer_length(["A"] * default_kmer_length_db) is False
```
