"""
Pytest unit tests for the KmerStrainDatabase class from src.strainr.kmer_database.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""

import pytest
import pandas as pd
import numpy as np
import pathlib
# import pickle # Pickle is no longer used directly for db files
from typing import List, Dict, Union, Any, Optional
from unittest.mock import patch, MagicMock

# Assuming src.strainr.* is in PYTHONPATH or tests are run from a suitable root
from strainr.kmer_database import StrainKmerDb
from strainr.genomic_types import CountVector

Kmer = Union[str, bytes]  # Kmer can be either str or bytes, depending on context

# --- Helper Functions & Fixtures ---

KMER_LEN_FOR_TESTS_KDB = (
    4  # Using a different constant name to avoid potential conflicts
)


def create_dummy_dataframe_for_kdb(  # Renamed to avoid conflict
    kmer_list: List[Union[str, bytes]],
    strain_names: List[str],
    counts: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Creates a DataFrame suitable for KmerStrainDatabase."""
    if not kmer_list and not strain_names:
        return pd.DataFrame()
    if not kmer_list:
        return pd.DataFrame(columns=strain_names)
    if not strain_names:
        # DataFrame with index but no columns
        return pd.DataFrame(
            index=pd.Index(kmer_list, name="kmer_idx")
        )  # Use a name for index

    if counts is None:
        counts_shape = (len(kmer_list), len(strain_names))
        if 0 in counts_shape:
            return pd.DataFrame(
                index=pd.Index(kmer_list, name="kmer_idx"), columns=strain_names
            )
        counts = np.random.randint(
            0, 256, size=counts_shape, dtype=np.uint8
        )  # KmerStrainDatabase uses uint8

    return pd.DataFrame(
        counts, index=pd.Index(kmer_list, name="kmer_idx"), columns=strain_names
    )


@pytest.fixture
def strain_names_fixture_kdb() -> List[str]:  # Renamed
    return ["StrainKDB_1", "StrainKDB_2"]


@pytest.fixture
def default_kmer_length_kdb() -> int:  # Renamed
    return KMER_LEN_FOR_TESTS_KDB


@pytest.fixture
def sample_kmers_str_kdb(default_kmer_length_kdb: int) -> List[str]:  # Renamed
    return [
        "AAAA",
        "CCCC",
        "GGGG",  # Ensure these are length default_kmer_length_kdb
    ][:3]  # Take first 3, ensure they match length


@pytest.fixture
def sample_kmers_bytes_kdb(sample_kmers_str_kdb: List[str]) -> List[Kmer]:  # Renamed
    return [k.encode("utf-8") for k in sample_kmers_str_kdb]


@pytest.fixture
def parquet_kdb_path( # Renamed fixture
    tmp_path: pathlib.Path, 
    request: pytest.FixtureRequest, 
    default_kmer_length_kdb: int, 
    strain_names_fixture_kdb: List[str], 
    sample_kmers_str_kdb: List[str], 
    sample_kmers_bytes_kdb: List[Kmer]
) -> pathlib.Path:
    params = getattr(request, "param", {})
    db_file = tmp_path / f"test_kdb_{request.node.name}.parquet" # Changed extension
    
    df_to_save: pd.DataFrame # Renamed variable

    kmer_type = params.get("kmer_type", "str")
    kmers_for_df: List[Union[str, bytes]]
    if params.get("custom_kmers"):
        kmers_for_df = params["custom_kmers"]
    elif kmer_type == "str":
        kmers_for_df = sample_kmers_str_kdb
    else:  # bytes
        kmers_for_df = sample_kmers_bytes_kdb

    counts_data = params.get("custom_counts")
    strains = params.get("custom_strains", strain_names_fixture_kdb)

    if params.get("empty_df", False):
        df_to_save = create_dummy_dataframe_for_kdb([], [])
    elif params.get("no_kmers", False): 
        df_to_save = create_dummy_dataframe_for_kdb([], strains)
    elif params.get("no_strains", False): 
        df_to_save = create_dummy_dataframe_for_kdb(kmers_for_df, [])
    elif params.get("inconsistent_kmer_len", False):
        if kmer_type == "str":
            kmers_for_df = [sample_kmers_str_kdb[0], sample_kmers_str_kdb[1][:default_kmer_length_kdb-1]]
        else: 
            kmers_for_df = [sample_kmers_bytes_kdb[0], sample_kmers_bytes_kdb[1][:default_kmer_length_kdb-1]]
        df_to_save = create_dummy_dataframe_for_kdb(kmers_for_df, strains)
    elif params.get("unsupported_kmer_type", False):
        kmers_for_df = [123, 456] # type: ignore
        df_to_save = create_dummy_dataframe_for_kdb(kmers_for_df, strains)
    elif params.get("non_numeric_counts", False):
        counts_data = np.array([["str_val1", "str_val2"], ["str_val3", "str_val4"]], dtype=object) # type: ignore
        df_to_save = create_dummy_dataframe_for_kdb(kmers_for_df[:2], strains, counts_data) 
    elif params.get("non_unique_kmers", False):
        kmers_for_df = [kmers_for_df[0], kmers_for_df[0]] + kmers_for_df[1:] 
        df_to_save = create_dummy_dataframe_for_kdb(kmers_for_df, strains)
    else: 
        df_to_save = create_dummy_dataframe_for_kdb(kmers_for_df, strains, counts_data)
        
    df_to_save.to_parquet(db_file, index=True) # Changed saving method
    return db_file


# --- Tests for __init__ and _load_database ---

@pytest.mark.parametrize("parquet_kdb_path", [{"kmer_type": "str"}], indirect=True) # Renamed fixture
def test_kdb_init_success_str_kmers(
    parquet_kdb_path: pathlib.Path, # Renamed fixture
    default_kmer_length_kdb: int, 
    strain_names_fixture_kdb: List[str], 
    sample_kmers_str_kdb: List[str]
):
    db = KmerStrainDatabase(parquet_kdb_path, expected_kmer_length=default_kmer_length_kdb) # Use renamed fixture
    assert db.kmer_length == default_kmer_length_kdb
    assert db.num_strains == len(strain_names_fixture_kdb)
    assert db.num_kmers == len(sample_kmers_str_kdb)
    assert all(isinstance(k, bytes) for k in db.kmer_to_counts_map.keys())
    assert db.strain_names == strain_names_fixture_kdb

@pytest.mark.parametrize("parquet_kdb_path", [{"kmer_type": "bytes"}], indirect=True) # Renamed fixture
def test_kdb_init_success_bytes_kmers(
    parquet_kdb_path: pathlib.Path, # Renamed fixture
    default_kmer_length_kdb: int, 
    strain_names_fixture_kdb: List[str], 
    sample_kmers_bytes_kdb: List[Kmer]
):
    db = KmerStrainDatabase(parquet_kdb_path, expected_kmer_length=default_kmer_length_kdb) # Use renamed fixture
    assert db.kmer_length == default_kmer_length_kdb
    assert db.num_strains == len(strain_names_fixture_kdb)
    assert db.num_kmers == len(sample_kmers_bytes_kdb)
    assert all(isinstance(k, bytes) for k in db.kmer_to_counts_map.keys())

@pytest.mark.parametrize("parquet_kdb_path", [{"kmer_type": "str"}], indirect=True) # Renamed fixture
def test_kdb_init_kmer_length_inferred(parquet_kdb_path: pathlib.Path, default_kmer_length_kdb: int, capsys: pytest.CaptureFixture): # Renamed fixture
    db = KmerStrainDatabase(parquet_kdb_path, expected_kmer_length=None) # Use renamed fixture
    assert db.kmer_length == default_kmer_length_kdb
    captured = capsys.readouterr()
    assert (
        f"K-mer length inferred from first k-mer: {default_kmer_length_kdb}"
        in captured.out
    )


@pytest.mark.parametrize("parquet_kdb_path", [{"kmer_type": "str"}], indirect=True) # Renamed fixture
def test_kdb_init_kmer_length_mismatch_error(parquet_kdb_path: pathlib.Path, default_kmer_length_kdb: int): # Renamed fixture
    with pytest.raises(ValueError, match="does not match length of first k-mer"):
        KmerStrainDatabase(parquet_kdb_path, expected_kmer_length=default_kmer_length_kdb + 1) # Use renamed fixture

def test_kdb_init_file_not_found_error():
    with pytest.raises(FileNotFoundError, match="Database file not found or is not a file:"):
        KmerStrainDatabase("nonexistent_kdb.parquet") # Updated extension

@patch("pandas.read_parquet") # Patched to read_parquet
def test_kdb_init_parquet_read_error(mock_read_parquet: MagicMock, tmp_path: pathlib.Path): # Renamed test and mock
    mock_read_parquet.side_effect = ValueError("Simulated Parquet read error") # Simulate Parquet error
    db_file = tmp_path / "bad_parquet_kdb.parquet" # Updated extension
    db_file.touch()
    # Adjusted error message to match potential Parquet loading issues in KmerStrainDatabase
    with pytest.raises(RuntimeError, match="Failed to read or process Parquet database file"): 
        KmerStrainDatabase(db_file)

@patch("pandas.read_parquet") # Patched to read_parquet
def test_kdb_init_parquet_not_dataframe_error(mock_read_parquet: MagicMock, tmp_path: pathlib.Path): # Renamed test and mock
    mock_read_parquet.return_value = "this is not a dataframe"
    db_file = tmp_path / "not_df_kdb.parquet" # Updated extension
    db_file.touch()
    with pytest.raises(
        RuntimeError, match="Data loaded from .* is not a pandas DataFrame"
    ):
        StrainKmerDb(db_file)


@pytest.mark.parametrize("parquet_kdb_path", [{"empty_df": True}], indirect=True) # Renamed fixture
def test_kdb_init_empty_dataframe_error(parquet_kdb_path: pathlib.Path): # Renamed fixture
    with pytest.raises(ValueError, match="Loaded database is empty"):
        KmerStrainDatabase(parquet_kdb_path) # Use renamed fixture

@pytest.mark.parametrize("parquet_kdb_path", [{"no_kmers": True}], indirect=True) # Renamed fixture
def test_kdb_init_no_kmers_error(parquet_kdb_path: pathlib.Path): # Renamed fixture
    with pytest.raises(ValueError, match="Database contains no k-mers"):
        KmerStrainDatabase(parquet_kdb_path) # Use renamed fixture

@pytest.mark.parametrize("parquet_kdb_path", [{"no_strains": True}], indirect=True) # Renamed fixture
def test_kdb_init_no_strains_error(parquet_kdb_path: pathlib.Path): # Renamed fixture
    with pytest.raises(ValueError, match="Database contains no strain information"):
        KmerStrainDatabase(parquet_kdb_path) # Use renamed fixture

@pytest.mark.parametrize("parquet_kdb_path", [{"inconsistent_kmer_len": True}], indirect=True) # Renamed fixture
def test_kdb_init_inconsistent_kmer_length_error(parquet_kdb_path: pathlib.Path, default_kmer_length_kdb: int): # Renamed fixture
    with pytest.raises(ValueError, match=f"Inconsistent k-mer string length at index 1. Expected {default_kmer_length_kdb}"):
        KmerStrainDatabase(parquet_kdb_path, expected_kmer_length=None) # Use renamed fixture

@pytest.mark.parametrize("parquet_kdb_path", [{"unsupported_kmer_type": True}], indirect=True) # Renamed fixture
def test_kdb_init_unsupported_kmer_type_error(parquet_kdb_path: pathlib.Path): # Renamed fixture
    with pytest.raises(TypeError, match="Unsupported k-mer type in index: <class 'int'>"):
        KmerStrainDatabase(parquet_kdb_path) # Use renamed fixture

@pytest.mark.parametrize("parquet_kdb_path", [{"custom_kmers": ["AA", "ÄA"], "kmer_type":"str"}], indirect=True) # Renamed fixture
def test_kdb_init_inconsistent_byte_len_after_encoding_error(parquet_kdb_path: pathlib.Path, capsys): # Renamed fixture
    with pytest.raises(ValueError, match="Post-encoding/cast k-mer byte length validation failed"):
         KmerStrainDatabase(parquet_kdb_path, expected_kmer_length=2) # Use renamed fixture

@pytest.mark.parametrize("parquet_kdb_path", [{"non_numeric_counts": True}], indirect=True) # Renamed fixture
def test_kdb_init_non_numeric_counts_error(parquet_kdb_path: pathlib.Path): # Renamed fixture
    with pytest.raises(TypeError, match="Could not convert database values to count matrix"):
        KmerStrainDatabase(parquet_kdb_path) # Use renamed fixture

@pytest.mark.parametrize("parquet_kdb_path", [{"non_unique_kmers": True}], indirect=True) # Renamed fixture
def test_kdb_init_non_unique_kmers_warning(parquet_kdb_path: pathlib.Path, capsys: pytest.CaptureFixture): # Renamed fixture
    KmerStrainDatabase(parquet_kdb_path) 
    captured = capsys.readouterr()
    assert "Warning: K-mer index in .* is not unique." in captured.out


# --- Tests for get_strain_counts_for_kmer ---
@pytest.mark.parametrize("parquet_kdb_path", [{"kmer_type": "str", "valid_data": True}], indirect=True) # Renamed fixture
def test_kdb_get_strain_counts_for_kmer(
    parquet_kdb_path: pathlib.Path, # Renamed fixture
    default_kmer_length_kdb: int, 
    strain_names_fixture_kdb: List[str], 
    sample_kmers_str_kdb: List[str]
):
    db = KmerStrainDatabase(parquet_kdb_path, expected_kmer_length=default_kmer_length_kdb) # Use renamed fixture
    
    known_kmer_bytes = sample_kmers_str_kdb[0].encode('utf-8')
    counts = db.get_strain_counts_for_kmer(known_kmer_bytes)
    assert counts is not None
    assert isinstance(counts, np.ndarray)
    assert counts.dtype == np.uint8
    assert len(counts) == len(strain_names_fixture_kdb)

    unknown_kmer = ("Z" * default_kmer_length_kdb).encode("utf-8")
    assert db.get_strain_counts_for_kmer(unknown_kmer) is None

    wrong_length_kmer = (sample_kmers_str_kdb[0][:-1]).encode('utf-8') 
    assert db.get_strain_counts_for_kmer(wrong_length_kmer) is None


# --- Tests for __len__ ---
@pytest.mark.parametrize("parquet_kdb_path", [{"kmer_type": "str", "valid_data": True}], indirect=True) # Renamed fixture
def test_kdb_len_method(parquet_kdb_path: pathlib.Path, sample_kmers_str_kdb: List[str]): # Renamed fixture
    db = KmerStrainDatabase(parquet_kdb_path) # Use renamed fixture
    assert len(db) == len(sample_kmers_str_kdb)

@pytest.mark.parametrize("parquet_kdb_path", [{"non_unique_kmers": True, "custom_kmers": ["AAAA", "AAAA", "CCCC", "GGGG"], "kmer_type":"str"}], indirect=True) # Renamed fixture
def test_kdb_len_method_non_unique_kmers(parquet_kdb_path: pathlib.Path, default_kmer_length_kdb: int): # Renamed fixture
    db = KmerStrainDatabase(parquet_kdb_path, expected_kmer_length=4) # Use renamed fixture
    assert len(db) == 3 


# --- Tests for __contains__ ---
@pytest.mark.parametrize("parquet_kdb_path", [{"kmer_type": "str", "valid_data": True}], indirect=True) # Renamed fixture
def test_kdb_contains_method(parquet_kdb_path: pathlib.Path, sample_kmers_str_kdb: List[str], default_kmer_length_kdb: int): # Renamed fixture
    db = KmerStrainDatabase(parquet_kdb_path) # Use renamed fixture
    
    known_kmer_bytes = sample_kmers_str_kdb[0].encode('utf-8')
    assert known_kmer_bytes in db

    unknown_kmer = ("Z" * default_kmer_length_kdb).encode("utf-8")
    assert unknown_kmer not in db
    
    known_kmer_as_str = sample_kmers_str_kdb[0]
    assert known_kmer_as_str not in db # type: ignore 
```
