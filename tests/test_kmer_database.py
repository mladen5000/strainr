"""
Pytest unit tests for the KmerStrainDatabase class from src.strainr.kmer_database.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""
import pytest
import pandas as pd
import numpy as np
import pathlib
import pickle
from typing import List, Dict, Union, Any, Optional # Added Optional
from unittest.mock import patch, MagicMock

# Assuming src.strainr.* is in PYTHONPATH or tests are run from a suitable root
from src.strainr.kmer_database import KmerStrainDatabase, Kmer, CountVector 

# --- Helper Functions & Fixtures ---

KMER_LEN_FOR_TESTS_KDB = 4 # Using a different constant name to avoid potential conflicts

def create_dummy_dataframe_for_kdb( # Renamed to avoid conflict
    kmer_list: List[Union[str, bytes]], 
    strain_names: List[str], 
    counts: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Creates a DataFrame suitable for KmerStrainDatabase."""
    if not kmer_list and not strain_names:
        return pd.DataFrame()
    if not kmer_list:
        return pd.DataFrame(columns=strain_names)
    if not strain_names:
        # DataFrame with index but no columns
        return pd.DataFrame(index=pd.Index(kmer_list, name="kmer_idx")) # Use a name for index
        
    if counts is None:
        counts_shape = (len(kmer_list), len(strain_names))
        if 0 in counts_shape :
             return pd.DataFrame(index=pd.Index(kmer_list, name="kmer_idx"), columns=strain_names)
        counts = np.random.randint(0, 256, size=counts_shape, dtype=np.uint8) # KmerStrainDatabase uses uint8
    
    return pd.DataFrame(counts, index=pd.Index(kmer_list, name="kmer_idx"), columns=strain_names)

@pytest.fixture
def strain_names_fixture_kdb() -> List[str]: # Renamed
    return ["StrainKDB_1", "StrainKDB_2"]

@pytest.fixture
def default_kmer_length_kdb() -> int: # Renamed
    return KMER_LEN_FOR_TESTS_KDB

@pytest.fixture
def sample_kmers_str_kdb(default_kmer_length_kdb: int) -> List[str]: # Renamed
    return [
        "AAAA", "CCCC", "GGGG" # Ensure these are length default_kmer_length_kdb
    ][:3] # Take first 3, ensure they match length

@pytest.fixture
def sample_kmers_bytes_kdb(sample_kmers_str_kdb: List[str]) -> List[Kmer]: # Renamed
    return [k.encode('utf-8') for k in sample_kmers_str_kdb]


@pytest.fixture
def pickled_kdb_path( # Renamed
    tmp_path: pathlib.Path, 
    request: pytest.FixtureRequest, 
    default_kmer_length_kdb: int, 
    strain_names_fixture_kdb: List[str], 
    sample_kmers_str_kdb: List[str], 
    sample_kmers_bytes_kdb: List[Kmer]
) -> pathlib.Path:
    params = getattr(request, "param", {})
    db_file = tmp_path / f"test_kdb_{request.node.name}.pkl"
    
    df_to_pickle: pd.DataFrame

    kmer_type = params.get("kmer_type", "str") # "str" or "bytes"
    kmers_for_df: List[Union[str, bytes]]
    if params.get("custom_kmers"):
        kmers_for_df = params["custom_kmers"]
    elif kmer_type == "str":
        kmers_for_df = sample_kmers_str_kdb
    else: # bytes
        kmers_for_df = sample_kmers_bytes_kdb
    
    counts_data = params.get("custom_counts")
    strains = params.get("custom_strains", strain_names_fixture_kdb)

    if params.get("empty_df", False):
        df_to_pickle = create_dummy_dataframe_for_kdb([], [])
    elif params.get("no_kmers", False): # No k-mers, but strains exist
        df_to_pickle = create_dummy_dataframe_for_kdb([], strains)
    elif params.get("no_strains", False): # K-mers exist, but no strains
        df_to_pickle = create_dummy_dataframe_for_kdb(kmers_for_df, [])
    elif params.get("inconsistent_kmer_len", False):
        # Create kmers with inconsistent lengths based on kmer_type
        if kmer_type == "str":
            kmers_for_df = [sample_kmers_str_kdb[0], sample_kmers_str_kdb[1][:default_kmer_length_kdb-1]]
        else: # bytes
            kmers_for_df = [sample_kmers_bytes_kdb[0], sample_kmers_bytes_kdb[1][:default_kmer_length_kdb-1]]
        df_to_pickle = create_dummy_dataframe_for_kdb(kmers_for_df, strains)
    elif params.get("unsupported_kmer_type", False):
        kmers_for_df = [123, 456] # type: ignore
        df_to_pickle = create_dummy_dataframe_for_kdb(kmers_for_df, strains)
    elif params.get("non_numeric_counts", False):
        counts_data = np.array([["str_val1", "str_val2"], ["str_val3", "str_val4"]], dtype=object) # type: ignore
        df_to_pickle = create_dummy_dataframe_for_kdb(kmers_for_df[:2], strains, counts_data) # Match dimensions
    elif params.get("non_unique_kmers", False):
        kmers_for_df = [kmers_for_df[0], kmers_for_df[0]] + kmers_for_df[1:] # Duplicate first kmer
        df_to_pickle = create_dummy_dataframe_for_kdb(kmers_for_df, strains)
    else: # Valid data by default
        df_to_pickle = create_dummy_dataframe_for_kdb(kmers_for_df, strains, counts_data)
        
    with open(db_file, "wb") as f:
        pickle.dump(df_to_pickle, f)
    return db_file

# --- Tests for __init__ and _load_database ---

@pytest.mark.parametrize("pickled_kdb_path", [{"kmer_type": "str"}], indirect=True)
def test_kdb_init_success_str_kmers(
    pickled_kdb_path: pathlib.Path, 
    default_kmer_length_kdb: int, 
    strain_names_fixture_kdb: List[str], 
    sample_kmers_str_kdb: List[str]
):
    db = KmerStrainDatabase(pickled_kdb_path, expected_kmer_length=default_kmer_length_kdb)
    assert db.kmer_length == default_kmer_length_kdb
    assert db.num_strains == len(strain_names_fixture_kdb)
    assert db.num_kmers == len(sample_kmers_str_kdb)
    assert all(isinstance(k, bytes) for k in db.kmer_to_counts_map.keys())
    assert db.strain_names == strain_names_fixture_kdb

@pytest.mark.parametrize("pickled_kdb_path", [{"kmer_type": "bytes"}], indirect=True)
def test_kdb_init_success_bytes_kmers(
    pickled_kdb_path: pathlib.Path, 
    default_kmer_length_kdb: int, 
    strain_names_fixture_kdb: List[str], 
    sample_kmers_bytes_kdb: List[Kmer]
):
    db = KmerStrainDatabase(pickled_kdb_path, expected_kmer_length=default_kmer_length_kdb)
    assert db.kmer_length == default_kmer_length_kdb
    assert db.num_strains == len(strain_names_fixture_kdb)
    assert db.num_kmers == len(sample_kmers_bytes_kdb)
    assert all(isinstance(k, bytes) for k in db.kmer_to_counts_map.keys())

@pytest.mark.parametrize("pickled_kdb_path", [{"kmer_type": "str"}], indirect=True)
def test_kdb_init_kmer_length_inferred(pickled_kdb_path: pathlib.Path, default_kmer_length_kdb: int, capsys: pytest.CaptureFixture):
    db = KmerStrainDatabase(pickled_kdb_path, expected_kmer_length=None) # Infer length
    assert db.kmer_length == default_kmer_length_kdb
    captured = capsys.readouterr()
    assert f"K-mer length inferred from first k-mer: {default_kmer_length_kdb}" in captured.out

@pytest.mark.parametrize("pickled_kdb_path", [{"kmer_type": "str"}], indirect=True)
def test_kdb_init_kmer_length_mismatch_error(pickled_kdb_path: pathlib.Path, default_kmer_length_kdb: int):
    with pytest.raises(ValueError, match="does not match length of first k-mer"):
        KmerStrainDatabase(pickled_kdb_path, expected_kmer_length=default_kmer_length_kdb + 1)

def test_kdb_init_file_not_found_error():
    with pytest.raises(FileNotFoundError, match="Database file not found or is not a file:"):
        KmerStrainDatabase("nonexistent_kdb.pkl")

@pytest.mark.parametrize("error_type", [EOFError, pickle.UnpicklingError])
@patch("pandas.read_pickle")
def test_kdb_init_pickle_read_error(mock_read_pickle: MagicMock, error_type: Exception, tmp_path: pathlib.Path):
    mock_read_pickle.side_effect = error_type
    db_file = tmp_path / "bad_pickle_kdb.pkl"
    db_file.touch()
    with pytest.raises(RuntimeError, match="Could not read or unpickle database file"):
        KmerStrainDatabase(db_file)

@patch("pandas.read_pickle")
def test_kdb_init_pickle_not_dataframe_error(mock_read_pickle: MagicMock, tmp_path: pathlib.Path):
    mock_read_pickle.return_value = "this is not a dataframe"
    db_file = tmp_path / "not_df_kdb.pkl"
    db_file.touch()
    with pytest.raises(RuntimeError, match="Data loaded from .* is not a pandas DataFrame"):
        KmerStrainDatabase(db_file)

@pytest.mark.parametrize("pickled_kdb_path", [{"empty_df": True}], indirect=True)
def test_kdb_init_empty_dataframe_error(pickled_kdb_path: pathlib.Path):
    with pytest.raises(ValueError, match="Loaded database is empty"):
        KmerStrainDatabase(pickled_kdb_path)

@pytest.mark.parametrize("pickled_kdb_path", [{"no_kmers": True}], indirect=True)
def test_kdb_init_no_kmers_error(pickled_kdb_path: pathlib.Path):
    with pytest.raises(ValueError, match="Database contains no k-mers"):
        KmerStrainDatabase(pickled_kdb_path)

@pytest.mark.parametrize("pickled_kdb_path", [{"no_strains": True}], indirect=True)
def test_kdb_init_no_strains_error(pickled_kdb_path: pathlib.Path):
    with pytest.raises(ValueError, match="Database contains no strain information"):
        KmerStrainDatabase(pickled_kdb_path)

@pytest.mark.parametrize("pickled_kdb_path", [{"inconsistent_kmer_len": True}], indirect=True)
def test_kdb_init_inconsistent_kmer_length_error(pickled_kdb_path: pathlib.Path, default_kmer_length_kdb: int):
    with pytest.raises(ValueError, match=f"Inconsistent k-mer string length at index 1. Expected {default_kmer_length_kdb}"):
        KmerStrainDatabase(pickled_kdb_path, expected_kmer_length=None) # Infer length

@pytest.mark.parametrize("pickled_kdb_path", [{"unsupported_kmer_type": True}], indirect=True)
def test_kdb_init_unsupported_kmer_type_error(pickled_kdb_path: pathlib.Path):
    with pytest.raises(TypeError, match="Unsupported k-mer type in index: <class 'int'>"):
        KmerStrainDatabase(pickled_kdb_path)

@pytest.mark.parametrize("pickled_kdb_path", [{"custom_kmers": ["AA", "ÄA"], "kmer_type":"str"}], indirect=True)
def test_kdb_init_inconsistent_byte_len_after_encoding_error(pickled_kdb_path: pathlib.Path, capsys):
    # "AA" is len 2. "ÄA" is len 2 as str, but 3 bytes in utf-8 (\xc3\x84A)
    # The code infers kmer_length=2 from "AA". Then tries to encode "ÄA".
    # The check `if len(kmer_bytes) != self.kmer_length:` inside _load_database will fail.
    # This was changed to a ValueError in the refactored code.
    with pytest.raises(ValueError, match="Post-encoding/cast k-mer byte length validation failed"):
         KmerStrainDatabase(pickled_kdb_path, expected_kmer_length=2) # Expect string length 2

@pytest.mark.parametrize("pickled_kdb_path", [{"non_numeric_counts": True}], indirect=True)
def test_kdb_init_non_numeric_counts_error(pickled_kdb_path: pathlib.Path):
    with pytest.raises(TypeError, match="Could not convert database values to count matrix"):
        KmerStrainDatabase(pickled_kdb_path)

@pytest.mark.parametrize("pickled_kdb_path", [{"non_unique_kmers": True}], indirect=True)
def test_kdb_init_non_unique_kmers_warning(pickled_kdb_path: pathlib.Path, capsys: pytest.CaptureFixture):
    KmerStrainDatabase(pickled_kdb_path) # Should load, using last occurrence for duplicates
    captured = capsys.readouterr()
    assert "Warning: K-mer index in .* is not unique." in captured.out


# --- Tests for get_strain_counts_for_kmer ---
@pytest.mark.parametrize("pickled_kdb_path", [{"kmer_type": "str", "valid_data": True}], indirect=True)
def test_kdb_get_strain_counts_for_kmer(
    pickled_kdb_path: pathlib.Path, 
    default_kmer_length_kdb: int, 
    strain_names_fixture_kdb: List[str], 
    sample_kmers_str_kdb: List[str]
):
    db = KmerStrainDatabase(pickled_kdb_path, expected_kmer_length=default_kmer_length_kdb)
    
    known_kmer_bytes = sample_kmers_str_kdb[0].encode('utf-8')
    counts = db.get_strain_counts_for_kmer(known_kmer_bytes)
    assert counts is not None
    assert isinstance(counts, np.ndarray)
    assert counts.dtype == np.uint8
    assert len(counts) == len(strain_names_fixture_kdb)

    unknown_kmer = ("Z" * default_kmer_length_kdb).encode('utf-8')
    assert db.get_strain_counts_for_kmer(unknown_kmer) is None

    wrong_length_kmer = (sample_kmers_str_kdb[0][:-1]).encode('utf-8') # Shorter
    assert db.get_strain_counts_for_kmer(wrong_length_kmer) is None


# --- Tests for __len__ ---
@pytest.mark.parametrize("pickled_kdb_path", [{"kmer_type": "str", "valid_data": True}], indirect=True)
def test_kdb_len_method(pickled_kdb_path: pathlib.Path, sample_kmers_str_kdb: List[str]):
    db = KmerStrainDatabase(pickled_kdb_path)
    assert len(db) == len(sample_kmers_str_kdb)

@pytest.mark.parametrize("pickled_kdb_path", [{"non_unique_kmers": True, "custom_kmers": ["AAAA", "AAAA", "CCCC", "GGGG"], "kmer_type":"str"}], indirect=True)
def test_kdb_len_method_non_unique_kmers(pickled_kdb_path: pathlib.Path, default_kmer_length_kdb: int):
    # Custom kmers: "AAAA", "AAAA", "CCCC", "GGGG" (all length 4, matching default_kmer_length_kdb)
    # Unique k-mers: "AAAA", "CCCC", "GGGG" -> 3 unique
    # Must pass expected_kmer_length that matches the custom_kmers.
    db = KmerStrainDatabase(pickled_kdb_path, expected_kmer_length=4)
    assert len(db) == 3 


# --- Tests for __contains__ ---
@pytest.mark.parametrize("pickled_kdb_path", [{"kmer_type": "str", "valid_data": True}], indirect=True)
def test_kdb_contains_method(pickled_kdb_path: pathlib.Path, sample_kmers_str_kdb: List[str], default_kmer_length_kdb: int):
    db = KmerStrainDatabase(pickled_kdb_path)
    
    known_kmer_bytes = sample_kmers_str_kdb[0].encode('utf-8')
    assert known_kmer_bytes in db

    unknown_kmer = ("Z" * default_kmer_length_kdb).encode('utf-8')
    assert unknown_kmer not in db
    
    # Test with string - Kmer type alias is bytes, so this should be False
    known_kmer_as_str = sample_kmers_str_kdb[0]
    assert known_kmer_as_str not in db # type: ignore # Test behavior with incorrect type
```
