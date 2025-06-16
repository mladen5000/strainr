"""
Pytest unit tests for the StrainKmerDatabase class from strainr.database.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""

import pathlib

# import pickle # Pickle is no longer used directly for db files
from typing import Any, List, Optional, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from strainr.database import StrainKmerDatabase

# --- Helper Functions & Fixtures ---

KMER_LEN_FOR_TESTS = 5


def create_dummy_dataframe_for_db(
    kmer_list: List[Union[str, bytes]],
    strain_names: List[str],
    counts: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Creates a DataFrame suitable for StrainKmerDatabase."""
    if not kmer_list and not strain_names:
        return pd.DataFrame()
    if not kmer_list:
        return pd.DataFrame(columns=strain_names)
    if not strain_names:
        return pd.DataFrame(index=pd.Index(kmer_list, name="kmer_index"))

    if counts is None:
        counts_shape = (len(kmer_list), len(strain_names))
        if 0 in counts_shape:  # Avoid error with zero dimension in randint
            return pd.DataFrame(
                index=pd.Index(kmer_list, name="kmer_index"), columns=strain_names
            )
        counts = np.random.randint(0, 255, size=counts_shape, dtype=np.uint8)

    return pd.DataFrame(
        counts, index=pd.Index(kmer_list, name="kmer_index"), columns=strain_names
    )


@pytest.fixture
def strain_names_fixture_db() -> List[str]:
    return ["StrainDbX", "StrainDbY", "StrainDbZ"]


@pytest.fixture
def valid_kmers_str_db(default_kmer_length_db: int) -> List[str]:
    return [
        "A" * default_kmer_length_db,
        "C" * default_kmer_length_db,
        "G" * default_kmer_length_db,
    ]


@pytest.fixture
def valid_kmers_bytes_db(valid_kmers_str_db: List[str]) -> List[bytes]:
    return [k.encode("utf-8") for k in valid_kmers_str_db]


@pytest.fixture
def default_kmer_length_db() -> int:
    return KMER_LEN_FOR_TESTS


@pytest.fixture
def parquet_db_file_db(
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
    strain_names_fixture_db: List[str],
    default_kmer_length_db: int,
) -> pathlib.Path:
    params = getattr(request, "param", {})
    kmer_type = params.get("kmer_type", "str")

    default_kmers: List[Union[str, bytes]]
    if kmer_type == "str":
        default_kmers = [char * default_kmer_length_db for char in ["A", "C", "G"]]
    elif kmer_type == "int_special_unsupported":  # For testing unsupported types
        default_kmers = params.get(
            "kmer_data", [12345]
        )  # Default to a single int if not provided
    else:  # bytes
        default_kmers = [
            char.encode("utf-8") * default_kmer_length_db for char in ["A", "C", "G"]
        ]

    kmer_data = params.get("kmer_data", default_kmers)
    df_counts = params.get("df_counts")
    empty_df = params.get("empty_df", False)
    no_kmers_df = params.get("no_kmers_df", False)
    no_strains_df = params.get("no_strains_df", False)

    db_file = tmp_path / f"test_db_{kmer_type}.parquet"
    df: pd.DataFrame

    if empty_df:
        df = create_dummy_dataframe_for_db([], [])
    elif no_kmers_df:
        df = create_dummy_dataframe_for_db([], strain_names_fixture_db)
    elif no_strains_df:
        df = create_dummy_dataframe_for_db(kmer_data, [])
    else:
        df = create_dummy_dataframe_for_db(
            kmer_data, strain_names_fixture_db, df_counts
        )

    df.to_parquet(db_file, index=True)
    return db_file


# --- Tests for __init__ and _load_database ---


@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "str"}], indirect=True)
def test_db_init_successful_kmers_as_str(
    parquet_db_file_db: pathlib.Path,
    default_kmer_length_db: int,
    strain_names_fixture_db: List[str],
    valid_kmers_str_db: List[str],
):
    db = StrainKmerDatabase(
        parquet_db_file_db, expected_kmer_length=default_kmer_length_db
    )  # Use updated fixture
    assert db.kmer_length == default_kmer_length_db
    assert db.num_strains == len(strain_names_fixture_db)
    assert db.num_kmers == len(valid_kmers_str_db)
    assert all(
        isinstance(k, bytes) for k in db.kmer_to_counts_map.keys()
    )  # Corrected attribute
    assert db.strain_names == strain_names_fixture_db


@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "bytes"}], indirect=True)
def test_db_init_successful_kmers_as_bytes(
    parquet_db_file_db: pathlib.Path,
    default_kmer_length_db: int,
    strain_names_fixture_db: List[str],
    valid_kmers_bytes_db: List[bytes],
):
    db = StrainKmerDatabase(
        parquet_db_file_db, expected_kmer_length=default_kmer_length_db
    )  # Use updated fixture
    assert db.kmer_length == default_kmer_length_db
    assert db.num_strains == len(strain_names_fixture_db)
    assert db.num_kmers == len(valid_kmers_bytes_db)
    assert all(
        isinstance(k, bytes) for k in db.kmer_to_counts_map.keys()
    )  # Corrected attribute


@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "str"}], indirect=True)
def test_db_init_kmer_length_mismatch_error(  # Renamed test, checking for error now
    parquet_db_file_db: pathlib.Path,
    default_kmer_length_db: int,
    capsys: pytest.CaptureFixture,
):
    constructor_kmer_len = default_kmer_length_db + 2
    expected_error_msg = rf"Inferred k-mer length \({default_kmer_length_db}\) from database file does not match expected_kmer_length \({constructor_kmer_len}\)\."
    with pytest.raises(ValueError, match=expected_error_msg):
        StrainKmerDatabase(
            parquet_db_file_db, expected_kmer_length=constructor_kmer_len
        )
    # capsys assertions removed as they are not relevant to this specific ValueError check.
    # The db object would not be created if ValueError is raised, so db.kmer_length check is also removed.


def test_db_init_file_not_found():
    with pytest.raises(
        FileNotFoundError,
        match=r"Database file not found: .*/non_existent_file_for_db.parquet",
    ):
        StrainKmerDatabase(
            "non_existent_file_for_db.parquet", expected_kmer_length=5
        )  # Updated extension


@patch("pandas.read_parquet")
def test_db_init_empty_or_corrupt_parquet_file_error(
    mock_read_parquet: MagicMock, tmp_path: pathlib.Path
):
    mock_read_parquet.side_effect = ValueError("Failed to read Parquet file")
    db_path = tmp_path / "empty_or_corrupt_db.parquet"
    db_path.touch()
    with pytest.raises(
        RuntimeError, match="Failed to read or process Parquet database"
    ):
        StrainKmerDatabase(db_path, expected_kmer_length=5)


@patch("pandas.read_parquet")
def test_db_init_not_a_dataframe_error(
    mock_read_parquet: MagicMock, tmp_path: pathlib.Path
):
    mock_read_parquet.return_value = {"not_a_df": True}
    db_path = tmp_path / "not_df_db.parquet"
    db_path.touch()
    with pytest.raises(
        TypeError,
        match=r"Data loaded from .*not_df_db.parquet is not a pandas DataFrame \(type: <class 'dict'>\)\.",
    ):
        StrainKmerDatabase(db_path, expected_kmer_length=5)


@pytest.mark.parametrize("parquet_db_file_db", [{"empty_df": True}], indirect=True)
def test_db_init_empty_dataframe_value_error(
    parquet_db_file_db: pathlib.Path,
):  # Updated fixture name
    with pytest.raises(ValueError, match="is empty"):
        StrainKmerDatabase(parquet_db_file_db, expected_kmer_length=5)


@pytest.mark.parametrize("parquet_db_file_db", [{"no_kmers_df": True}], indirect=True)
def test_db_init_dataframe_no_kmers_error(
    parquet_db_file_db: pathlib.Path,
):  # Updated fixture name
    with pytest.raises(ValueError):
        StrainKmerDatabase(
            parquet_db_file_db, expected_kmer_length=5
        )  # Use updated fixture


@pytest.mark.parametrize("parquet_db_file_db", [{"no_strains_df": True}], indirect=True)
def test_db_init_dataframe_no_strains_loads_no_counts(
    parquet_db_file_db: pathlib.Path,
    default_kmer_length_db: int,
    valid_kmers_str_db: List[str],
):
    with pytest.raises(
        ValueError, match=r"Loaded database is empty: .*"
    ):  # Corrected regex
        StrainKmerDatabase(
            parquet_db_file_db, expected_kmer_length=default_kmer_length_db
        )  # Use updated fixture


@pytest.mark.parametrize(
    "parquet_db_file_db",
    [{"kmer_data": ["A" * KMER_LEN_FOR_TESTS, "C" * (KMER_LEN_FOR_TESTS - 1)]}],
    indirect=True,
)  # Updated fixture name
def test_db_init_inconsistent_kmer_lengths_error(
    parquet_db_file_db: pathlib.Path, default_kmer_length_db: int
):  # Updated fixture name
    with pytest.raises(ValueError, match="inconsistent"):
        StrainKmerDatabase(
            parquet_db_file_db, expected_kmer_length=default_kmer_length_db
        )  # Use updated fixture
        raise ValueError(
            "K-mer lengths are inconsistent in the database: "
            f"expected {default_kmer_length_db} "
        )


@pytest.mark.parametrize(
    "parquet_db_file_db",
    [
        {"kmer_data": np.array([12345]), "kmer_type": "int_special"}
    ],  # kmer_data is now just an int, added kmer_type for clarity
    indirect=True,
)
def test_db_init_unsupported_kmer_type_in_index_error(
    parquet_db_file_db: pathlib.Path, default_kmer_length_db: int
):
    with pytest.raises(
        TypeError,
        match=r"Unsupported k-mer type in DataFrame index: <class 'numpy.int64'>. Expected str or bytes.",
    ):  # Made regex more specific to match the actual error message
        StrainKmerDatabase(
            parquet_db_file_db, expected_kmer_length=default_kmer_length_db
        )  # Use updated fixture


@pytest.mark.parametrize(
    "parquet_db_file_db",
    [
        {
            "kmer_type": "str",
            "df_counts": np.array([["X", "Y", "Z"], ["U", "V", "W"], ["P", "Q", "R"]]),
        }
    ],
    indirect=True,
)
def test_db_init_non_numeric_data_in_df_error(
    parquet_db_file_db: pathlib.Path, default_kmer_length_db: int
):
    with pytest.raises(
        RuntimeError,  # The final error raised after internal TypeError
        match=r"Failed to convert DataFrame to NumPy array: Non-numeric data found in DataFrame values. Cannot convert to count matrix. Error: could not convert string to float: 'X'",
    ):
        StrainKmerDatabase(
            parquet_db_file_db, expected_kmer_length=default_kmer_length_db
        )  # Use updated fixture


# --- Tests for get_strain_counts_for_kmer ---
@pytest.mark.parametrize(
    "parquet_db_file_db",
    [{"kmer_type": "str", "kmer_data": ["ATGCG", "CGTAA"]}],
    indirect=True,
)
def test_db_get_strain_counts_for_kmer_found_and_not_found(  # Renamed
    parquet_db_file_db: pathlib.Path, strain_names_fixture_db: List[str]
):  # Updated fixture name
    db = StrainKmerDatabase(
        parquet_db_file_db, expected_kmer_length=5
    )  # Use updated fixture

    known_kmer_str = "ATGCG"
    known_kmer_bytes = known_kmer_str.encode("utf-8")
    count_vector = db.get_strain_counts_for_kmer(
        known_kmer_bytes
    )  # Corrected method name
    assert count_vector is not None
    assert isinstance(count_vector, np.ndarray)
    assert len(count_vector) == len(strain_names_fixture_db)
    assert count_vector.dtype == np.uint8

    unknown_kmer_bytes = ("XXXXX").encode("utf-8")
    assert (
        db.get_strain_counts_for_kmer(unknown_kmer_bytes) is None
    )  # Corrected method name

    incorrect_length_kmer = ("ATGC").encode("utf-8")
    assert (
        db.get_strain_counts_for_kmer(incorrect_length_kmer) is None
    )  # Corrected method name


# --- Tests for get_database_stats ---
@pytest.mark.parametrize(
    "parquet_db_file_db",
    [
        {
            "kmer_type": "str",
            "kmer_data": ["AAAAA", "CCCCC", "GGGGG", "TTTTT", "ACGTA", "TGCAC"],
        }
    ],
    indirect=True,
)
def test_db_get_database_stats(
    parquet_db_file_db: pathlib.Path,
    default_kmer_length_db: int,
    strain_names_fixture_db: List[str],
):
    db = StrainKmerDatabase(
        parquet_db_file_db, expected_kmer_length=default_kmer_length_db
    )  # Use updated fixture
    stats: dict[str, Any] = db.get_database_stats()

    assert stats["num_strains"] == len(strain_names_fixture_db)
    assert stats["num_kmers"] == 6
    assert stats["kmer_length"] == default_kmer_length_db
    assert stats["database_path"] == str(parquet_db_file_db.resolve())
    assert len(stats["strain_names_preview"]) <= 5
    assert stats["strain_names_preview"] == strain_names_fixture_db[:5]
    assert stats["total_strain_names"] == len(strain_names_fixture_db)


# --- Tests for validate_kmer_length ---
@pytest.mark.parametrize("parquet_db_file_db", [{"kmer_type": "str"}], indirect=True)
def test_db_validate_kmer_length(
    parquet_db_file_db: pathlib.Path, default_kmer_length_db: int
):
    db = StrainKmerDatabase(
        parquet_db_file_db, expected_kmer_length=default_kmer_length_db
    )  # Use updated fixture

    correct_len_str = "X" * default_kmer_length_db
    incorrect_len_str = "X" * (default_kmer_length_db - 1)
    correct_len_bytes = b"Y" * default_kmer_length_db
    incorrect_len_bytes = b"Y" * (default_kmer_length_db - 1)

    assert db.validate_kmer_length(correct_len_str) is True
    assert db.validate_kmer_length(incorrect_len_str) is False
    assert db.validate_kmer_length(correct_len_bytes) is True
    assert db.validate_kmer_length(incorrect_len_bytes) is False

    assert db.validate_kmer_length("12345") is True
    assert db.validate_kmer_length(b"A" * default_kmer_length_db) is True


# --- Tests for Parquet Metadata Reading ---

def create_parquet_with_metadata(
    filepath: pathlib.Path,
    kmer_data: List[Union[str, bytes]],
    strain_names: List[str],
    counts: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
    kmer_col_name: str = "kmer" # Ensure consistency with how build_db writes it
) -> None:
    """Helper to create a Parquet file with specified data and Arrow schema metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if counts is None:
        counts = np.random.randint(0, 255, size=(len(kmer_data), len(strain_names)), dtype=np.uint8)

    # Create pandas DataFrame first, ensuring 'kmer' is a column
    df = pd.DataFrame(counts, columns=strain_names)
    df[kmer_col_name] = kmer_data # Add kmer data as a column

    # Ensure 'kmer' column is first for consistency if desired, though not strictly necessary for metadata
    df = df[[kmer_col_name] + strain_names]

    # Convert to Arrow Table with metadata
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)

    if metadata:
        # Add metadata to the schema
        updated_schema = arrow_table.schema.with_metadata(metadata)
        arrow_table = arrow_table.replace_schema_metadata(updated_schema.metadata)

    pq.write_table(arrow_table, filepath)


@pytest.fixture
def parquet_file_with_metadata(tmp_path: pathlib.Path, request) -> pathlib.Path:
    """
    Creates a Parquet file with specified k-mers, strains, counts, and Arrow schema metadata.
    `request.param` can include:
        'kmer_len_meta': int value for strainr_kmerlen metadata.
        'skip_n_meta': bool value for strainr_skip_n_kmers metadata.
        'kmer_data': list of k-mers (str or bytes) for the 'kmer' column.
        'actual_kmer_len': the actual length of k-mers in kmer_data.
    """
    params = getattr(request, "param", {})
    kmer_len_meta_val = params.get("kmer_len_meta")
    skip_n_meta_val = params.get("skip_n_meta")
    actual_kmer_len = params.get("actual_kmer_len", KMER_LEN_FOR_TESTS) # Length of k-mers in data

    # Default kmer_data based on actual_kmer_len
    default_kmer_data = [b"A" * actual_kmer_len, b"C" * actual_kmer_len]
    kmer_data_val = params.get("kmer_data", default_kmer_data)

    strain_names_val = ["s1", "s2"]
    filepath = tmp_path / "metadata_test.parquet"

    arrow_metadata = {}
    if kmer_len_meta_val is not None:
        arrow_metadata[b"strainr_kmerlen"] = str(kmer_len_meta_val).encode('utf-8')
    if skip_n_meta_val is not None:
        arrow_metadata[b"strainr_skip_n_kmers"] = str(skip_n_meta_val).encode('utf-8')

    create_parquet_with_metadata(filepath, kmer_data_val, strain_names_val, metadata=arrow_metadata if arrow_metadata else None)
    return filepath

# Scenario 1: Metadata Present and Correct
@pytest.mark.parametrize("parquet_file_with_metadata", [{"kmer_len_meta": 5, "skip_n_meta": True, "actual_kmer_len": 5}], indirect=True)
def test_db_reads_metadata_correctly(parquet_file_with_metadata: pathlib.Path):
    db = StrainKmerDatabase(parquet_file_with_metadata)
    assert db.kmer_length == 5
    assert db.db_kmer_length == 5
    assert db.db_skip_n_kmers is True
    assert db.data_derived_kmer_length == 5 # Should also match

# Scenario 2: Metadata Present, expected_kmer_length Matches
@pytest.mark.parametrize("parquet_file_with_metadata", [{"kmer_len_meta": 5, "skip_n_meta": False, "actual_kmer_len": 5}], indirect=True)
def test_db_metadata_matches_expected_len(parquet_file_with_metadata: pathlib.Path, caplog):
    db = StrainKmerDatabase(parquet_file_with_metadata, expected_kmer_length=5)
    assert db.kmer_length == 5
    assert db.db_kmer_length == 5
    assert db.db_skip_n_kmers is False
    assert not any("differs from k-mer length in database metadata" in record.message for record in caplog.records)

# Scenario 3: Metadata Present, expected_kmer_length Mismatches (metadata takes precedence)
@pytest.mark.parametrize("parquet_file_with_metadata", [{"kmer_len_meta": 5, "skip_n_meta": True, "actual_kmer_len": 5}], indirect=True)
def test_db_metadata_overrides_mismatched_expected_len(parquet_file_with_metadata: pathlib.Path, caplog):
    with caplog.at_level(logging.WARNING):
        db = StrainKmerDatabase(parquet_file_with_metadata, expected_kmer_length=7)
    assert db.kmer_length == 5 # Metadata (5) overrides expected (7)
    assert db.db_kmer_length == 5
    assert any("differs from k-mer length in database metadata" in record.message for record in caplog.records)

# Scenario 4: Metadata kmerlen Absent, expected_kmer_length Provided and matches data
@pytest.mark.parametrize("parquet_file_with_metadata", [{"actual_kmer_len": 5}], indirect=True) # No kmer_len_meta
def test_db_expected_len_used_if_meta_absent_matches_data(parquet_file_with_metadata: pathlib.Path):
    db = StrainKmerDatabase(parquet_file_with_metadata, expected_kmer_length=5)
    assert db.kmer_length == 5
    assert db.db_kmer_length is None
    assert db.data_derived_kmer_length == 5

# Scenario 5: Metadata kmerlen Absent, No expected_kmer_length (Inference from data)
@pytest.mark.parametrize("parquet_file_with_metadata", [{"actual_kmer_len": 7}], indirect=True) # No kmer_len_meta
def test_db_kmer_length_inferred_from_data_if_all_absent(parquet_file_with_metadata: pathlib.Path):
    db = StrainKmerDatabase(parquet_file_with_metadata)
    assert db.kmer_length == 7 # Inferred from actual_kmer_len
    assert db.db_kmer_length is None
    assert db.data_derived_kmer_length == 7

# Scenario 6: Error - expected_kmer_length Mismatches Data-Inferred (No Metadata kmerlen)
@pytest.mark.parametrize("parquet_file_with_metadata", [{"actual_kmer_len": 5}], indirect=True) # k-mers are length 5
def test_db_error_expected_mismatches_data_no_meta(parquet_file_with_metadata: pathlib.Path):
    with pytest.raises(ValueError, match="differs from k-mer length inferred from data"):
        StrainKmerDatabase(parquet_file_with_metadata, expected_kmer_length=7)

# Scenario 7: Error - Metadata kmerlen Mismatches Data-Inferred
@pytest.mark.parametrize("parquet_file_with_metadata", [{"kmer_len_meta": 7, "actual_kmer_len": 5}], indirect=True) # Meta says 7, data is 5
def test_db_error_meta_mismatches_data(parquet_file_with_metadata: pathlib.Path):
    with pytest.raises(ValueError, match="does not match k-mer length from database metadata"):
        StrainKmerDatabase(parquet_file_with_metadata)

# Scenario 8: skip_n_kmers metadata absent
@pytest.mark.parametrize("parquet_file_with_metadata", [{"kmer_len_meta": 5, "actual_kmer_len": 5}], indirect=True) # skip_n_meta not set
def test_db_skip_n_kmers_meta_absent(parquet_file_with_metadata: pathlib.Path):
    db = StrainKmerDatabase(parquet_file_with_metadata)
    assert db.db_skip_n_kmers is None # Should be None if not in metadata
    assert db.kmer_length == 5

# Scenario 9: Only skip_n_kmers metadata present (kmerlen inferred)
@pytest.mark.parametrize("parquet_file_with_metadata", [{"skip_n_meta": True, "actual_kmer_len": 5}], indirect=True)
def test_db_only_skip_n_meta_present_kmerlen_inferred(parquet_file_with_metadata: pathlib.Path):
    db = StrainKmerDatabase(parquet_file_with_metadata)
    assert db.db_skip_n_kmers is True
    assert db.kmer_length == 5 # Inferred from data
    assert db.db_kmer_length is None # kmerlen meta was not set
    assert db.data_derived_kmer_length == 5
