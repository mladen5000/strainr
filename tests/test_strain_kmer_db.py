"""
Pytest unit tests for the consolidated StrainKmerDatabase class from src.strainr.database.
These tests assume the file is in the root directory, and 'src' is a subdirectory.
"""

import pathlib

# import pickle # Pickle is no longer used directly for db files
from typing import Any, List, Optional, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import the consolidated class
from strainr.database import StrainKmerDatabase

# --- Helper Functions & Fixtures (adapted from previous test_kmer_database.py) ---
# Define Kmer type alias locally as it's no longer exported by the database module
Kmer = Union[str, bytes]


KMER_LEN_FOR_TESTS_SKDB = 4  # Using a distinct constant name
Kmer = Union[str, bytes]  # Define Kmer type for clarity


def create_dummy_dataframe_for_skdb(
    kmer_list: List[Union[str, bytes]],
    strain_names: List[str],
    counts: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Creates a DataFrame suitable for StrainKmerDatabase."""  # Corrected class name in docstring
    if not kmer_list and not strain_names:
        return pd.DataFrame()
    if not kmer_list:
        return pd.DataFrame(columns=strain_names)
    if not strain_names:
        return pd.DataFrame(index=pd.Index(kmer_list, name="kmer_idx"))

    if counts is None:
        counts_shape = (len(kmer_list), len(strain_names))
        if 0 in counts_shape:
            return pd.DataFrame(
                index=pd.Index(kmer_list, name="kmer_idx"), columns=strain_names
            )
        counts = np.random.randint(0, 256, size=counts_shape, dtype=np.uint8)

    return pd.DataFrame(
        counts, index=pd.Index(kmer_list, name="kmer_idx"), columns=strain_names
    )


@pytest.fixture
def strain_names_fixture_skdb() -> List[str]:
    return ["StrainConsolidated1", "StrainConsolidated2"]


@pytest.fixture
def default_kmer_length_skdb() -> int:
    return KMER_LEN_FOR_TESTS_SKDB


@pytest.fixture
def sample_kmers_str_skdb(default_kmer_length_skdb: int) -> List[str]:
    return [
        "AAAA",
        "CCCC",
        "GGGG",
        "TTTT",  # Length 4
    ][:3]


@pytest.fixture
def sample_kmers_bytes_skdb(
    sample_kmers_str_skdb: List[str],
) -> List[Kmer]:  # Kmer type hint is now locally defined
    return [k.encode("utf-8") for k in sample_kmers_str_skdb]


@pytest.fixture
def parquet_skdb_path(
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
    default_kmer_length_skdb: int,
    strain_names_fixture_skdb: List[str],
    sample_kmers_str_skdb: List[Kmer],
    sample_kmers_bytes_skdb: List[Kmer],
) -> pathlib.Path:
    params = getattr(request, "param", {})
    db_file = tmp_path / f"test_strain_kmer_db_{request.node.name}.parquet"

    df_to_save: pd.DataFrame
    kmer_type = params.get("kmer_type", "str")

    kmers_for_df: List[Union[str, bytes]]
    if params.get("custom_kmers"):
        kmers_for_df = params["custom_kmers"]
    elif kmer_type == "str":
        kmers_for_df = sample_kmers_str_skdb
    elif kmer_type == "int_special_unsupported":  # For testing unsupported types
        kmers_for_df = params["kmer_data"]  # e.g. [12345]
    else:  # bytes
        kmers_for_df = sample_kmers_bytes_skdb

    counts_data = params.get("custom_counts")
    strains = params.get("custom_strains", strain_names_fixture_skdb)

    if params.get("empty_df", False):
        df_to_save = create_dummy_dataframe_for_skdb([], [])
    elif params.get(
        "no_kmers", False
    ):  # This will create an empty DataFrame if kmer_list is empty
        df_to_save = create_dummy_dataframe_for_skdb([], strains)
    elif params.get(
        "no_strains", False
    ):  # This will create a DataFrame with an index but no columns
        df_to_save = create_dummy_dataframe_for_skdb(kmers_for_df, [])
    elif params.get("inconsistent_kmer_len", False):
        # This case creates a list with one kmer of correct length and one of incorrect length
        if kmer_type == "str":
            kmers_for_df = [
                sample_kmers_str_skdb[0],  # Correct length
                sample_kmers_str_skdb[1][: default_kmer_length_skdb - 1],  # Incorrect
            ]
        else:  # bytes
            kmers_for_df = [
                sample_kmers_bytes_skdb[0],  # Correct length
                sample_kmers_bytes_skdb[1][: default_kmer_length_skdb - 1],  # Incorrect
            ]
        df_to_save = create_dummy_dataframe_for_skdb(kmers_for_df, strains)
    elif params.get("unsupported_kmer_type", False):
        # kmers_for_df is already set (e.g. [12345] by "int_special_unsupported" kmer_type)
        df_to_save = create_dummy_dataframe_for_skdb(kmers_for_df, strains)
    elif params.get("non_numeric_counts", False):
        target_kmer_list: list[bytes | str] | Any | Unknown = kmers_for_df
        if len(kmers_for_df) < 2 and len(kmers_for_df) > 0:
            target_kmer_list = [kmers_for_df[0], kmers_for_df[0]]
        elif not kmers_for_df:
            target_kmer_list = (
                sample_kmers_str_skdb[:2]
                if kmer_type == "str"
                else sample_kmers_bytes_skdb[:2]
            )
            if len(target_kmer_list) < 2:
                target_kmer_list = [
                    "A" * default_kmer_length_skdb,
                    "C" * default_kmer_length_skdb,
                ]
            target_kmer_list = [kmers_for_df[0], kmers_for_df[0]]
        elif not kmers_for_df:  # Ensure at least two k-mers for 2x2 non-numeric counts
            target_kmer_list = (
                sample_kmers_str_skdb[:2]
                if kmer_type == "str"
                else sample_kmers_bytes_skdb[:2]
            )
            if len(target_kmer_list) < 2:  # Fallback if sample kmers are less than 2
                target_kmer_list = [
                    "A" * default_kmer_length_skdb,
                    "C" * default_kmer_length_skdb,
                ]

        counts_data = np.array([["val1", "val2"], ["val3", "val4"]], dtype=object)
        df_to_save = create_dummy_dataframe_for_skdb(
            target_kmer_list[:2], strains, counts_data
        )
    elif params.get("non_unique_kmers", False):
        # Ensure there's at least one kmer to duplicate
        if not kmers_for_df:
            kmers_for_df = (
                sample_kmers_str_skdb if kmer_type == "str" else sample_kmers_bytes_skdb
            )
            if not kmers_for_df:  # Still no kmers, add a default one
                kmers_for_df = ["A" * default_kmer_length_skdb]

        kmers_for_df_updated = [kmers_for_df[0], kmers_for_df[0]] + (
            kmers_for_df[1:] if len(kmers_for_df) > 1 else []
        )
        df_to_save = create_dummy_dataframe_for_skdb(kmers_for_df_updated, strains)
        df_to_save = create_dummy_dataframe_for_skdb(kmers_for_df_updated, strains)
    else:
        df_to_save = create_dummy_dataframe_for_skdb(kmers_for_df, strains, counts_data)

    df_to_save.to_parquet(db_file, index=True)
    return db_file


@pytest.mark.parametrize("parquet_skdb_path", [{"kmer_type": "str"}], indirect=True)
def test_skdb_init_success_str_kmers(
    parquet_skdb_path: pathlib.Path,
    default_kmer_length_skdb: int,
    strain_names_fixture_skdb: List[str],
    sample_kmers_str_skdb: List[str],
):
    db = StrainKmerDatabase(
        parquet_skdb_path, expected_kmer_length=default_kmer_length_skdb
    )
    assert db.kmer_length == default_kmer_length_skdb
    assert db.num_strains == len(strain_names_fixture_skdb)
    assert db.num_kmers == len(sample_kmers_str_skdb)
    assert all(isinstance(k, bytes) for k in db.kmer_to_counts_map.keys())
    assert db.strain_names == strain_names_fixture_skdb


@pytest.mark.parametrize("parquet_skdb_path", [{"kmer_type": "bytes"}], indirect=True)
def test_skdb_init_success_bytes_kmers(
    parquet_skdb_path: pathlib.Path,
    default_kmer_length_skdb: int,
    strain_names_fixture_skdb: List[str],
    sample_kmers_bytes_skdb: List[Kmer],
):
    db = StrainKmerDatabase(
        parquet_skdb_path, expected_kmer_length=default_kmer_length_skdb
    )
    assert db.kmer_length == default_kmer_length_skdb
    assert db.num_strains == len(strain_names_fixture_skdb)
    assert db.num_kmers == len(sample_kmers_bytes_skdb)
    assert all(isinstance(k, bytes) for k in db.kmer_to_counts_map.keys())


@pytest.mark.parametrize("parquet_skdb_path", [{"kmer_type": "str"}], indirect=True)
def test_skdb_init_kmer_length_inferred(
    parquet_skdb_path: pathlib.Path,
    default_kmer_length_skdb: int,
    capsys: pytest.CaptureFixture,
):
    db = StrainKmerDatabase(parquet_skdb_path, expected_kmer_length=None)
    assert db.kmer_length == default_kmer_length_skdb
    captured = capsys.readouterr()
    # The specific log message about inference is not present, the main check is db.kmer_length
    # Ensure the summary print contains the kmer length, e.g.
    assert f"- K-mer length: {default_kmer_length_skdb}" in captured.out


@pytest.mark.parametrize("parquet_skdb_path", [{"kmer_type": "str"}], indirect=True)
def test_skdb_init_kmer_length_mismatch_error(
    parquet_skdb_path: pathlib.Path, default_kmer_length_skdb: int
):
    with pytest.raises(ValueError):
        StrainKmerDatabase(
            parquet_skdb_path, expected_kmer_length=default_kmer_length_skdb + 1
        )


def test_skdb_init_file_not_found_error():
    with pytest.raises(
        FileNotFoundError, match=r"Database file not found: .*/nonexistent_skdb.parquet"
    ):
        StrainKmerDatabase("nonexistent_skdb.parquet")


@patch("pandas.read_parquet")
def test_skdb_init_parquet_read_error(
    mock_read_parquet: MagicMock, tmp_path: pathlib.Path
):
    mock_read_parquet.side_effect = ValueError("Simulated Parquet read error")
    db_file = tmp_path / "bad_parquet_skdb.parquet"
    db_file.touch()
    with pytest.raises(RuntimeError):
        StrainKmerDatabase(db_file)


@pytest.mark.parametrize("parquet_skdb_path", [{"empty_df": True}], indirect=True)
def test_skdb_init_empty_dataframe_error(parquet_skdb_path: pathlib.Path):
    with pytest.raises(ValueError, match=r"Loaded database is empty: .*"):
        StrainKmerDatabase(parquet_skdb_path)


@pytest.mark.parametrize("parquet_skdb_path", [{"no_kmers": True}], indirect=True)
def test_skdb_init_no_kmers_error(parquet_skdb_path: pathlib.Path):
    with pytest.raises(
        ValueError, match=r"Loaded database is empty: .*"
    ):  # Corrected regex
        StrainKmerDatabase(parquet_skdb_path)


@pytest.mark.parametrize("parquet_skdb_path", [{"no_strains": True}], indirect=True)
def test_skdb_init_no_strains_error(parquet_skdb_path: pathlib.Path):
    with pytest.raises(
        ValueError, match=r"Loaded database is empty: .*"
    ):  # Corrected regex
        StrainKmerDatabase(parquet_skdb_path)


@pytest.mark.parametrize(
    "parquet_skdb_path",
    [{"inconsistent_kmer_len": True, "kmer_type": "str"}],
    indirect=True,
)
def test_skdb_init_inconsistent_str_kmer_length_warning(
    parquet_skdb_path: pathlib.Path,
    default_kmer_length_skdb: int,
    capsys: pytest.CaptureFixture,
):
    # This test now checks for warning and correct loading of valid k-mers
    db = StrainKmerDatabase(
        parquet_skdb_path, expected_kmer_length=None
    )  # kmer_length will be inferred as 4 from 'AAAA'
    captured = capsys.readouterr()
    # The kmer "CCC" (index 1) has length 3, expected is 4 (inferred from "AAAA")
    expected_warning_fragment = r"Warning: K-mer 'CCC' (index 1) has inconsistent length: 3. Expected 4. Skipping."
    actual_output = captured.out
    # Search for the pattern in the actual output
    assert any(
        expected_warning_fragment in line for line in actual_output.splitlines()
    ), (
        f"Expected warning fragment '{expected_warning_fragment}' not found in output: {actual_output}"
    )
    assert db.num_kmers == 1  # Only the first k-mer ("AAAA") should be loaded


@pytest.mark.parametrize(
    "parquet_skdb_path",
    [{"kmer_data": [12345], "kmer_type": "int_special_unsupported"}],
    indirect=True,
)
def test_skdb_init_unsupported_kmer_type_error(parquet_skdb_path: pathlib.Path):
    with pytest.raises(
        TypeError,
        match=r"Unsupported k-mer type in DataFrame index: <class 'numpy.int64'>. Expected str or bytes.",
    ):
        StrainKmerDatabase(parquet_skdb_path)


@pytest.mark.parametrize(
    "parquet_skdb_path", [{"non_numeric_counts": True}], indirect=True
)
def test_skdb_init_non_numeric_counts_error(parquet_skdb_path: pathlib.Path):
    with pytest.raises(
        RuntimeError,
        match=r"Failed to convert DataFrame to NumPy array: Non-numeric data found in DataFrame values. Cannot convert to count matrix. Error: could not convert string to float: 'val1'",
    ):
        StrainKmerDatabase(parquet_skdb_path)


@pytest.mark.parametrize(
    "parquet_skdb_path", [{"non_unique_kmers": True}], indirect=True
)
def test_skdb_init_non_unique_kmers_warning(
    parquet_skdb_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    StrainKmerDatabase(parquet_skdb_path)
    captured = capsys.readouterr()
    assert any(
        "Warning: K-mer index in" in line and "is not unique." in line
        for line in captured.out.splitlines()
    )


# --- Tests for get_strain_counts_for_kmer (adapted from lookup_kmer) ---
@pytest.mark.parametrize(
    "parquet_skdb_path",
    [{"kmer_type": "str", "custom_kmers": ["ATGC", "CGTA"]}],
    indirect=True,
)
def test_skdb_get_strain_counts_for_kmer(
    parquet_skdb_path: pathlib.Path, strain_names_fixture_skdb: List[str]
):
    db = StrainKmerDatabase(parquet_skdb_path, expected_kmer_length=4)

    known_kmer_bytes = b"ATGC"
    counts = db.get_strain_counts_for_kmer(known_kmer_bytes)
    assert counts is not None
    assert isinstance(counts, np.ndarray)
    assert counts.dtype == np.uint8
    assert len(counts) == len(strain_names_fixture_skdb)

    unknown_kmer_bytes = b"XXXX"
    assert db.get_strain_counts_for_kmer(unknown_kmer_bytes) is None

    incorrect_length_kmer = b"ATG"
    assert db.get_strain_counts_for_kmer(incorrect_length_kmer) is None

    assert db.get_strain_counts_for_kmer("ATGC") is None  # type: ignore


# --- Tests for __len__ ---
@pytest.mark.parametrize("parquet_skdb_path", [{"kmer_type": "str"}], indirect=True)
def test_skdb_len_method(
    parquet_skdb_path: pathlib.Path, sample_kmers_str_skdb: List[str]
):
    db = StrainKmerDatabase(parquet_skdb_path)
    assert len(db) == len(sample_kmers_str_skdb)


# --- Tests for __contains__ ---
@pytest.mark.parametrize("parquet_skdb_path", [{"kmer_type": "str"}], indirect=True)
def test_skdb_contains_method(
    parquet_skdb_path: pathlib.Path,
    sample_kmers_str_skdb: List[str],
    default_kmer_length_skdb: int,
):
    db = StrainKmerDatabase(parquet_skdb_path)

    known_kmer_bytes = sample_kmers_str_skdb[0].encode("utf-8")
    assert known_kmer_bytes in db

    unknown_kmer = ("Z" * default_kmer_length_skdb).encode("utf-8")
    assert unknown_kmer not in db

    known_kmer_as_str = sample_kmers_str_skdb[0].encode("utf-8")
    assert known_kmer_as_str in db  # Corrected: known_kmer_as_str should be in db


# --- Tests for get_database_stats (New tests for merged method) ---
@pytest.mark.parametrize(
    "parquet_skdb_path",
    [
        {
            "kmer_type": "str",
            "custom_kmers": ["AAAA", "CCCC", "GGGG", "TTTT", "ACGT", "TGCA"],
        }
    ],
    indirect=True,
)
def test_skdb_get_database_stats(
    parquet_skdb_path: pathlib.Path,
    default_kmer_length_skdb: int,
    strain_names_fixture_skdb: List[str],
):
    db = StrainKmerDatabase(
        parquet_skdb_path, expected_kmer_length=default_kmer_length_skdb
    )
    stats = db.get_database_stats()

    assert stats["num_strains"] == len(strain_names_fixture_skdb)
    assert stats["num_kmers"] == 6
    assert stats["kmer_length"] == default_kmer_length_skdb
    assert stats["database_path"] == str(parquet_skdb_path.resolve())  # Corrected key
    assert len(stats["strain_names_preview"]) <= 5
    assert stats["strain_names_preview"] == strain_names_fixture_skdb[:5]
    assert stats["total_strain_names"] == len(strain_names_fixture_skdb)


# --- Tests for validate_kmer_length (New tests for merged method) ---
@pytest.mark.parametrize("parquet_skdb_path", [{"kmer_type": "str"}], indirect=True)
def test_skdb_validate_kmer_length(
    parquet_skdb_path: pathlib.Path, default_kmer_length_skdb: int
):
    db = StrainKmerDatabase(
        parquet_skdb_path, expected_kmer_length=default_kmer_length_skdb
    )

    correct_len_str = "X" * default_kmer_length_skdb
    incorrect_len_str = "X" * (default_kmer_length_skdb - 1)
    correct_len_bytes = b"Y" * default_kmer_length_skdb
    incorrect_len_bytes = b"Y" * (default_kmer_length_skdb - 1)

    assert db.validate_kmer_length(correct_len_str) is True
    assert db.validate_kmer_length(incorrect_len_str) is False
    assert db.validate_kmer_length(correct_len_bytes) is True
    assert db.validate_kmer_length(incorrect_len_bytes) is False

    assert (
        db.validate_kmer_length("AAAAA") == default_kmer_length_skdb
    ) is False  # Length 5, kmer_length is 4
    assert db.validate_kmer_length("".join(["A"] * default_kmer_length_skdb)) is True
    assert (
        db.validate_kmer_length("".join(["A"] * (default_kmer_length_skdb - 1)))
        is False
    )
