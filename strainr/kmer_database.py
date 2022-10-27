from dataclasses import dataclass, field
from typing import ClassVar, Union
import pathlib
import pandas as pd

import numpy as np
import numpy.typing as npt

# CountVector = NDArray[Shape["*"], UInt8]
CountVector = npt.NDArray(np.uint8)
Kmer = Union[memoryview, bytes]
KmerDict = dict[Kmer, CountVector]


class StrainDatabase:
    """Class to hold the strain database"""

    def __init__(self, kmerdb_path: str, k: int = 31) -> None:
        """Load the database from a file"""

        kmer_strain_table: pd.DataFrame = pd.read_pickle(kmerdb_path)
        kmer_strain_table.index

        strain_refs = kmer_strain_table.columns  # .to_list()
        kmers = kmer_strain_table.index
        frequencies = kmer_strain_table.to_numpy()

        assert all(kmer_strain_table.index.str.len() == k)  # Check all kmers
        self.k = 31 if not k else k
        self._data = dict(zip(kmers, strain_frequency))

        print(f"Database of {len(strains)} strains loaded")
        return db, strains, kmerlen

    ref_strains: list
    k: int = 31
    keys: str

    db: ClassVar[KmerDict] = dict(zip(df.index, df.to_numpy()))
    filepath: pathlib.Path = "~/Strainr_Extra/databases_idk/elenta_10genomes.db"

    df = pd.read_pickle(filepath)
    # strain_names: ClassVar[list[str]] = list(df.columns)


# @dataclass(slots=True)
# class StrainDatabase:
#     """Class to hold the strain database"""

# filepath: pathlib.Path = "~/Strainr_Extra/databases_idk/elenta_10genomes.db"

# df = pd.read_pickle(filepath)
# strain_names: ClassVar[list[str]] = list(df.columns)
# num_strains: ClassVar[int] = df.shape[1]
# db: ClassVar[KmerDict] = dict(zip(df.index, df.to_numpy()))
# k: ClassVar[int] = 31

# def __init__(self, filepath) -> None:
#     self.filepath = filepath
#     self.df = pd.read_pickle(self.filepath)
#     num_kmers, num_strains = df.shape
#     strain_names = list(df.columns)
#     db = dict(zip(df.index, df.to_numpy(dtype=np.uint8)))

# def __post_init__(self) -> None:
#     """
#     Calculate the data attributes and load the database
#     Args: filepath (str): Location of the pickle file containing the database
#     # global self.num_strain, self.strains_names, self.db
#     """
