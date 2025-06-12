from dataclasses import dataclass, field
from typing import ClassVar, Union
import pathlib

import mmh3
import numpy as np
import pandas as pd

from Bio.Seq import Seq
try:
    from nptyping import NDArray, Shape, UInt8
except ImportError:
    # Fallback type hints if nptyping is not available
    from typing import Any
    NDArray = Any
    Shape = Any
    UInt8 = Any

CountVector = NDArray[Shape["*"], UInt8]
Kmer = Union[memoryview, bytes]
KmerDict = dict[Kmer, CountVector]


@dataclass(order=True, slots=True)
class SimpleSeq:
    """
    SimpleSeq: A slotted class to hold a self.seq and a name
        Just a slotted class to hold a self.seq and a name
        Held as bytes and slotted, so it's immutable and hashable
        Also a decent amount of validation for bytes.
    """

    name: str = field(compare=False)
    seq: bytes
    kmers: list[bytes] = field(default_factory=list, compare=False, init=False)
    strain_counts: CountVector = field(
        init=False,
        compare=False,
    )

    def parse_seq(self) -> bytes:
        # Check for minimum length
        if len(self.seq) == 0:
            raise ValueError("self.seq must be non-empty")

        # type check
        if not isinstance(self.seq, bytes):
            try:
                self.seq = bytes(self.seq, encoding="ascii")
            except:
                raise ValueError("self.seq must be ASCII-encoded")

        # basepair check
        allowed_bases = {"A", "C", "G", "T", "N"}
        found_bases = set(self.seq.decode())

        if found_bases.difference(allowed_bases):
            raise ValueError(
                f"self.seq contains invalid characters: {found_bases.difference(allowed_bases)}"
            )

        return self.seq

    def __post_init__(self) -> None:
        """1. Validate the self.seq data type [parse_seq]
        2. initialize the strain_counts array
        """

        self.seq = self.parse_seq()
        self.strain_counts = np.zeros(len(num_strains), dtype=np.uint8)

    def __hash__(self) -> int:
        return mmh3.hash_bytes(self.seq)

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, index: int) -> str:
        return self.seq.decode()[index]

    def __str__(self) -> str:
        return self.seq.decode(encoding="ascii")

    def get_kmers(self, StrainDatabase) -> tuple[str, np.ndarray]:
        """Main function to assign strain hits to reads"""
        na_zeros: CountVector = np.zeros(StrainDatabase.num_strains, dtype=np.uint8)
        max_index: int = len(self.seq) - StrainDatabase.k + 1

        with memoryview(self.seq) as kmer_view:
            return [
                StrainDatabase.db.get(
                    kmer_view[index : index + StrainDatabase.k], na_zeros
                )
                for index in range(max_index)
            ]


@dataclass(
    slots=True,
)
class StrainDatabase:
    """Class to hold the strain database"""

    global num_strains, strain_names, db, kmer_len
    strain_names: ClassVar[list[str]] = field(init=False)
    num_kmers: int = field(init=False)
    num_strains: ClassVar[int] = field(init=False)
    db: ClassVar[KmerDict] = field(init=False, repr=False)

    filepath: pathlib.Path = field(init=False)
    kmer_len: ClassVar[int] = 31

    def __post_init__(self) -> None:
        """
        Calculate the data attributes and load the database
        Args: filepath (str): Location of the Parquet file containing the database
        # global self.num_strain, self.strains_names, self.db
        """
        # load parquet
        df = pd.read_parquet(self.filepath)

        # assign data attributes
        self.strain_names = list(df.columns)
        self.num_kmers, self.num_strains = df.shape
        self.kmer_len = len(df.index[0])
        self.db = dict(zip(df.index, df.to_numpy()))


getattr(StrainDatabase, "__slots__")


elenta_path = pathlib.Path(
    "~/Strainr_Extra/databases_idk/elenta_10genomes.db.parquet"
)  # Updated extension
db = StrainDatabase(elenta_path)

for k, v in StrainDatabase.__dict__.items():
    print(k, v, sep="\t")


def get_kmers(read: SimpleSeq, db: StrainDatabase, k=31) -> tuple[str, np.ndarray]:
    """Main function to assign strain hits to reads"""
    CountVector = NDArray[Shape["*"], UInt8]
    na_zeros: CountVector = np.zeros(db.num_strains, dtype=np.uint8)
    max_index: int = len(read.seq) - k + 1
    with memoryview(read.seq) as kmer_view:
        SimpleSeq.strain_counts = np.sum(
            db.db.get(kmer_view[index : index + k], na_zeros)
            for index in range(max_index)
        )


def hash_kmers(kmers):
    # hash and collect all kmers
    hashes = []
    for kmer in kmers:
        hashes.append(hash_kmer(kmer))
    return hashes


def hash_kmer(DNA_read: Seq):
    # calculate the reverse complement
    DNA_read_rc: str | Seq = Seq(DNA_read).reverse_complement()

    # determine whether original k-mer or reverse complement is lesser
    if DNA_read < DNA_read_rc:
        canonical_read = DNA_read
    else:
        canonical_read: str | Seq = DNA_read_rc

    # calculate murmurhash using a hash seed of 42
    hash: int = mmh3.hash64(canonical_read, 42)[0]
    if hash < 0:
        hash += 2**64

    # done
    return hash


def main() -> None:
    myseq2 = SimpleSeq(name="s1", seq="ATCGATCGATCG")
    NDArray[Shape["*"], UInt8]
    elenta = StrainDatabase(
        "~/Strainr_Extra/databases_idk/elenta_10genomes.db.parquet"
    )  # Updated extension
    get_kmers(myseq2, elenta)


if __name__ == "__main__":
    main()
