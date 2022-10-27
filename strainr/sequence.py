from dataclasses import dataclass, field
import mmh3


@dataclass(order=True, slots=True)
class SimpleSeq:
    """
    SimpleSeq: A slotted class to hold a self.seq and a name
        Just a slotted class to hold a self.seq and a name
        Held as bytes and slotted, so it's immutable and hashable
        Also a decent amount of validation for bytes.


    Example:
        x = SimpleSeq(name='sequence_1',seq=b'ACTTTAAGGGGTTAAACCCCCG'*100)
        # x.get_kmers(StrainDatabase)
        [bytes(i) for i in x.kmers]

    """

    name: str = field(compare=False)
    seq: bytes

    def validate_sequence(self) -> bytes:
        """Validate the self.seq data type"""
        # Check for minimum length
        if len(self.seq) == 0:
            raise ValueError("self.seq must be non-empty")
        # type check
        if not isinstance(self.seq, bytes):
            try:
                self.seq = bytes(self.seq, encoding="ascii")
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise TypeError from err

        # basepair check
        allowed_bases = {"A", "C", "G", "T", "N"}
        found_bases = set(self.seq.decode())

        if found_bases.difference(allowed_bases):
            raise ValueError(
                f"self.seq contains invalid characters: {found_bases.difference(allowed_bases)}"
            )

        return self.seq

    def __post_init__(self) -> None:
        """1. Validate the self.seq data type [parse_seq]"""

        self.seq = self.validate_sequence()

    def __hash__(self) -> int:
        return mmh3.hash_bytes(self.seq)

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, index: int) -> str:
        return self.seq.decode()[index]

    def __str__(self) -> str:
        return self.seq.decode(encoding="ascii")


def get_kmers(sequence: SimpleSeq, k: int = 31) -> list[bytes]:
    """Main function to assign strain hits to reads"""
    max_index: int = len(sequence.seq) - k + 1
    with memoryview(sequence.seq) as kmer_view:
        sequence.kmers = [kmer_view[index : index + k] for index in range(max_index)]  # type: ignore
