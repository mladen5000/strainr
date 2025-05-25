import dataclasses
import pathlib

from strainr.kmer_database import KmerStrainDatabase


class SequenceFile(pathlib.Path):
    """Class to hold the input sequence file"""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path

    def __post_init__(self) -> None:
        """Check that the input file exists"""
        if not self.exists():
            raise FileNotFoundError(f"File {self} does not exist")


def main():
    """_summary_"""

    args = params.process_arguments()
    DATA_DIR = (pathlib.Path(__file__).parent / "data").resolve()

    # From the command line
    input_sequences: SequenceFile = args.input
    results_dir: pathlib.Path = args.outdir
    db_path: pathlib.Path = args.db
    k = args.k

    # Generate the database
    database = KmerStrainDatabase(args.k)
    strain_run = Runner(input_sequences, database)


@dataclasses.dataclass
class Runner:
    """Container class for run parameters"""

    fasta = pathlib.Path
    kmer_database: KmerStrainDatabase = KmerStrainDatabase()
    k: int = 31

    # db, strains, kmerlen = build_database(args.db)
    # p = pathlib.Path().cwd()
    # rng = np.random.default_rng()
    # print("\n".join(f"{k} = {v}" for k, v in vars(args).items()))
    # for in_fasta in args.input:
    #     t0 = time.time()
    #     fasta = pathlib.Path(in_fasta)
    #     out: pathlib.Path = args.out / str(fasta.stem)
    #     if not out.exists():
    #         print(f"Input file:{fasta}")
    #         main()
    #         print(f"Time for {fasta}: {time.time()-t0}")
    #         print(f"Time for {fasta}: {time.time()-t0}")
