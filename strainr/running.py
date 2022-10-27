from configparser import ConfigParser
import configparser
from dataclasses import dataclass


import dataclasses
from types import NoneType
import biom

from matplotlib.path import Path
import argparse
from pysam import FastaFile

from sklearn.feature_selection import SequentialFeatureSelector
from sympy import GeneratorsError

from strainr.kmer_database import StrainDatabase


def main():

    DATA_DIR = (Path(__file__).parent / "data").resolve()
    config_args= argparse.process_arguments()

    # Generate a StrainDatabase object
    database = StrainDatabase()


    strain_run = Runner(config_args, database)

from typing import Generator, Type
import Bio.SeqIO.FastaIO as b
biom.# from Bio import SeqIO

@dataclasses.dataclass
class Runner:

    FastaFile = Type[Bio.SeqIO.FastaIO.FastaIterator]
    fasta= SeqIO.parse(ConfigParser.input, "fastq")
    k: int = 31
    kmer_database: StrainDatabase = StrainDatabase()


    # args = process_arguments.parse_args()
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
