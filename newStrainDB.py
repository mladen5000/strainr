#!/usr/bin/env python

import argparse
import gzip
import pathlib
import pickle
import sys
import logging
from collections import defaultdict
from functools import partial
from mimetypes import guess_type

from tqdm import tqdm
from Bio import SeqIO
import pandas as pd
import ncbi_genome_download as ngd
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def parse_assembly_level():
    """
    Return ncbi-genome-download parameters based on input
    """
    c = params["assembly_levels"]
    print(c)
    if c == "complete1" or c == "complete2":
        return "complete"
    elif c == "chromosome":
        return "complete,chromosome"
    elif c == "scaffold":
        return "complete,chromosome,scaffold"
    elif c == "contig":
        return "complete,chromosome,scaffold,contig"
    else:
        raise ValueError("Incorrect assembly level selected.")
    return


def download_strains():
    """ """

    assembly_level = parse_assembly_level()
    print(assembly_level)
    if params["taxid"] and params["assembly_accessions"]:
        raise ValueError("Cannot select both taxid and accession")
    elif params["taxid"]:
        ngd.download(
            flat_output=True,
            groups="bacteria",
            file_formats="fasta",
            output=(p / "genomes"),
            metadata_table=(p / "ngdmeta.tsv"),
            assembly_levels=assembly_level,
            species_taxids=params["taxid"],
            section=params["source"],
            parallel=params["procs"],
        )
    elif params["assembly_accessions"]:
        ngd.download(
            flat_output=True,
            groups="bacteria",
            file_formats="fasta",
            output=(p / "genomes"),
            metadata_table=(p / "ngdmeta.tsv"),
            assembly_levels=assembly_level,
            section=params["source"],
            parallel=params["procs"],
            assembly_accessions=params["assembly_accessions"],
        )
    else:
        raise ValueError(
            "Need to choose either taxid or provide an accession list from a file"
        )
    return


def count_kmers(genome_file):
    encoding = guess_type(genome_file)[1]  # uses file extension
    _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open
    with _open(genome_file) as g:
        record = SeqIO.read(g, "fasta")
    kmerlen = params["kmerlen"]
    max_index = len(record.seq) - kmerlen + 1
    with memoryview(bytes(record.seq)) as seq_buffer:
        kmerset = {bytes(seq_buffer[i : i + kmerlen]) for i in range(max_index)}
    logger.info(len(kmerset))
    return kmerset


def extract_acc(filepath):
    fname = filepath.stem
    return fname[:15]


def analyze_genome(genome_file):
    kmers = count_kmers(genome_file)
    acc = extract_acc(genome_file)
    metadata_dict = dict(filepath=genome_file, kmer_set=kmers, accession=acc)
    return metadata_dict


def pickle_genome(metadata, kmerdir):
    ppath = kmerdir / metadata["accession"] + ".pkl"
    with ppath.open("wb") as ph:
        pickle.dump(metadata, ph)
    return


def build_database(genome_files):
    """
    Input: List of single-sequence (genome) fasta files
    Full build - functional programming style.
    Grabs each file and generates kmers which are then placed into the
    dictionary.
    Strain ID is appended upon collision
    Output: Database of kmer: strain_hits
    """
    logger.info("Building database....")
    database = defaultdict(list)
    kmerlen = params["kmerlen"]
    for genome_file in tqdm(genome_files):
        encoding = guess_type(genome_file)[1]  # uses file extension
        _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open
        # Get record
        with _open(genome_file) as g:
            # record = SeqIO.read(g, "fasta")
            for record in SeqIO.parse(g, "fasta"):

                # Main loop
                max_index = len(record.seq) - kmerlen + 1
                acc = genome_file.stem[:15]
                with memoryview(bytes(record.seq)) as seq_buffer:
                    for i in range(max_index):
                        kmer = seq_buffer[i : i + kmerlen]
                        database[bytes(kmer)].append(acc)
    return database


def full_sort(db):
    """Sorts each list as well as the dict itself"""
    logger.info("Optimizing The Dictionary")
    return {key: sorted(db[key]) for key in sorted(db)}


def convert_to_presence_absence(db):
    """
    Convert database so each entry only has a strain appearing up to 1 time
    """
    return {k: set(v) for k, v in db.items()}


def filter_most(db, num_strains):
    """Remove kmers that have hits for most strains"""
    return {k: v for k, v in db.items() if len(v) > num_strains // 2}


def filter_by_length(db, max_len):
    """
    Remove kmers that have more than n hits
    eg) max_len = 1 for unique kmers only
    """
    return {k: v for k, v in db.items() if len(v) > max_len}


def pickle_db(database):
    outfile = params["out"] + ".pkl"
    logger.info(f"Saving database as {outfile}")
    with open(outfile, "wb") as ph:
        pickle.dump(database, ph, protocol=pickle.HIGHEST_PROTOCOL)
    return


def multi_encode(db):
    """
    Return each hit as a binary set of presence/absence
    Requires sklearn
    """
    mlb = MultiLabelBinarizer()
    val_array = mlb.fit_transform(db.values())
    strains = mlb.classes_
    logger.info(strains)
    return strains, val_array


def build_df(db, val_array, strain_list):
    """Build the dataframe"""
    df = pd.DataFrame(val_array, index=db.keys(), dtype=bool)
    df.columns = strain_list
    df.index = df.index  # .str.decode('utf-8')
    logger.debug(df)
    return df


def save_df(df, filename, method="pickle"):

    outfile = params["out"] + ".db"
    if method == "pickle":
        df.to_pickle(outfile)
    elif method == "hdf":
        outfile = params["out"] + ".sdb"
        df.to_hdf(outfile)
        pd.DataFrame().to_hdf
    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--taxid",
        type=str,
        help="Species taxonomic ID from which all strains will be downloaded",
        required=False,
    )
    parser.add_argument(
        "-f",
        "--assembly_accessions",
        type=str,
        help="List of assembly accessions - 1 per line",
    )
    parser.add_argument(
        "-k",
        "--kmerlen",
        help="K-mer length. (default: 31)\n",
        type=int,
        default=31,
    )
    parser.add_argument(
        "-l",
        "--assembly_levels",
        help="""
        Assembly levels of genomes to download (default: complete1).
        Each option includes previous options e.g) 'contig' will download
        complete genomes, chromosomes and scaffolds as well.
        Complete1 only downloads complete genomes which have taxonomic IDs
        whereas complete2 will download all complete genomes\n
        """,
        choices=["complete1", "complete2", "chromosome", "scaffold", "contig"],
        type=str,
        default="complete1",
    )
    parser.add_argument(
        "-s",
        "--source",
        help="Choice of NCBI reference database (default: refseq)\n",
        choices=["refseq", "genbank"],
        type=str,
        default="refseq",
    )
    parser.add_argument(
        "-p",
        "--procs",
        type=int,
        default=1,
        help="Number of cores to use (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default="database",
        help="Output name of the database (optional)\n",
    )
    return parser


def main():
    # Run - Download
    download_strains()
    file_list = list((p / "genomes").glob("*fna.gz"))
    logger.info(f"{len(file_list)} genomes found.")

    # Run - Build
    database = build_database(file_list)

    logger.debug("Before modifications")
    logger.debug(f"{len(database)} kmers in database")
    logger.debug(sys.getsizeof(database))
    logger.debug("DB complete, encoding..")

    # Modifications
    # database = convert_to_presence_absence(database)
    # database = filter_most(database, len(file_list))
    # database = filter_by_length(database, 5)
    # database = full_sort(database)

    # Modifications 2
    strain_list, encoded = multi_encode(database)
    df = build_df(database, encoded, strain_list)
    save_df(df, params["out"])

    logger.info("After modifications")
    logger.debug(len(database))
    logger.debug(sys.getsizeof(database))
    pickle_db(database)


if __name__ == "__main__":
    p = pathlib.Path().cwd()
    params = vars(get_args().parse_args())
    main()
