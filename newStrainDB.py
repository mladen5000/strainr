#!/usr/bin/env python

import argparse
import shutil
import gzip
import pathlib
import pickle
import sys
import logging
from collections import defaultdict
from mimetypes import guess_type
from functools import partial

from tqdm import tqdm
from Bio import SeqIO
import numpy as np
import pandas as pd
import ncbi_genome_download as ngd

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
    if c == "complete":
        return "complete"
    elif c == "chromosome":
        return "complete,chromosome"
    elif c == "scaffold":
        return "complete,chromosome,scaffold"
    elif c == "contig":
        return "complete,chromosome,scaffold,contig"
    else:
        raise ValueError("Incorrect assembly level selected.")


def download_strains():
    """ """

    assembly_level = parse_assembly_level()
    if params["taxid"] and params["assembly_accessions"]:
        raise ValueError("Cannot select both taxid and accession")
    elif params["taxid"]:
        exitcode = ngd.download(
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
        exitcode = ngd.download(
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
            "Need to choose either taxid or provide an accession list from a file."
        )
    if exitcode != 0:
        raise ValueError(f"Downloading strains returned with exit code {exitcode}")

    return exitcode


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


def build_database(genome_files, strain_names):
    """
    Input: List of single-sequence (genome) fasta files
    Full build - functional programming style.
    Grabs each file and generates kmers which are then placed into the
    dictionary.
    Strain ID is appended upon collision
    Output: Database of kmer: strain_hits
    """
    logger.info("Building database....")
    # database = defaultdict(list)
    database = defaultdict(partial(np.zeros, len(strain_names), dtype=bool))
    kmerlen = params["kmerlen"]
    for genome_file in tqdm(genome_files):
        encoding = guess_type(genome_file)[1]  # uses file extension
        _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open
        # Get record
        with _open(genome_file) as g:
            for record in SeqIO.parse(g, "fasta"):

                max_index = len(record.seq) - kmerlen + 1
                acc = genome_file.stem[:15]
                with memoryview(bytes(record.seq)) as seq_buffer:
                    for i in range(max_index):
                        kmer = seq_buffer[i : i + kmerlen]
                        # database[bytes(kmer)].append(acc)
                        database[bytes(kmer)][strain_names.index(acc)] = True
    print(list(database.items())[:10])
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


def build_df(db, strain_list):
    """Build the dataframe"""
    values = np.array(list(db.values()))
    df = pd.DataFrame(values, index=db.keys(), columns=strain_list, dtype=bool)
    # df = pd.DataFrame.from_dict(db, orient="index", columns=strain_list, dtype=bool)
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


def parse_meta():
    """Parse assembly data"""
    meta = pd.read_csv("ngd_meta.tsv", sep="\t").set_index("assembly_accession")
    # meta_vars = list(meta.T.index)
    return meta


def unique_taxid_strains():
    """
    To be used with complete 1 and filter out for genomes without
    strain taxonomic IDs and those without unique strain taxonomic IDs
    Ideally to be used for large genomes such as ecoli with large redundancy
    """
    meta = pd.read_csv("ngdmeta.tsv", sep="\t").set_index("assembly_accession")
    print(meta)
    mask1 = meta.taxid != meta.species_taxid
    mask2 = meta.taxid.notna()
    filtered = meta[mask1 & mask2]
    filtered.to_csv("ngdmeta2.tsv", sep="\t")
    strain_files = filtered.local_filename.to_list()
    strain_files = [p / i for i in strain_files]
    return strain_files


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
        Assembly levels of genomes to download (default: complete).
        Each option includes previous options e.g) 'contig' will download
        complete genomes, chromosomes and scaffolds as well.\n
        """,
        choices=["complete", "chromosome", "scaffold", "contig"],
        type=str,
        default="complete",
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
    parser.add_argument(
        "--custom",
        type=str,
        required=False,
        help="Folder containing custom set of genomes for download",
    )
    parser.add_argument(
        "--unique-taxid",
        action="store_true",
        required=False,
        help="""
                        Optional flag to only build the database for genomes that 
                        have a unique strain taxonomic ID, all downloaded
                        genomes without a taxid 
                        or only with a species-taxid will be downloaded, 
                        but not incorporated into the final database. 
                        Useful for species which have a large number of 
                        genomes in the database, such as E. Coli.""",
    )
    return parser


def download_and_filter_genomes():
    gdir = p / "genomes"
    if not params["custom"]:
        if gdir.is_dir():
            shutil.rmtree(gdir)
        download_strains()

        if params["unique_taxid"]:
            file_list = unique_taxid_strains()
        else:
            file_list = list(gdir.glob("*fna.gz"))

    else:  # custom
        file_list = list((p / params["custom"]).glob("*"))

    return file_list


def get_genome_names(genome_files):
    """Function to go from files -> genome names"""
    if not params["custom"]:
        return [gf.stem[:15] for gf in genome_files]
    else:
        return [gf.stem[:15] for gf in genome_files]


def main():
    # Run - Download
    genome_files = download_and_filter_genomes()
    genome_ids = get_genome_names(genome_files)
    logger.info(f"{len(genome_files)} genomes found.")

    # Build Database
    database = build_database(genome_files, genome_ids)
    df = build_df(database, genome_ids)
    save_df(df, params["out"])

    logger.debug("Before modifications")
    logger.debug(f"{len(database)} kmers in database")
    logger.debug(f"{sys.getsizeof(database)//1e6} MB")
    logger.debug("Kmer-building complete, creating db..")

    # Modifications 2
    # strain_list, encoded = multi_encode(database)

    # Modifications
    # database = convert_to_presence_absence(database)
    # database = filter_most(database, len(file_list))
    # database = filter_by_length(database, 5)
    # database = full_sort(database)

    # logger.info("After modifications")
    # logger.debug(len(database))
    # logger.debug(sys.getsizeof(database))
    # pickle_db(database)


if __name__ == "__main__":
    p = pathlib.Path().cwd()
    params = vars(get_args().parse_args())
    main()
