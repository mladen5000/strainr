#!/usr/bin/env python

import argparse
import gzip
import logging
import pathlib
import pickle
import sys
from collections import defaultdict
from functools import partial
from mimetypes import guess_type

import ncbi_genome_download as ngd
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


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
        "-g",
        "--genus",
        type=str,
        required=False,
        help="Name of the genus",
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


def parse_assembly_level():
    """
    Return ncbi-genome-download parameters based on input
    """
    if params["assembly_accessions"]:
        return "all"

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
    """Calls ncbi-genome-download with some preset arguments:
    Returns:
        genome_directory named based on input information
        exit_code: for ncbi-genome-download checking
    """

    # Get inclusive assembly level
    assembly_level: str = parse_assembly_level()

    # Build command dict
    ncbi_kwargs = {
        "flat_output": True,
        "groups": "bacteria",
        "file_formats": "fasta",
        "section": params["source"],
        "parallel": params["procs"],
        "assembly_levels": assembly_level,
    }

    if params["taxid"] and params["assembly_accessions"]:
        raise ValueError("Cannot select both taxid and accession")

    elif params["taxid"]:
        ncbi_kwargs["species_taxids"] = params["taxid"]
        output_dir = "genomes_s" + str(params["taxid"])
        accession_summary = "summary_" + str(params["taxid"]) + ".tsv"

    elif params["assembly_accessions"]:
        ncbi_kwargs["assembly_accessions"] = params["assembly_accessions"]
        output_dir = "genomes_f" + params["assembly_accessions"]
        accession_summary = "summary.tsv"

    elif params["genus"]:
        ncbi_kwargs["genera"] = params["genus"]
        output_dir = "genomes_g" + params["genus"]
        accession_summary = "summary_" + params["genus"] + ".tsv"

    else:
        raise ValueError(
            "Need to choose either taxid or provide an accession list from a file."
        )

    output_dir = p / output_dir
    accession_summary = p / output_dir / accession_summary
    ncbi_kwargs.update(
        {
            "output": output_dir,
            "metadata_table": accession_summary,
        }
    )

    # Call ncbi-genome-download
    exitcode: int = ngd.download(**ncbi_kwargs)
    if exitcode != 0:
        raise ConnectionError(
            "ncbi-genome-download did not successfully download the genomes"
        )

    return output_dir, accession_summary


def download_and_filter_genomes():
    """Grab all, taxid-only, or custom lists"""

    # Skip ncbi-download if providing genomes
    if params["custom"]:
        accfile = ""
        return list((p / params["custom"]).glob("*")), accfile

    # Call ncbi-genome-download
    genomedir, accfile = download_strains()

    # Filter genome list if only unique taxids desired
    if params["unique_taxid"]:
        return unique_taxid_strains(accfile), accfile

    return list(genomedir.glob("*fna.gz")), accfile


def count_kmers(genome_file):
    klen = params["kmerlen"]
    kmerset = set()
    encoding = guess_type(genome_file)[1]  # uses file extension
    _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open

    with _open(genome_file) as g:
        for record in SeqIO.parse(g, "fasta"):
            max_index = len(record.seq) - klen + 1
            print(max_index)
            with memoryview(bytes(record.seq)) as seq_buffer:
                kmers = {bytes(seq_buffer[i : i + klen]) for i in range(max_index)}
            kmerset.update(kmers)
        logger.info(len(kmerset))
    return kmerset


def build_database(genome_files, sequence_names):
    """
    Input:
        List of single-sequence (genome) fasta files
    Does:
        Full build - functional programming style.
        Grabs each file and generates kmers which are then placed into the
        dictionary.
        Strain ID is appended upon collision
    Output:
        Database of kmer: strain_hits
    """
    idx = 0
    kmerlen = params["kmerlen"]
    database = defaultdict(partial(np.zeros, len(genome_files), dtype=bool))

    # logger.info("Building database....")

    # Each genome file as a label
    for genome_file in tqdm(genome_files):
        print(f"Genome file: {genome_file}")
        encoding = guess_type(str(genome_file))[1]
        _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open

        with _open(genome_file) as g:
            for record in SeqIO.parse(
                g, "fasta"
            ):  # Include all subsequences under one label
                max_index = len(record.seq) - kmerlen + 1
                with memoryview(bytes(record.seq)) as seq_buffer:
                    for i in range(max_index):
                        kmer = seq_buffer[i : i + kmerlen]
                        database[bytes(kmer)][idx] = True
        idx += 1

    return database


def build_parallel(genome_file, genome_files, full_set):
    """
    Input: List of single-sequence (genome) fasta files
    Full build - functional programming style.
    Grabs each file and generates kmers which are then placed into the
    dictionary.
    Strain ID is appended upon collision
    Output: Database of kmer: strain_hits
    """
    kmerlen = params["kmerlen"]
    col = genome_files.index(genome_file)
    rows = []
    encoding = guess_type(str(genome_file))[1]
    _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open

    with _open(genome_file) as g:
        for record in SeqIO.parse(
            g, "fasta"
        ):  # Include all subsequences under one label
            max_index = len(record.seq) - kmerlen + 1

            with memoryview(bytes(record.seq)) as seq_buffer:
                for i in range(max_index):
                    kmer = seq_buffer[i : i + kmerlen]
                    rows.append(full_set.index(kmer))

    print(col, rows)
    return rows


def build_df(db, strain_list):
    """Build the dataframe"""

    values = np.array(list(db.values()))
    df = pd.DataFrame(values, index=db.keys(), columns=strain_list, dtype=bool)

    # df = pd.DataFrame.from_dict(db, orient="index", columns=strain_list, dtype=bool)
    logger.debug(df)
    return df


def parse_meta():
    """Parse assembly data"""

    meta = pd.read_csv("ngd_meta.tsv", sep="\t").set_index("assembly_accession")
    # meta_vars = list(meta.T.index)

    return meta


def unique_taxid_strains(accession_file):
    """
    To be used with complete 1 and filter out for genomes without
        strain taxonomic IDs and those without unique strain taxonomic IDs
        Ideally to be used for large genomes such as ecoli with large redundancy
    """

    accessions = pd.read_csv(accession_file, sep="\t").set_index("assembly_accession")

    # Filter out non-uniques
    unique = accessions.taxid != accessions.species_taxid
    exists = accessions.taxid.notna()
    accessions = accessions[unique & exists]

    # Write new accession details
    accessions.to_csv(accession_file, sep="\t")

    return [(p / i) for i in accessions["local_filename"].to_list()]


def get_genome_names(genome_files, accfile):
    """Function to go from files -> genome names"""

    if not params["custom"]:
        accessions = pd.read_csv(accfile, sep="\t").set_index("assembly_accession")
        genome_names = []

        for gf in genome_files:
            acc = gf.stem[:15]
            genome_name = accessions.loc[acc]["organism_name"]
            accessions["infraspecific_name"] = accessions.infraspecific_name.astype(str)
            strain_name = accessions.loc[acc]["infraspecific_name"]
            strain_name = strain_name.replace("strain=", " ")
            genome_names.append(genome_name + strain_name + " " + acc)

            # encoding = guess_type(str(gf))[1]  # uses file extension
            # _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open
            # with _open(gf) as gfo:
            #     for g in SeqIO.parse(gfo,'fasta'):
            #         sequence_names.append(g.description)

    # if not params["custom"]:
    #     genome_names = [gf.stem[:15] for gf in genome_files]

    else:
        genome_names = [gf.stem for gf in genome_files]
    assert len(genome_files) == len(genome_names)

    return genome_names


def pickle_db(database, fout):
    outfile = fout + ".pkl"
    logger.info(f"Saving database as {outfile}")

    with open(outfile, "wb") as ph:
        pickle.dump(database, ph, protocol=pickle.HIGHEST_PROTOCOL)

    return


def pickle_df(df, filename, method="pickle"):
    outfile = params["out"] + ".db"

    if method == "pickle":
        df.to_pickle(outfile)

    elif method == "hdf":
        outfile = params["out"] + ".sdb"
        df.to_hdf(outfile)
        pd.DataFrame().to_hdf

    return


def main():
    # Run - Download
    genome_files, accession_summary = download_and_filter_genomes()
    sequence_ids = get_genome_names(genome_files, accession_summary)

    # Log results 1
    print(sequence_ids)
    logger.info(f"{len(genome_files)} genomes found.")

    # Build Database
    database = build_database(genome_files, sequence_ids)
    df = build_df(database, sequence_ids)

    # Log results 2
    logger.debug(f"{len(database)} kmers in database")
    logger.debug(f"{sys.getsizeof(database)//1e6} MB")
    logger.debug(f"Kmer-building complete, saving db as {params['out']+'.db'}.")

    # Save Database
    pickle_df(df, params["out"])

    return
    logger.info(f"Database saved to {params['out']} ")


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


if __name__ == "__main__":
    p = pathlib.Path().cwd()
    params = vars(get_args().parse_args())
    main()
