#!/usr/bin/env python

import argparse
import gzip
import logging
import pathlib
import pickle
import subprocess
import sys

import csv
from collections import defaultdict
from functools import partial
from mimetypes import guess_type

import ncbi_genome_download as ngd
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


class ArgumentError(Exception):
    pass


def get_args() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--cluster",
        action="store_true",
        required=False,
        help="""
            Flag to select clustering, see --ani_cutoff for cutoff.
            This will cluster genomes based on their ANI and build a
            database of genomes with a unique strain taxonomic ID.
            """,
    )
    parser.add_argument(
        "--ani_cutoff",
        type=float,
        required=False,
        default=0.001,
        help="""
            Cutoff for ANI clustering, default is 0.001 (99.99'%' identity).
            """,
    )
    return parser


def parse_assembly_level(level: str) -> str:
    """
    Takes the assembly_level option and selects the all-inclusive optarg for which ncbi-genome-download will use.
    """
    # Implies custom so genomes are already defined
    if args.assembly_accessions:
        return "all"

    level_options = {
        "complete": "complete",
        "chromosome": "complete,chromosome",
        "scaffold": "complete,chromosome,scaffold",
        "contig": "complete,chromosome,scaffold,contig",
    }

    if level not in level_options:
        # However argparse should handle prior to this point
        raise ArgumentError("Incorrect assembly level selected.")

    return level_options[level]


def download_strains():
    """Calls ncbi-genome-download with some preset arguments:
    Returns:
        genome_directory named based on input information
        exit_code: for ncbi-genome-download checking
    """

    # Get inclusive assembly level
    assembly_level: str = parse_assembly_level(args.assembly_levels)

    # Build command dict
    ncbi_kwargs = {
        "flat_output": True,
        "groups": "bacteria",
        "file_formats": "fasta",
        "section": args.source,
        "parallel": args.procs,
        "assembly_levels": assembly_level,
    }

    if args.taxid and args.assembly_accessions:
        raise ValueError("Cannot select both taxid and accession")

    elif args.taxid:
        ncbi_kwargs["species_taxids"] = args.taxid
        output_dir = "genomes_s" + str(args.taxid)
        accession_summary = "summary_" + str(args.taxid) + ".tsv"

    elif args.assembly_accessions:
        ncbi_kwargs["assembly_accessions"] = args.assembly_accessions
        output_dir = "genomes_f" + args.assembly_accessions
        accession_summary = "summary.tsv"

    elif args.genus:
        ncbi_kwargs["genera"] = args.genus
        output_dir = "genomes_g" + args.genus
        accession_summary = "summary_" + args.genus + ".tsv"

    else:
        raise ValueError(
            "Need to choose either taxid or provide an accession list from a file."
        )

    d_out = {
        "output": p / output_dir,
        "metadata_table": p / output_dir / accession_summary,
    }
    ncbi_kwargs.update(d_out)
    print(ncbi_kwargs)

    # Call ncbi-genome-download
    exitcode: int = ngd.download(**ncbi_kwargs)
    if exitcode != 0:
        raise ConnectionError(
            "ncbi-genome-download did not successfully download the genomes"
        )

    # return d_out["output"], d_out["metadata_table"]
    return ncbi_kwargs


# def download_and_filter_genomes():
#     """Grab all, taxid-only, or custom lists"""

#     # Call ncbi-genome-download
#     genomedir, accfile = download_strains()

#     # Filter genome list if only unique taxids desired
#     if params["unique_taxid"]:
#         return unique_taxid_strains(accfile), accfile

#     return list(genomedir.glob("*fna.gz")), accfile

hi := 

def cluster_strains(
    df: pd.DataFrame, ani_threshold=0.001, out1_path=None, out2_path=None
) -> tuple[dict[str, list], dict[str, str], int]:
    """
    Input:
        dists: Mash distance table, all v all genomes (mash sketch * -o blah, mash dist blah.msh blah.msh -t > idklol)
        outfile: File that contains each genome, its' cluster, and dist between genome and cluster rep, (tsv)
        out_clust_table: Column of clustered rep_genomes, followed by members to the right, (tsv)
        thr: threshold, default set to 0.01, smaller ani_threshold will increase the number of clusters.

    Function does the following:
        1. Cluster the genomes
        2. Get representative genome for each cluster

    Returns:
        2 saved files and a clust dict object
    """

    # Hierarchical clustering and cutoff
    Z = sch.linkage(squareform(df.values))
    clust_ids = sch.fcluster(Z, t=ani_threshold, criterion="distance")
    num_clusters = len(np.unique(clust_ids))

    print(
        f"""
            Reducing {len(clust_ids)} genome sequences into {num_clusters} clusters
            using a {(1 - ani_threshold) * 100}% similarity criteria.
        """
    )

    # Tally the genome members in each cluster
    parent_to_child = {}
    for ci in np.unique(clust_ids):
        idx = np.where(ci == clust_ids)[0]

        if idx.shape[0] == 1:  # Single genome in cluster
            r = df.index[idx[0]]
            parent_to_child[r] = [r]

        else:  # Multiple genomes in cluster
            # Put these idx into df for distance table and sum to see total distance
            dist_sum = df.iloc[idx, idx].sum(axis=0)
            rep = dist_sum.idxmin()  # Choose the genome with minimum distance
            parent_to_child[rep] = df.index[idx].tolist()  # Put into dictionary

    # Reverse dict
    child_to_parent = {
        gi: rep_gi
        for rep_gi, genome_list in parent_to_child.items()
        for gi in genome_list
    }

    # TODO - optional output
    # TODO - optional output 2
    if out1_path:
        log_clusters1(out1_path, df, parent_to_child, child_to_parent)
    if out2_path:
        log_clusters2(out2_path, parent_to_child)

    return parent_to_child, child_to_parent, num_clusters


def build_database2(
    p2c: dict[str, list], c2p: dict[str, str], kmerlen: int = 31
) -> tuple[dict[bytes, np.ndarray], list]:
    # TODO: Temporary function to incorporate clusters
    """
    Input:
        List of reference genome files
        p2c - parent genome to list of cluster members d[str,list]
        c2p - child genome matched to str parent d[str,str]
        clusters =unique values in c2p
        genomes = unique keys in c2p
    Output:
        database:
            format: dict[bytes(kmer), np.ndarray(dtype=bool)]
            Each element in np.array represents reference
            bool determines presence/absence of k-mer key
    Note: No labels at this point, so order of genome list must be preserved.
    """
    idx = 0
    cluster_set = list({parent for parent in c2p.values()})
    cluster_set.sort()
    genomes = {fasta for fasta in c2p.keys()}
    database = defaultdict(partial(np.zeros, len(cluster_set), dtype=bool))

    logger.debug(f"There are {len(cluster_set)} clusters and {len(genomes)} genomes")
    logger.info("Building database....")
    logger.debug(f"the kmer length is {kmerlen}")

    """
        For each cluster,
            For each genome within cluster
                For each record in a genome
                    For each kmer in the record
                        DB[kmer]@[cluster_index] = True

    """
    for cluster in tqdm(cluster_set):  # For each cluster
        print(cluster)
        # for genome_file in p2c[cluster]:  # Retrieve children in the cluster
        # with gzopen(genome_file) as gf:
        with gzopen(cluster) as gf:
            for record in SeqIO.parse(gf, "fasta"):
                with memoryview(bytes(record.seq)) as memseq:
                    for ki in range(memseq.nbytes - kmerlen + 1):
                        kmer = memseq[ki : ki + kmerlen]
                        database[bytes(kmer)][idx] = True
        idx += 1
    return database, [pathlib.Path(g) for g in cluster_set]


def log_clusters2(out2_path, clust):
    """Write tsv where each row starts with cluster genome, followed by each member genome within cluster"""
    df_clusters = pd.DataFrame.from_dict(clust, orient="index")
    df_clusters.index.name = "genome_rep"
    most_members = lambda gi: df_clusters.count(axis=1)[gi]
    df_clusters = df_clusters.sort_index(key=most_members, ascending=False)
    df_clusters.to_csv(out2_path, sep="\t")


def log_clusters1(out1_path, df, clust, child_to_parent):
    """Writes tsv where each row is original genome, parent cluster genome and d(genome,cluster_amb)"""
    with open(out1_path, "w") as of:
        writer = csv.writer(of, delimiter="\t", lineterminator="\n")
        writer.writerow(["genome", "rep_genome", "distance"])
        for rep_gi, genome_list in clust.items():
            for gi in genome_list:
                child_to_parent[gi] = rep_gi
                writer.writerow([gi, rep_gi, f"{df.loc[gi, rep_gi]:.6f}"])


def build_database(genomes: list, kmerlen: int = 31) -> dict[bytes, np.ndarray]:
    """
    Input:
        List of reference genome files
    Output:
        database:
            format: dict[bytes(kmer), np.ndarray(dtype=bool)]
            Each element in np.array represents reference
            bool determines presence/absence of k-mer key
    Note: No labels at this point, so order of genome list must be preserved.
    """
    idx = 0
    database = defaultdict(partial(np.zeros, len(genomes), dtype=bool))
    logger.info("Building database....")
    import mmh3 #test hashed kmers
    SEED = 432

    for genome_file in tqdm(genomes):
        # Open fasta
        with gzopen(genome_file) as gf:
            print(f"Genome file: {genome_file}")

            # Parse every sequence, but all kmer hits fall under genome_file
            for record in SeqIO.parse(gf, "fasta"):
                with memoryview(bytes(record.seq)) as memseq:
                    for ki in range(memseq.nbytes - kmerlen + 1):
                        kmer = memseq[ki : ki + kmerlen]
                        database[bytes(kmer)][idx] = True
                        # database[mmh3.hash(bytes(kmer),signed=False,seed=SEED)][idx] = True #test hashed kmers
        idx += 1
    return database


def gzopen(file):
    encoding = guess_type(str(file))[1]
    _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open
    return _open(file)


def build_parallel(genome_file, genome_files, full_set):
    """
    Input: List of single-sequence (genome) fasta files
    Full build - functional programming style.
    Grabs each file and generates kmers which are then placed into the
    dictionary.
    Strain ID is appended upon collision
    Output: Database of kmer: strain_hits
    """
    kmerlen = args.kmerlen
    col = genome_files.index(genome_file)
    rows = []
    encoding = guess_type(str(genome_file))[1]
    _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open
    with _open(genome_file) as g:
        for record in SeqIO.parse(g, "fasta"):
            # Include all subsequences under one label
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
    df.columns = [pathlib.Path(f).name for f in df.columns]
    # df.index = [pathlib.Path(f).name for f in df.index]
    # df = pd.DataFrame.from_dict(db, orient="index", columns=strain_list, dtype=bool)
    logger.debug(df)
    print(df.head)
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
    accession_df = pd.read_csv(accession_file, sep="\t").set_index("assembly_accession")

    # Filter out non-uniques
    unique = accession_df.taxid != accession_df.species_taxid
    exists = accession_df.taxid.notna()
    accession_df = accession_df[unique & exists]

    # Write new accession details
    accession_df.to_csv(accession_file, sep="\t")

    return [(p / i) for i in accession_df["local_filename"].to_list()]


def get_genome_names(
    genome_files: list[pathlib.Path], accfile: pathlib.Path
) -> list[str]:
    """Function to go from files -> genome names"""

    accession_df = pd.read_csv(accfile, sep="\t").set_index("assembly_accession")
    genome_names = []

    for gf in genome_files:
        acc = gf.stem[:15]
        # The line below is what messes up when u dont delete genome folder name
        genome_name = accession_df.loc[acc]["organism_name"]  # TODO
        accession_df["infraspecific_name"] = accession_df["infraspecific_name"].astype(
            str
        )
        strain_name = accession_df.loc[acc]["infraspecific_name"]
        strain_name = strain_name.replace("strain=", " ")
        genome_names.append(genome_name + strain_name + " " + acc)

    assert len(genome_files) == len(genome_names)
    return genome_names


def pickle_db(database, fout):
    outfile = fout + ".pkl"
    logger.info(f"Saving database as {outfile}")
    with open(outfile, "wb") as ph:
        pickle.dump(database, ph, protocol=pickle.HIGHEST_PROTOCOL)
    return


def pickle_df(df, filename, method="pickle"):


    outfile = args.out + ".db"
    if method == "pickle":
        df.to_pickle(outfile)
    elif method == "hdf":
        outfile = args.out + ".sdb"
        df.to_hdf(outfile)
        pd.DataFrame().to_hdf
    return


def custom_stuff():
    # TODO: Custom stuff should be done entirely in here
    genome_files = list((p / args.custom).glob("*"))
    genome_names = [gf.stem for gf in genome_files]
    return genome_names, genome_files


def get_mash_dist(genome_files: list[pathlib.Path]):
    # Generate a mash command for each row in the dataframe
    # mash_cmd = ("mash sketch " + true_col + " " + pred_col).str.split()
    # mash_cmd = ["mash sketch " + str(gfile) for gfile in genome_files]
    # mash sketch *fna.gz -p 24 -o mash_all_genomes
    # msh_file = ""
    import shlex

    # Generate sketch
    sketch_file = genome_files[0].with_name("mash_all_genomes")
    mash_cmd = shlex.split(
        "mash sketch "
        + " ".join([str(i) for i in genome_files])
        + " -o "
        + str(sketch_file)
    )
    subprocess.Popen(
        mash_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ).wait()

    dist_file = sketch_file.with_name("mash_table.tsv")
    with dist_file.open("w") as dtable:
        dist_cmd = shlex.split(
            "mash dist "
            + str(sketch_file)
            + ".msh "
            + str(sketch_file)
            + ".msh "
            + "-t"
        )
        p = subprocess.Popen(
            dist_cmd,
            stdout=dtable,
            stderr=subprocess.PIPE,
        )
        r = p.wait()
    mash_df = pd.read_csv(dist_file, index_col=0, sep="\t")
    # mash_df.columns = [pathlib.Path(f).name for f in mash_df.columns]
    # mash_df.index = [pathlib.Path(f).name for f in mash_df.index]
    # print(mash_df)
    return mash_df


def main():
    ### 1. Get the genomes files and names for the database

    # 1a. Custom route
    if args.custom:  # Do custom build
        genome_names, genome_files = custom_stuff()
        accession_summary_file = None

    # 1b. NCBI - download route
    else:
        ngd_kwargs = download_strains()
        genomes_dir, accession_summary_file = (
            ngd_kwargs["output"],
            ngd_kwargs["metadata_table"],
        )
        genome_files = list(genomes_dir.glob("*fna.gz"))

        if args.unique_taxid:  # Filter genome list if only unique taxids desired
            genome_files = unique_taxid_strains(accession_summary_file)

        # Run - Download
        genome_names = get_genome_names(genome_files, accession_summary_file)

    logger.info(genome_names)
    logger.info(f"{len(genome_files)} genomes found.")

    ## 2. Build Database
    if args.cluster:
        o1path = args.out + "clusterreps.tsv"
        o2path = args.out + "clustermembers.tsv"
        mash_distance_table = get_mash_dist(genome_files)
        p2c, c2p, numclusties = cluster_strains(
            mash_distance_table,
            ani_threshold=args.ani_cutoff,
            out1_path=o1path,
            out2_path=o2path,
        )
        database, cluster_ids = build_database2(p2c, c2p)
        genome_names = get_genome_names(cluster_ids, accession_summary_file)

    else:
        database = build_database(genome_files)

    df = build_df(database, genome_names)

    logger.debug(f"{len(database)} kmers in database")
    logger.debug(f"{sys.getsizeof(database)//1e6} MB")
    logger.debug(f"Kmer-building complete, saving db as {args.out+'.db'}.")

    ## 3. Save the results
    pickle_df(df, args.out)

    return


if __name__ == "__main__":
    p = pathlib.Path().cwd()
    # p = pathlib.Path("/media/mladen/kai/sarc/")  # .cwd()
    args = get_args().parse_args()
    print(args)
    main()
