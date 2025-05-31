#!/usr/bin/env python

import argparse
import gzip
import logging
import pathlib

# import pickle # Will be removed if pickle_db function is removed or no longer uses it.
import sys
import csv
import shutil

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
        help="Custom name for output prefix. Final database will be <prefix>.db.parquet.",
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


def parse_assembly_level(input_args: argparse.ArgumentParser) -> str:
    """
    Takes the assembly_level option and selects the all-inclusive optarg for which ncbi-genome-download will use.
    """
    level = input_args.assembly_levels
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


def download_strains(args: argparse.ArgumentParser) -> dict:
    """Calls ncbi-genome-download with some preset arguments:
    Returns:
        genome_directory named based on input information
        exit_code: for ncbi-genome-download checking
    """

    # Build ncbi_genome_download parameters
    assembly_level: str = parse_assembly_level(args)
    download_parameters = ncbi_genome_download_parameters(args, assembly_level)
    dbtype, keywords = parse_ngd_keywords(args, assembly_level)
    download_parameters.update(keywords)

    # Display the results
    print(f"The dbtype is {dbtype}")
    for k, v in download_parameters.items():
        print(f"{k}={v}")

    # Call ncbi-genome-download
    returncode = ngd.download(**download_parameters)

    if returncode:
        raise ConnectionError(
            "ncbi-genome-download did not successfully download the genomes"
        )

    return download_parameters
    # return download_parameters["output"], download_parameters["metadata_table"]


def ncbi_genome_download_parameters(args, assembly_level, filetypes=["fasta"]):
    """
    Generate parameters for ncbi_genome_download
    """
    filetypes = ",".join(filetypes)
    print(f"the filetypes are {filetypes}")

    return {
        "flat_output": True,
        "groups": "bacteria",
        "file_formats": filetypes,
        "section": args.source,
        "parallel": args.procs,
        "assembly_levels": assembly_level,
    }


def parse_ngd_keywords(args, assembly_level):
    """
    Determine which keywords are present and choose foldername
    """
    if args.taxid:
        dbtype = "species"
        keywords = {"species_taxids": args.taxid}
        outdir = f"species_{args.taxid}_{assembly_level}"

    elif args.assembly_accessions:
        dbtype = "custom"
        keywords = {"assembly_accessions": args.assembly_accessions}
        outdir = "custom_accessions"

    elif args.genus:
        dbtype = "genus"
        keywords = {"genera": args.genus}
        outdir = f"genus_{args.genus}"
    else:
        raise ValueError(
            """
        Genome download requires either one of the following:
            1. Species-level taxonomic ID 
            2. Genus 
            3. Accession file contain a list of assembly accesssions 
            """
        )

    keywords["output"] = p / outdir
    keywords["metadata_table"] = p / outdir / "genomesummary.tsv"

    for k, v in keywords.items():
        print(f"keywords key: {k} = {v}")

    return dbtype, keywords


# def download_and_filter_genomes():
#     """Grab all, taxid-only, or custom lists"""

#     # Call ncbi-genome-download
#     genomedir, accfile = download_strains()

#     # Filter genome list if only unique taxids desired
#     if params["unique_taxid"]:
#         return unique_taxid_strains(accfile), accfile

#     return list(genomedir.glob("*fna.gz")), accfile


def cluster_strains(
    df: pd.DataFrame, ani_threshold=0.01, out1_path=None, out2_path=None
) -> tuple[dict[str, list], dict[str, str], int]:
    """
    Input:
        dists: Mash distance table, all v all genomes (mash sketch * -o blah, mash dist blah.msh blah.msh -t > idklol)
        outfile: File that contains each genome, its' cluster, and dist between genome and cluster rep, (tsv)
        out_clust_table: Column of clustered rep_genomes, followed by members to the right, (tsv)
        thr: threshold, default set to 0.01, smaller means size of clusters smallers, so n_clusters approaches n_genomes.

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
        database
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

    def most_members(gi):
        return df_clusters.count(axis=1)[gi]

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


def save_dataframe_db(
    df, filename_prefix: str
):  # Renamed function, removed method argument
    """Saves the DataFrame to a Parquet file."""
    # outfile = filename_prefix + ".db.parquet" # Assuming filename_prefix is args.out
    # The global `args` is used here, which is not ideal but follows existing pattern.
    outfile = args.out + ".db.parquet"
    logger.info(f"Saving DataFrame database to (Parquet format): {outfile}")
    try:
        df.to_parquet(outfile, index=True)
        logger.info("DataFrame database saved successfully to Parquet.")
    except Exception as e:
        logger.error(
            f"Failed to save DataFrame database to {outfile} (Parquet format): {e}"
        )
        raise
    return


def custom_stuff():
    # TODO: Custom stuff should be done entirely in here
    genome_files = list((p / args.custom).glob("*"))
    genome_names = [gf.stem for gf in genome_files]
    return genome_names, genome_files


def get_mash_dist(genome_files: list[pathlib.Path]):
    """
    Compute pairwise genome distances using sourmash (Python implementation).
    This replaces the external 'mash' command with a pure Python approach.
    Requires: pip install sourmash
    """
    try:
        import sourmash
        from sourmash import MinHash, signature
        from sourmash.compare import compare_all
    except ImportError:
        raise ImportError(
            "sourmash is required for get_mash_dist. Install with 'pip install sourmash'."
        )

    ksize = 21  # Mash default is 21, can be parameterized
    n_hashes = 1000  # Number of hashes for sketching
    sigs = []
    names = []
    for genome_file in genome_files:
        mh = MinHash(n=n_hashes, ksize=ksize, is_protein=False)
        with gzopen(genome_file) as gf:
            for record in SeqIO.parse(gf, "fasta"):
                seq = str(record.seq).upper()
                mh.add_sequence(seq, force=True)
        sigs.append(mh)
        names.append(pathlib.Path(genome_file).name)

    # Compute pairwise distances (Jaccard similarity, convert to Mash distance)
    # sourmash compare_all returns a similarity matrix; Mash distance = 1 - similarity
    sim_matrix = compare_all(sigs)
    dist_matrix = 1.0 - sim_matrix
    mash_df = pd.DataFrame(dist_matrix, index=names, columns=names)
    return mash_df


def call_ngd(ngd_args):
    """
    Actually execute ncbi_genome_download
    Input:
        ngd_args: The parameters calculated from the user input
    Procedure:
        1. Check if the directory exists (if yes remove it)
        2. Download the genomes from ncbi

    """

    # Check for existing directory , and if so, remove it
    genome_folder: pathlib.Path = ngd_args["output"]
    try:
        shutil.rmtree(genome_folder)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    # TODO: Add option to do more than just fasta
    return_code = ngd.download(**ngd_args)

    if return_code:
        raise ConnectionError(
            "ncbi-genome-download did not successfully download the genomes"
        )


def get_genome_files(ngd_args: argparse.ArgumentParser) -> list[pathlib.Path]:
    """Returns list of genome paths after NGD successfully downloads them"""
    genomes_folder = ngd_args["output"]
    return list(genomes_folder.glob("*fna.gz"))


def main(args: argparse.ArgumentParser):
    if args.custom is None:
        ngd_args = download_strains(args)
        genome_files = get_genome_files(ngd_args)
        call_ngd(ngd_args)
        summary_table = ngd_args["metadata_table"]

        if args.unique_taxid:
            # Filter genome list if only unique taxids desired
            genome_files = unique_taxid_strains(summary_table)

        genome_names = get_genome_names(genome_files, summary_table)

    elif args.custom:  # Do custom build
        genome_names, genome_files = custom_stuff()
        summary_table = None

    else:
        raise ValueError("uh oh")

    logger.info(genome_names)
    logger.info(f"{len(genome_files)} genomes found.")

    ## 2. Build Database
    clustering = False  # More k-mers for bigger clusters - not functional atm
    if clustering:
        o1path = args.out + "clusterreps.tsv"
        o2path = args.out + "clustermembers.tsv"
        mash_distance_table = get_mash_dist(genome_files)
        p2c, c2p, numclusties = cluster_strains(
            mash_distance_table,
            ani_threshold=0.01,
            out1_path=o1path,
            out2_path=o2path,  # TODO: consider thresholding as a parameter
        )
        database, cluster_ids = build_database2(p2c, c2p)
        genome_names = get_genome_names(cluster_ids, summary_table)

    else:
        database = build_database(genome_files)

    df = build_df(database, genome_names)

    logger.debug(f"{len(database)} kmers in database")
    logger.debug(f"{sys.getsizeof(database) // 1e6} MB")
    logger.debug(
        f"Kmer-building complete, saving db as {args.out + '.db.parquet'}."
    )  # Updated log

    ## 3. Save the results
    save_dataframe_db(df, args.out)  # Call renamed function

    return


if __name__ == "__main__":
    p = pathlib.Path().cwd()
    args = get_args().parse_args()
    main(args)
