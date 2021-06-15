#!/usr/bin/env python
import argparse
import functools

# from genericpath import exists
import glob
import gzip
import itertools
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from mimetypes import guess_type
from multiprocessing import Pool,Process
from pathlib import Path
from typing import (
    Any,
    Generator,
    Iterator,
    List,
    Tuple,
)

# import numpy as np
import pandas as pd
from statistics import multimode

# from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.SeqIO.QualityIO import FastqGeneralIterator
import logging


# try this idk
# os.system("taskset -p 0xff %d" % os.getpid())


def build_logger():
    # set up logging to file - see previous section for more details
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename="/tmp/mlrad_strain.log",
        filemode="w",
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)
    logger = logging.getLogger("simple_example")
    return logger


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", type=str, nargs="+", help="List of fwd input files")
    ap.add_argument("-d", "--db", type=str, required=True)
    ap.add_argument("-p", "--procs", help="", type=int, default=1)
    ap.add_argument("-o", "--output", type=str)
    ap.add_argument("-w", "--writefiles", action="store_true")
    ap.add_argument("-a", "--athresh", help="", type=float, default=0.01)
    ap.add_argument("-t", "--ftype", choices=["fasta", "fastq"], type=str)
    ap.add_argument("-k", "--kmerlen", help="", type=float, default=31)
    ap.add_argument("-j", "--reverse_file", type=str)
    ap.add_argument("--paired", type=bool, default=False, help="Paired Reads")
    ap.add_argument(
        "-m",
        "--mtpc",
        help="Max tasks per child, set higher for increased speed with additional memory usage",
        type=int,
        default=None,
    )
    ap.add_argument(
        "-n",
        "--minhits",
        help="Minimum number of unique hits to be considered for binning",
        type=int,
        default=1000,
    )
    ap.add_argument(
        "-s",
        "--max_strain_bins",
        help="Maximum number of strain bins, where [n] represents the n strains with the most hits",
        type=int,
        dest="max_strain_bins",
        default=5,
    )
    # Things to do later
    #
    # ap.add_argument('write_strain_fastas', type=open)
    # ap.add_argument('source_file', type=open)
    # ap.add_argument('dest_file', type=argparse.FileType('w', encoding='latin-1'))
    # ap.add_argument('datapath', type=pathlib.Path)

    # Initalize and Return
    return ap.parse_args()




def save_maps(
    outdir: str, pfile: str, reads_map: dict, _pl: int = pickle.HIGHEST_PROTOCOL
):
    """Save the {Strain : Reads} dictionaries for later use"""
    pfile = os.path.join(BASE, outdir, pfile)
    with open(pfile, "wb") as pf:
        pickle.dump(reads_map, pf, protocol=_pl)
    return


def sample_cardinality(output, reads_dict):
    """
    1. Sorts the results by length
    2. Distribution of uniqueness
    3. Clarifies Ambig dict
    """
    uniqueness_dist = defaultdict(list)

    # Sort reads by length of hits
    reads_dict = dict(sorted(reads_dict.items(), key=lambda k: len(k[1])))

    # Histogram of read uniqueness
    for k, v in reads_dict.items():
        uniqueness_dist[len(v)].append(k)

    # Print the Histogram
    out = os.path.join(output, "hist.csv")
    with open(out, "w") as wh:
        for k, v in uniqueness_dist.items():
            print(k, len(v), file=wh)

    # TODO: Note, NAs are counted under 1 here,
    # but shouldn't be

    # now sorted
    return reads_dict


def separate_hits(taxdict):
    """
    After all the queries have been completed,
    split up taxdict into 3 different locations
    """
    onehit, multihit, nohit = {}, {}, {}
    miss = "X"

    for read, hitlist in taxdict.items():
        # hitlist of length 0 cant be indexed
        if not len(hitlist) or (len(hitlist) == 1 and hitlist[0] == miss):
            nohit[read] = tuple(miss)
        # unique-hit
        elif len(hitlist) == 1 and hitlist[0] != miss:
            onehit[read] = hitlist
        # multi-hit
        elif len(hitlist) > 1:
            multihit[read] = tuple(v for v in hitlist if (v != miss or v != tuple()))
        # error
        else:
            raise Exception(
                f"{read} the key for {hitlist}, should be length 1 and either contain a strain or unclassifed"
            )

    # print(f"Uniques: {len(onehit)}, \n Ties: {len(multihit)}, \n NA: {len(nohit)}")
    return onehit, multihit, nohit


def normalize_dist(countdict):
    total = sum(countdict.values())
    # TODO: test out new dict
    # return { k: v/total for k,v in countdict.items() }
    for el in countdict.keys():
        countdict[el] /= total
    return countdict


def output_abundance(strain_list, dt_abund, filepath):
    """This will print the abundance profile based on all of the strains"""
    strain_dict = show_all_strains(dt_abund, strain_list)
    # with #open(filepath, "w") as f:

    with filepath.open("w") as f:
        for k, v in sorted(strain_dict.items(), key=lambda kv: kv[1]):
            print(k, "\t", round(v, 4), file=f)
    return


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def show_all_strains(relab, strain_ll):
    strain_dict = defaultdict(float)
    for i in strain_ll:
        strain_dict[i] = relab[i]
    return {k: v for k, v in sorted(strain_dict.items())}
    # , key=lambda val: val[1], reverse=True) }


def group_all_reads(rd, ad, nd):
    """
    * Collect the resolved, ambiguous and NA into 1 dict object *

    Note: main script has above and below in the final_reads
    and below in the na reads, so not super clear but
    not necessary yet

    """
    final_reads = {}
    if rd:
        final_reads.update(rd)
    if ad:
        final_reads.update(ad)
    if nd:
        final_reads.update(nd)
    return final_reads


### Utils ########
def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def dbtake(n, db):
    "Return a slice of the dict"
    dbh = {i: db[i] for i in list(itertools.islice(db, n))}
    return dbh


def hashk(kmer: str):
    base_fwd = "ACGTN"
    base_rev = "TGCAN"
    comp_tab = str.maketrans(base_fwd, base_rev)
    rc_kmer = kmer.translate(comp_tab)[::-1]
    if rc_kmer < kmer:
        kmer = rc_kmer
    return kmer



def omni_open(infile):
    encoding = guess_type(str(infile))[1]  # uses file extension
    _open = functools.partial(gzip.open, mode="rt") if encoding == "gzip" else open
    file_object = _open(
        infile,
    )
    return file_object




def merge_paired(g1, g2):
    for a, b in zip(g1, g2):
        new_seq = a.seq + "N" + b.seq
        yield SeqIO.SeqRecord(id=a.id,seq=new_seq)


def preprocess_reads(infile, input_ext, reverse_file):
    """Allows for fastq v fasta, paired v unpaired, and zipped vs unzipped"""
    fastq_condition = (
        input_ext.endswith("q")
        or input_ext.endswith("q.gz")
        or input_ext.endswith("q.gzip")
    )
    fasta_condition = (
        input_ext.endswith("a")
        or input_ext.endswith("a.gz")
        or input_ext.endswith("a.gzip")
    )
    if fastq_condition:
        ft = "fastq"
    elif fasta_condition:
        ft = "fasta"
    else:
        raise ValueError("Neither fastq nor fasta file detected from filesuffix")
    f1 = omni_open(infile)
    f2 = omni_open(infile)

    g1 = SeqIO.parse(f1, ft)
    if reverse_file:
        g2 = SeqIO.parse(f2, ft)
        records = merge_paired(g1, g2)
    else:
        records = g1
    return records




def sfetch_kmers(seqrecord):
    max_index = len(seqrecord.seq) - KMERLEN + 1
    # Fetch records
    matched_kmer_strains = []
    with memoryview(bytes(seqrecord.seq)) as seq_buffer:
        for i in range(max_index):
            kmer = seq_buffer[i:i+KMERLEN]
            returned_strains = kmer_database.get( kmer, tuple())
            matched_kmer_strains.append(returned_strains)

    # Grab most frequent strains
    if matched_kmer_strains:
        top_hits = multimode(i2 for i in matched_kmer_strains for i2 in i)
        return seqrecord.id,tuple(top_hits)
    else: 
        return seqrecord.id, tuple(tuple("X"))





def mp_classify(infile, input_ext, reverse_file=None, nproc=8):
    """
    Parallelizes db lookup using multiprocessing
    """
    records = preprocess_reads(infile, input_ext, reverse_file)
    t0 = time.time()
    if nproc == 1:
        print(f"\tSingle-core classification ")
        reads_postlookup = dict(map(sfetch_kmers, records))

    else:  # Multiprocessing
        print(f"\tMultiprocessing classification with {nproc} processors")
        with Pool(processes=nproc, maxtasksperchild=mtpc) as pool:
            reads_postlookup = dict(
                pool.imap_unordered(sfetch_kmers, records, chunksize=1000)
            )
    tf = time.time()

    print(f"\tClassification complete using [{nproc}] cores is {tf-t0}")
    print(f"\t Processed {len(reads_postlookup)} reads")

    return reads_postlookup


def generate_initial_probability(clear_hits, ambig_reads):
    print(f"Generating initial probability estimate of strains")
    #### Building the prior
    prior_model: str = "unique"

    if prior_model == "unique":
        unique_hit_distribution = Counter([v[0] for v in clear_hits.values()])
        normalized_prob = normalize_dist(unique_hit_distribution)

    elif prior_model == "unique_ambig":  # Not utilized for now
        # Put clear and ambigous lists into single dict
        multi_hits = clear_hits.copy()
        multi_hits.update(ambig_reads)

        multi_hits_distribution = Counter(
            [
                each_strain
                for each_hit in multi_hits.values()
                for each_strain in each_hit
            ]
        )
        normalized_prob = normalize_dist(multi_hits_distribution)
    else:
        print("Incorrect Model selected, switching to default")
        prior_model = "unique"
        unique_hit_distribution = Counter([v[0] for v in clear_hits.values()])
        normalized_prob = normalize_dist(unique_hit_distribution)

    print_relab(normalized_prob)
    print(f"Initial model complete, updating ambiguous reads..")
    return normalized_prob


def mp_break_ties(kcount: list, prior: dict) -> List[Any]:
    """Intersect prior probability for all strains with available strains in that read and return the maximum strain"""
    strain_to_prob = {strain_id: prior[strain_id] for strain_id in kcount}
    # Select max choice only
    selection_max = (max(strain_to_prob, key=strain_to_prob.get),)

    # random choice
    if sum(strain_to_prob.values()) > 0:
        selection_randweighted = tuple(
            random.choices(
                list(strain_to_prob.keys()), weights=strain_to_prob.values(), k=1
            )
        )
    else:
        selection_randweighted = selection_max

    return (
        # selection_randweighted
        selection_randweighted
    )


def mp_disambiguate(clear_hits: dict, ambig_reads: dict, nproc: int):
    # ) -> Tuple[dict, dict, Counter[str]]:
    """
    Disambiguate the remaining strains
    """
    t0 = time.time()

    # Generate initial probability from reads with clear strain signal
    normalized_prob = generate_initial_probability(clear_hits, ambig_reads)

    # Multiprocess to get maximum likelihood strain based on intersection of read's top strains with probability distribution
    resolved_reads = {}
    maximize_strain_probability = functools.partial(
        mp_break_ties, prior=normalized_prob
    )
    if nproc == 1:
        for read, argmax_strain in zip(
            ambig_reads.keys(), map(maximize_strain_probability, ambig_reads.values())
        ):
            resolved_reads[read] = argmax_strain

    else:
        with Pool(processes=nproc) as pool:
            for read, argmax_strain in zip(
                ambig_reads.keys(),
                pool.map(maximize_strain_probability, ambig_reads.values()),
            ):
                resolved_reads[read] = argmax_strain

    new_ambig = {}
    clear_hits.update(resolved_reads)

    print(f"Tie-breaking complete")
    print(f"The time for {nproc} procs for tie_breaking is {time.time() - t0}")

    return clear_hits, new_ambig, normalized_prob


def calculate_threshold(res_reads, thresh_val, prior_prob):
    NAstring = "X"

    # TODO
    hits_only_reads = {k: v for k, v in res_reads.items() if v[0] != NAstring}
    na_reads = {k: v for k, v in res_reads.items() if v[0] == NAstring}

    # na_reads.update?

    """ Threshold """
    # TODO: Need to reclassify these somewhere.
    # Thresh based on prior probability

    above_thresh_reads = {
        k: v for k, v in hits_only_reads.items() if prior_prob[v[0]] >= thresh_val
    }

    below_thresh_reads = {
        k: [NAstring]
        for k, v in hits_only_reads.items()
        if prior_prob[v[0]] < thresh_val
    }

    num_above = len(above_thresh_reads)
    num_below = len(below_thresh_reads)

    print(f"\n\nThresholding-")
    print(f"Above threshold reads: {num_above}")
    print(f"Below threshold reads: {num_below}")
    print(
        f"{num_above} hits, {num_below} reads assigned strains below {thresh_val}, removing."
    )

    return above_thresh_reads, below_thresh_reads


def print_relab(posterior_prob, nstrains=10):
    """Displays Top 10 relative abundance strains"""
    print(f"\nRelative Abundance Estimation")
    for k, v in posterior_prob.most_common(n=nstrains):
        print(k, "\t", round(v, 5))
    print(f"\n")


def printout_lengths(res_reads, ambig_reads, na_reads):
    print(f"{len(res_reads)} hits, {len(ambig_reads)} ambiguous, {len(na_reads)} X")


def get_strain_list(reads_mcs):
    strain_list = list(set([z for y in reads_mcs.values() for z in y]))
    return strain_list


def results_pivot_table(mydict: dict):
    mydict = {k: v[0] for k, v in mydict.items()}
    df = pd.Series(mydict).reset_index()
    df.columns = ["reads", "strains"]
    df = df.set_index("reads")
    df["values"] = int(1)
    # print(df)
    df = df.pivot(columns="strains", values="values")
    return df


def fresults_pivot_table(inputfile):
    df = pd.read_csv(inputfile, sep="\t", header=0)
    df.columns = ["reads", "strains"]
    df = df.set_index("reads")
    df["values"] = int(1)
    df = df.pivot(columns="strains", values="values")
    return df


def get_db_kmerlen(kmer_database):
    return len(kmer_database.popitem()[0])


def load_kmerdb(dbpath: str):
    print(f"Loading kmer database...")
    t = time.time()
    with open(dbpath, "rb") as f:
        kmer_database = pickle.load(f)
    print(f"Database loaded in {time.time() - t} seconds")
    return kmer_database


def hash_kmer(kmer):
    # calculate the reverse complement
    base_fwd = "ACGT"
    base_rev = "TGCA"
    comp_tab = str.maketrans(base_fwd, base_rev)
    rc_kmer = kmer.translate(comp_tab)[::-1]
    # determine whether original k-mer or reverse complement is lesser
    return kmer if kmer < rc_kmer else rc_kmer


def grab_output_dir():
    """Overly complicated way to get a dir no matter what"""
    output_folders = glob.glob("wevote_output*")
    standard = params.output
    standard = "wevote_output{ival}"
    if standard not in output_folders:
        os.mkdir(standard)
        return standard
    else:
        out = standard + "01"
        os.mkdir(out)
        return out


def strains_as_keys(reads2strains_dict: dict):
    """strains -> read.id dictionary"""
    strain_dict = defaultdict(list)
    for r, slist in reads2strains_dict.items():
        for ss in slist:
            strain_dict[ss].append(r)
    return strain_dict
    # return { strain : for read, strain_set in
    #         reads_as_keys.items() for strain in strain_set }


def create_newdir(output: str):
    if Path(output).exists():
        print("Directory already exists..")
        renamed_old = f"{output}_old{random.randint(1,1000)}"
        Path(output).rename(renamed_old)
        print(f"Renamed old directory to {renamed_old}..")
    outpath = Path(output).resolve()
    outpath.mkdir()
    return outpath


def write_strain_fastas(
    strain_dict: dict,
    input_file: str,
    outdir: str,
    reverse_file: str = None,
    minreads: int = 500,
    fastadirname: str = "fastas",
):
    """
    Write fasta files selected classified reads for assembly
    Uses output_prefix as the directory to hold files
    Use this for post-processing.

    Args:
        strain_dict ([dict]): [Set of Reads mapped to a strain]
        input_file ([file]): [Original Fasta/q file]
        fext ([string]): [fasta or fastq]
        unique_reads_only (bool, optional): [Whether fastas readsets are mutually exclusive]. Defaults to False.

    Returns:
        [dict]: [original strain_dict]
    """

    fastadir = os.path.join(outdir, fastadirname)

    if not os.path.exists(fastadir):
        os.mkdir(fastadir)

    # Sort strains by number of reads and select only the top n
    max_n = params.max_strain_bins  # if params.max_strain_bins else None
    print(f"Generating sequence files for the top {max_n} strains.")
    strain_dict = dict(sorted(strain_dict.items(), key=lambda item: item[1])[:max_n])

    # Loop through each strain to make the file
    procs, saved_strains = [], []
    for strain_id, read_set in strain_dict.items():
        if strain_id == "X":
            print("Skip Unclassifieds")
            continue

        if len(read_set) > minreads:
            saved_strains.append(strain_id)
            print(f"Writing {len(read_set)} reads for {strain_id}...")
            # filetype = (
            #     input_file.split(".")[-2]
            #     # if input_file.endswith("z")
            #     # else input_file.split(".")[1]
            # )
            filetype = "fastq"  # TODO mlml need to make this viable lol
            p = Process(
                target=mp_write_single_strain,
                args=(
                    strain_id,
                    read_set,
                    filetype,
                    input_file,
                    fastadir,
                    reverse_file,
                ),
            )
            p.start()
            procs.append(p)

    # [p.join() for p in procs]
    return set(saved_strains), procs


def mp_write_single_strain(
    strain_id: str, read_set, fext: str, r1file: str, fastadir: str, r2file: str
):

    encoding = guess_type(str(r1file))[1]  # uses file extension
    _open = functools.partial(gzip.open, mode="rt") if encoding == "gzip" else open

    # Loop through R1 and R2
    for i, readsfile in enumerate([r1file, r2file]):

        if i == 1:
            # Change read_sets to match /1 /2 in file (reads currently named after R1 file)
            read_set_i = set(
                read_id.replace("/1", "/2") for read_id in read_set
            )  # For 1:N:0:1
            # read_set_i = set(read_id[:-1] + str(i + 1) for read_id in read_set)
            # read_set_i = set( read_id.replace("1:N", "2:N") for read_id in read_set)  # For 1:N:0:1
            # read_set_i = set( (read_id + " /" + str(i+1) ) for read_id in read_set)  # For 1:N:0:1
            # read_set_i = set(read_id for read_id in read_set)
        else:
            read_set_i = set(read_id for read_id in read_set)

        # Strain_R1.fastx or Strain_R2.fastx
        wbase = f"{strain_id}_R{i+1}.{fext}"  # TODO - for fastas and gzip context
        writefile = os.path.join(fastadir, wbase)

        # Open input, find reads that match dict, write to strain_fasta
        with _open(readsfile) as rhandle, open(writefile, "w") as whandle:

            if fext.endswith("q"):  # Fastq
                for (rid, seq, q) in FastqGeneralIterator(rhandle):
                    if rid in read_set_i:
                        print(f"@{rid}\n{seq}\n+\n{q}", file=whandle)
            else:  # Fasta
                for rid, seq in SimpleFastaParser(rhandle):
                    if rid in read_set_i:
                        print(f">{rid}\n{seq}", file=whandle)

    return


def arg_main(
    inputfile, inputsuffix, procs=16, reverse_file=None, athresh=0.00, output=None
):
    t0 = time.time()

    reads_mcs = mp_classify(
        inputfile, inputsuffix, reverse_file=reverse_file, nproc=procs
    )
    print("Finished classifying..\n")

    # Get all possible strain labels
    strain_list = get_strain_list(reads_mcs)

    res_reads, ambig_reads, na_reads = separate_hits(reads_mcs)

    print("\nUpdating estimates..\n")
    res_reads, ambig_reads, prior_prob = mp_disambiguate(
        res_reads, ambig_reads, nproc=procs
    )

    # if len(ambig_reads):
    #     print(f"Cannot resolve {len(ambig_reads)} ambiguous reads")
    #     na_reads.update(ambig_reads)

    print(f"Final distribution of hits/ties/miss:\n")
    printout_lengths(res_reads, ambig_reads, na_reads)

    all_reads = {}
    dict_list = [res_reads, ambig_reads, na_reads]
    strlist = ["Hits:", "Ambiguous", "NA"]

    print(f"\t Overall Stats: Read summary:")
    for i, read_dict in enumerate(dict_list):
        print(f"{strlist[i]}: \t { len(read_dict) } hits")
        all_reads.update(read_dict)
    print(f"Total: \t {len(all_reads)} reads processed")

    all_reads = dict(sorted(all_reads.items(), key=lambda kv: len(kv[0][0])))

    print("\nCalculating threshold")
    above_threshr, below_threshr = calculate_threshold(res_reads, athresh, prior_prob)
    intra_relab = normalize_dist(
        Counter([v[0] for v in above_threshr.values()])
    )  # removes list

    # Preprocess output stuff
    # TODO: figure out with print
    above_threshr.update(below_threshr)  # (above) <- (below)
    res_reads = above_threshr  # res = (above + below)
    na_reads.update(below_threshr)  # below -> na
    print(f" Has been {time.time() - t0}, finished threshold ")

    # 5. Finalizing ===============================
    final_reads = group_all_reads(res_reads, ambig_reads, na_reads)

    print(f"Final # of hits per strain")
    relab_count = Counter([v[0] for v in final_reads.values()])
    print_relab(relab_count)

    print(f"Relative abundance:  (over all reads)")
    relab_w_unclass = normalize_dist(Counter([v[0] for v in final_reads.values()]))
    print_relab(relab_w_unclass)  # print

    print(f"Intra-species abundance:  (over all hits)")
    print_relab(intra_relab)

    # 5. Output Directory===============================
    print("Output Stuff")
    if output:
        output = create_newdir(output)  # from str to path
        reads_mcs = sample_cardinality(output, reads_mcs)  # hist.txt

        # p_out = Path(output)
        # abundance_file = "abundance.tsv" #+ p_out.stem + ".tsv"

        abundance_file = output / "abundance.tsv"
        output_abundance(strain_list, relab_count, abundance_file)

        abundance_file2 = output / "abundance2.tsv"
        output_abundance(strain_list, intra_relab, abundance_file2)

        # for paper
        statsfile = output / "stats.tsv"
        calc_tp_fp_fn(final_reads, statsfile)

        print(f"Saving run data.")
        minhits = params.minhits
        # Save reads-> strains and strains-> reads mapping for intermediate hits
        sr_all = strains_as_keys(reads_mcs)  # All inclusive
        sr_unique = strains_as_keys(final_reads)  # Not all inclusive
        sr_nonunique = {
            k: v for k, v in sr_all.items() if (len(sr_unique[k]) > minhits)
        }

        # Dump all read-mappings to pickles in output folder
        savemap_filelist = [
            "read2s-final.pkl",
            "read2s-all.pkl",
            "strain2r-all.pkl",
            "strain2r-final.pkl",
        ]
        savemap_varlist = [final_reads, reads_mcs, sr_all, sr_unique]
        for fname, varname in zip(savemap_filelist, savemap_varlist):
            save_maps(output, fname, varname)

        if writefiles:

            xstrain_set, xplist = write_strain_fastas(
                sr_unique,
                inputfile,
                output,
                reverse_file=reverse_file,
                minreads=minhits,
                fastadirname="xfastas",
            )

            # minhits is useless here
            astrain_set, aplist = write_strain_fastas(
                sr_nonunique,
                inputfile,
                output,
                reverse_file=reverse_file,
                minreads=minhits,
                fastadirname="afastas",
            )

        # call processes to finish
        [xp.join() for xp in xplist]
        [ap.join() for ap in aplist]

        # assert len(astrain_set) == len(xstrain_set)

    print("Done.")
    print(f"Run completed in {time.time() - t0} seconds")

    return


def calc_tp_fp_fn(final_reads, statsfile):
    tp, fp, fn, nono = 0, 0, 0, 0
    for k, v in final_reads.items():
        res = v[0]
        if res.startswith("G") and res in k:
            tp += 1
        elif res.startswith("G") and res not in k:
            fp += 1
        elif not res.startswith("G"):
            fn += 1
        else:
            nono += 1
    print(k, v)

    with open(statsfile, "w") as f:
        print(f"tp\tfp\tfn\tnono", file=f)
        print(f"{tp}\t{fp}\t{fn}\t{nono}", file=f)
    return


# MAIN HERE


if __name__ == "__main__":
    p = Path.cwd()
    BASE = p.resolve()  # TODO
    # logger = build_logger()

    # Get params

    if len(sys.argv) > 3:
        params = get_args()

        if params.writefiles and not params.reverse_file:
            raise argparse.ArgumentParser().error(
                "Parameter Dependence error: Binned file generation requires paired files."
            )
        writefiles = params.writefiles

        # Input and Output
        f1 = Path(params.input_file[0])
        # f2 = f1.parent / ( f1.name.replace('R1','R2') ) if params.reverse_file else None
        f2 = params.reverse_file
        results_folder = params.output if params.output else None
        fext = "fastq" if any(ext.endswith("q") for ext in f1.suffixes) else "fasta"

        # TODO: make multiple items in list work

        # Multiprocessing and Threshold
        nproc = params.procs if params.procs else 4
        threshold_value = params.athresh
        mtpc_default = 48000 // nproc
        mtpc = params.mtpc if params.mtpc else mtpc_default
        print(f"MTPC: {mtpc}")
        # Load Database
        kmer_database = load_kmerdb(params.db)
        KMERLEN = get_db_kmerlen(kmer_database)
        # pool = Pool(processes=nproc, maxtasksperchild=mtpc)

        # Call main
        arg_main(
            f1,
            fext,
            procs=nproc,
            reverse_file=f2,
            athresh=threshold_value,
            output=results_folder,
        )

    else:  # Custom
        # ecoli_dict = "/home/mladen/wvstrain/Assessment/fixed_genbank_databases/562_genomes/mini.10th.562_k31.pickle" # Fast debug
        # fileglob = glob.glob("/home/mladen/antibiotics/atlas2/input_originals/M*R1_001.fastq.gz")
        # fileglob = glob.glob("/hdd/abx_fresh/LungStudy/Run160_161/qc_inputs/M*R1.fastq.gz")
        # fileglob = glob.glob("/hdd/ecoli_outbreak_run/inputfiles/trimmed_data/*_R1_trimmed.fastq.gz")
        fileglob = glob.glob(
            "/hdd/ecoli_outbreak_run/inputfiles/trimmed_data/*_R1_trimmed.fastq.gz"
        )
        fileglob = glob.glob("/hdd/diabbimmune_t2d/G*1.fastq.gz")
        # taxid_dict = f"/hdd/diabbimmune_t2d/databases/33038_genomes/b33038_k31.pickle"

        # CHANGE THESE
        taxid = 562  # TODO mlml
        ncbi_source = "genbank"  # TODO mlml refseq or genbank
        fileglob = glob.glob("/hdd/ecoli2/c*1_final.fastq.gz")
        nproc = 16
        mtpc = 30
        cutoff_value = 0.0
        writefiles = True
        out_prefix = "strainr_"
        r1str = "1_final"
        r2str = "2_final"
        minhits = 1000
        max_strain_bins = 5

        # END CHANGE

        taxid_dict = f"/home/mladen/wvstrain/Assessment/fixed_{ncbi_source}_databases/{taxid}_genomes/b{taxid}_k31.pickle"
        kmer_database = load_kmerdb(taxid_dict)
        KMERLEN = get_db_kmerlen(kmer_database)

        t0 = time.time()
        print(f"{len(fileglob)} files to process")

        for i, f1 in enumerate(fileglob):
            f1 = Path(f1)  # ? mlml
            # f2 = f1.with_name(f1.name.replace("R1", "R2"))
            f2 = f1.with_name(f1.name.replace(r1str, r2str))
            outdir = p / (out_prefix + f1.stem)
            print(f"\nInput: {f1.stem} + {f2.stem}")
            print(f"\nTime so far: {time.time() - t0}")
            print(f"\n{i} out of {len(fileglob)} completed.")
            if not os.path.exists(outdir):
                with Pool(processes=nproc, maxtasksperchild=mtpc) as pool:
                    try:
                        arg_main(
                            f1,
                            "fastq",
                            procs=nproc,
                            reverse_file=f2,
                            athresh=cutoff_value,
                            output=outdir,
                        )
                        # no ouput`
                        # arg_main( f1, "fastq", procs=nproc, reverse_file=f2, athresh=cutoff_value,)

                    except FileExistsError as e:
                        print(f"An exception occured with {f1.stem}")
                        sys.exit()
            else:
                print(f"{outdir} already exists... skipping.")

    # TODO: 1. fileglob,
    # 2. outdir,
    # 3. DB taxid + source,
    # 4. nproc,
    # 5. maxtasksperchild,
    # 6. write or nowrite,
    # 7. p.join or nah

############################################################

# taskcap = 48// nproc
# f.name of f.base now
# f1base, f2base = os.path.basename(params.input_file),os.path.basename(params.reverse_file)

# used to go right before pool
# logger.debug(f"Fext = {fext}, prcs = {nproc}, tasks = {taskcap}")

# TODO - list
# elif len(params.input_file) > 1: #list
#     f1 = [Path(i) for i in params.input_file][0] if len(params.input_file) > 1 else Path(params.input_file[0])
# if params.reverse_file:
#     f2 =  [Path(i.name.replace('R1','R2')) for i in f1][0]
#     # f2 = Path(f1.name.replace('R1','R2'))
# else: f2 = None
############################################################

# Looped version for me below mlml
