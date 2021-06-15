#!/usr/bin/env python

import argparse
from collections import OrderedDict,Counter
import glob
import os
import pickle
import re
import shlex
import subprocess as sp
import sys
import tarfile
import zlib

import time
import pandas as pd
import requests
from multiprocessing import Process, Manager
from multiprocessing import Pool, TimeoutError
from multiprocessing.pool import ThreadPool
from collections import defaultdict
from functools import partial
import numpy as np
from sortedcontainers import SortedDict




def download_ncbi(url, filename, save_path):
    ncbi_prefix = "https://ftp.ncbi.nlm.nih.gov/"
    url = ncbi_prefix + url
    try:
        r_ncbi = requests.get(url, stream=True)
        r_ncbi.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise SystemExit(e)

    filepath = os.path.join(save_path, filename)
    fd = open(filepath, "wb")
    # print("assembly summary found")

    if filename == "taxdump.tar":  # tar.gz specific
        fd.write(zlib.decompress(r_ncbi.content, zlib.MAX_WBITS | 32))
        tar = tarfile.open(filepath).extractall(os.path.dirname(filepath))
        # tar.close()
    else:
        fd.write(r_ncbi.content)
    return None


def pd_assembly(species_tax, tax_path, genomes_folder):
    print(species_tax, tax_path, genomes_folder)
    df = pd.read_csv(
        os.path.join(tax_path, "assembly_summary.txt"),
        skiprows=0,
        delimiter="\t",
        header=1,
    )
    mask1 = df["assembly_level"] == "Complete Genome"
    mask2 = df["species_taxid"] == int(species_tax)
    mask3 = df.taxid != df.species_taxid
    mask1_chrom = df["assembly_level"] == "Chromosome"
    mask1_nc = df["assembly_level"] != "Contig"
    mask1_ns = df["assembly_level"] != "Scaffold"

    # CURATIONS OPTIONS: (from most restrictive to least)
    # TODO
    # Strains w/ taxids only
    print(f"Downloading Complete Genomes strict")
    df = df[(mask1 & mask2 & mask3)]

    # Complete Genomes Only (w/ or without taxid)
    # print(f"Downloading Complete Genomes")
    # df = df[(mask2 & mask1)]

    # Complete Genomes and Chromosomes
    # print(f"Downloading Complete Genomes and Chromosomes")
    # df = df[(mask2 & mask1_ns & mask1_nc)]

    ###ml - used this one a bit
    # CG, Chrom, Scaffolds (Everything but Contigs)
    # print(f"Downloading Complete Genomes, Chromosomes, and Scaffolds")
    # df = df[(mask2 & mask1_nc)]

    # if no contigs and normal species
    # df = df[(mask2 & mask1_nc & mask2)]

    term = df.ftp_path.str.split("/", expand=True).iloc[:, -1]
    df["ftp_path"] = df.ftp_path + "/" + term + "_genomic.fna.gz"
    df["ftp_path"] = df.ftp_path.str.replace("ftp://", "https://")

    summary_file = os.path.join(genomes_folder, species_tax + "summary_file.csv")
    df.to_csv(summary_file, index=False)
    # moving 3rd mask after file saving
    # df = df[(mask1 & mask2 & mask3)]
    return df


def request_genomes(df, genomes_folder):
    for url in df.ftp_path:
        genome_filename = url.split("/")[-1].strip(".gz")
        if os.path.exists(os.path.join(genomes_folder, genome_filename)):
            print(f"The file {genome_filename} exists already")
        else:
            try:
                r = requests.get(url)
            except requests.exceptions.HTTPError as e:
                raise SystemExit(e)

            with open(os.path.join(genomes_folder, genome_filename), "wb") as fd:
                    fd.write(zlib.decompress(r.content, zlib.MAX_WBITS | 32))

    return


def slugify(value):
    """
    from django
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata

    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
    # ...
    return value


def GCF_to_taxa(genomes_folder):
    # for strain names
    # gcf_to_id = dict(df[["# assembly_accession", "infraspecific_name"]].dropna().values)
    # gcf_to_id = {k: str(v).split("=")[-1] for k, v in gcf_to_id.items()}

    # for k, v in gcf_to_id.items():
    #     v = re.sub("[^\w\s-]", "", v).strip()
    #     gcf_to_id[k] = re.sub("[-\s]+", "-", v)

    # # for taxids
    # gcf_to_id = dict(
    # df[['# assembly_accession', 'taxid']].dropna().values)

    # for gcf ids
    # gcf_to_id = {k:k for k,v in gcf_to_id.items()}

    fastafiles = glob.glob(os.path.join(genomes_folder , "GC*"))
    fastafiles2 = [os.path.basename(f)[:15] for f in fastafiles]
    gcf_to_id = {f:f for f in fastafiles2}
    print(f"Found {len(fastafiles2)} genomes.")

    GCF_to_filename = {}
    for fasta in fastafiles:
        gcfterm = os.path.basename(fasta)[:15]
        # grep_terms.append(gcfterm)
        GCF_to_filename[gcfterm] = fasta
    return gcf_to_id, GCF_to_filename


def build_jdb_files(kmerlen, GCF_to_tax, GCF_to_filename, genomes_folder):
    """ Making jellyfish files """
    print("The kmerlength is {}".format(kmerlen))
    for k in GCF_to_tax.keys():
        infile = GCF_to_filename[k]
        outfile = "{}/{}".format(genomes_folder, GCF_to_tax[k])
        print(outfile + ".jdb")
        if not os.path.exists(outfile + ".jdb"):
            jstr = "jellyfish count -m {} -s 100M -t {} -o {}.jf".format(
                kmerlen, infile, outfile
            )
            sp.run(shlex.split(jstr))
            jstr2 = "jellyfish dump {0}.jf".format(outfile)
            with open(outfile + ".jdb", "wb") as out:
                sp.run(shlex.split(jstr2), stdout=out)
            # Sed remove lines save backup to outfile.bak
            jstr3 = "sed -i.bak '/^>[0-9]/d' {0}.jdb".format(outfile)
            sp.run(shlex.split(jstr3))
            os.remove(outfile + ".jf")
            os.remove(outfile + ".jdb.bak")
        else:
            print(outfile, ".jdb found already")

    jdb_files = glob.glob(os.path.join(genomes_folder, "*.jdb"))
    print("Part 3 - Concatenating the databases")
    print("this many jdb files", len(jdb_files))
    return jdb_files

    """ Add kmer to dictionary """


def jellyfish_build(kmerlen, GCF_to_tax, GCF_to_filename, genomes_folder):
    """ using popen instead of run to avoid wait call"""
    print("The kmerlength is {}".format(kmerlen))
    jf_commands = []
    print(GCF_to_tax.items())
    print(GCF_to_filename.items())
    in_out = [ (GCF_to_filename[k], f"{genomes_folder}/{GCF_to_tax[k]}") for k in GCF_to_tax.keys()
    ] # Do this to get GCF instead of strain names etc 
    in_out = [ (GCF_to_filename[k], f"{genomes_folder}/{GCF_to_tax[k]}") for k in GCF_to_tax.keys()
    ] 
    for io in in_out:
        jf_cmd1 = shlex.split(
            f"jellyfish count -m {kmerlen} -s 1000M {io[0]} -t 8 -o {io[1]}.jf"
        )
        jf_cmd2 = shlex.split(f"jellyfish dump {io[1]}.jf")
        jf_cmd3 = shlex.split(f"sed -i.bak '/^>[0-9]/d' {io[1]}.jdb")
        jf_cmd4 = shlex.split(f"sort -o {io[1]}.jdb {io[1]}.jdb") # sort in place
        jf_commands.append((jf_cmd1, io[1], jf_cmd2, jf_cmd3,jf_cmd4))
    return jf_commands


def mp_build(jf_run):
    t0 = time.time()
    outfile = jf_run[1]
    # print(outfile + ".jdb")

    if not os.path.exists(outfile + ".jdb"):
        sp.run(jf_run[0]) #count kmers
        with open(outfile + ".jdb", "wb") as out:
            sp.run(jf_run[2], stdout=out) #write to jf file
        sp.run(jf_run[3]) # sed operation
        sp.run(jf_run[4]) # sort in place
    if os.path.exists(outfile + ".jf"):
        os.remove(outfile + ".jf")
    if os.path.exists(outfile + ".jdb.bak"):
        os.remove(outfile + ".jdb.bak")
    else:
        print(f"{os.path.basename(outfile)} .jdb found already")
    return

    """ Add kmer to dictionary """


def build_kmer_dict_new(jdb_files):
    kdict = defaultdict(list)
    for jf in jdb_files:
        taxid, _ = os.path.splitext(os.path.basename(jf))
        with open(jf, "r") as f:
            for line in f:
                kdict[line.strip()].append(taxid)
    return kdict

def mp_build_kmer_dict_new(jdbfile):
    kd = defaultdict(list)
    taxid, _ = os.path.splitext(os.path.basename(jdbfile))
    with open(jdbfile, "r") as f:
        for line in f:
            kd[line.strip()].append(taxid)
    return kd


def build_kmer_dict(jdb_files):
    kdict = {}
    for jf in jdb_files:
        taxid, _ = os.path.splitext(os.path.basename(jf))
        with open(jf, "r") as f:
            for line in f:
                line = line.strip()
                if line not in kdict:
                    kdict[line] = [taxid]
                else:
                    kdict[line].append(taxid)
    for i,k in enumerate(kdict.keys()):
        if i > 100:
            break
        print(i,'\t',k,kdict[k])
    print('mlmlml')
    return kdict


def mp_build_kmer_dict(jf):
    kd = defaultdict(list)
    taxid, _ = os.path.splitext(os.path.basename(jf))
    with open(jf, "r") as f:
        for line in f:
            kd[line.strip()].append(taxid)
    return kd


def optimize_kmer_dict(kdict):
    """ Sorts, tuples, and unsorts """
    print("Optimizing The Dictionary")
    k1 = {}
    for k, v in kdict.items():
        k1[k] = tuple(v)

    print('Sorting..')
    t0 = time.time()
    k1 = dict(sorted(k1.items()))
    print("Sorting Done.")

    t0 = time.time()
    k1 = SortedDict(k1.items())
    print("Sorting Done.")

    print(time.time() - t0)
    # k1 = dict(sorted(k1.items(), key=lambda kv: len(kv[1]), reverse=True))
    return k1


def filter_kmer_dict(kdict, jdb_files):
    """ Filter out very common k-mers """
    print("Filtering Out Degenerate k-mers..")
    print(f"Initial # of k-mers: {len(kdict)}")
    if not len(jdb_files) == 1:
            
        kmer_uniqueness = Counter([len(v) for v in kdict.values()])
        print(kmer_uniqueness.most_common())
        # num_degen_kmers = len(kdict) - len(kdict_discrim)
        
        print(f"Initial # of k-mers: {len(kdict)}")

        # How many will be tossed out
        bad_kmers = {k:v for k,v in kdict.items() if len(v) == len(jdb_files)}
        print(f"K-mers that are present in 100% of the files: {len(bad_kmers)}")

        good_kmers = {k:v for k,v in kdict.items() if len(v) < len(jdb_files)}
        print(f"K-mers that are NOT present in 100% of the files: {len(good_kmers)}")
        print(f"{len(kdict)} = {len(bad_kmers)} + {len(good_kmers)}")
    else:
        print("Only 1 strain present")
        print("Returning original")
        return kdict

    return good_kmers



def write_json(kdict, genomes_folder):
    import json

    jsonpath = os.path.join(genomes_folder, "strainDB.json")
    with open(jsonpath, "w") as jh:
        json.dump(kdict, jh)
    return


def write_pickle(kdict, species_tax, kmerlen, genomes_folder):
    pickle_name = "b" + species_tax + "_k" + kmerlen + ".pickle"
    pickpath = os.path.join(genomes_folder, pickle_name)
    with open(pickpath, "wb") as ph:
        pickle.dump(kdict, ph, protocol=pickle.HIGHEST_PROTOCOL)

    # Convert to pandas
    # df = pd.DataFrame.from_dict(kdict,orient='index',dtype='category')
    # df.columns =  ['col'+str(i) for i in df.columns]
    # Write to disk
    # df.to_pickle(pickpath)
    # return

    return

    """ ---------------------------------------------- """
    # old - look in assembly summary to get gcf to tax dictionary

    jelly_path = jellyfolder
    jelly_files = [
        os.path.join(speciesdb, "jelly", f)
        for f in os.listdir(jelly_path)
        if f.endswith(".jdb")
    ]
    kdict = {}
    # print(jelly_files)
    print("this many jdb files", len(jelly_files))

    """ Add kmer to dictionary """

    for jf in jelly_files:
        with open(jf, "r") as f:
            # Note, file_extension is never used, delete it?
            taxid, file_extension = os.path.splitext(jf)
            line = f.readline()
            # Go through every line in the file
            for line in f:
                line = line.strip()
                # If this is a unique line, add to dict
                if line not in kdict:
                    kdict[line] = [taxid]
                # If this is already here, append the taxid to it
                # TODO:defaultdict
                else:
                    kdict[line].append(taxid)


# analysis stuff
# taxlist = list(kdict.values())
# kmer_hist = [i for alist in taxlist for i in alist]
# kmer_len_hist= [len(i) for i in taxlist]
# kh = collections.Counter(kmer_hist)
# khl = collections.Counter(kmer_len_hist)


def main():

    os.chdir(BASEDIR)
    print(BASEDIR, "\t", species_tax, "\t", kmerlen)

    genomes_folder = BASEDIR

    # tax_suffix="pub/taxonomy/taxdump.tar.gz"
    # download_ncbi(tax_suffix,'taxdump.tar',tax_path)

    # TODO: make arg
    # print(f"\n\nUsing refseq genomes")
    # assembly_summary = 'genomes/refseq/bacteria/assembly_summary.txt'

    # print(f"Using genbank genomes")
    # assembly_summary = "genomes/genbank/bacteria/assembly_summary.txt"

    # step 1
    # download_ncbi(assembly_summary, "assembly_summary.txt", tax_path)

    # step 2
    # print(f"Extracting Genomes")
    # df = pd_assembly(species_tax, tax_path, genomes_folder)


    # print(f"Extracting species of interest genomes ")
    # # step 3
    # request_genomes(df, genomes_folder)

    # step 4
    gcf_tax_dict, gcf_filename_dict = GCF_to_taxa(genomes_folder)

    # step 5
    jf_commands = jellyfish_build(
        kmerlen, gcf_tax_dict, gcf_filename_dict, genomes_folder
    )

    tp = time.time()
    with Pool(processes=16) as pool:
        pool.map(mp_build, jf_commands)
    print(f"{time.time()-tp}s to build jdb files")

    debug = False 
    if debug:
        print("DEBUG MODE IS ON RETURNING EARLY")
        return

    jdb_files = glob.glob(os.path.join(genomes_folder, "*.jdb"))


    # Build Dictionary - defaultdict
    t0 = time.time()
    kdict= build_kmer_dict_new(jdb_files)
    print(f"Took {time.time()-t0} seconds")

    
    if kdict:
        print(len(kdict))

    print(f"The DB contains {len(kdict)}({sys.getsizeof(kdict)/1000000} MB) kmers")
    print(f"Filtering the DB..")
    kdict = filter_kmer_dict(kdict, jdb_files)  # can do this in classify
    print(f"Done filtering the DB")

    # Sort and make values constant
    print(f"Optimizing the DB")
    kdict = optimize_kmer_dict(kdict)

    # Make binary
    kdict={k.encode():v for k,v in kdict.items()}

    # Write output
    print(f"Saving the DB")
    write_pickle(kdict, species_tax, kmerlen, genomes_folder)

    print(f"Cleaning up ")
    for i in jdb_files:
        os.remove(i)
    return


if __name__ == "__main__":
    t00 = time.time()
    print(f"This is main hi")
    print(f"Please provide [basedirectory,species_taxid,kmerlen]")
    print(f"Example: python StrainDatabase.py mystraindb_folder 727 31")
    BASEDIR = os.path.abspath(sys.argv[1])
    species_tax = sys.argv[2]
    kmerlen = sys.argv[3]
    main()
    print(f"all done!")
    print(f"Database built in {time.time() - t00} seconds.")


