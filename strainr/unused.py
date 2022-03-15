#!/usr/bin/env python

import collections
import random

import numpy as np
from Bio import SeqIO


def disambiguate(ambig_hits, prior, selection="multinomial"):  # TODO: not implemented
    """
    Assign a strain to reads with ambiguous k-mer signals
    by maximum likelihood.
    Currently 3 options: random, max, and dirichlet.
    (dirichlet is slow and performs similar to random)
    For all 3, threshold spectra to only include maxima, multiply w/ prior.
    """

    rng = np.random.default_rng()
    new_clear, new_ambig = {}, {}

    for (read, hits) in ambig_hits.items():  # Treshold at max
        belowmax = hits < np.max(hits)
        hits[belowmax] = 0

        # Apply prior
        mlehits = hits * prior

        # Weighted assignment
        if selection == "random":
            select_random = random.choices(range(len(hits)), weights=mlehits, k=1).pop()
            new_clear[read] = select_random

        # Maximum Likelihood assignment
        elif selection == "max":
            select_max = np.argwhere(mlehits == np.max(mlehits)).flatten()
            if len(select_max) == 1:
                new_clear[read] = int(select_max)

        # Dirichlet assignment
        elif selection == "dirichlet":
            mlehits[mlehits == 0] = 1e-10
            select_dirichlet = rng.dirichlet(mlehits, 1)
            new_clear[read] = np.argmax(select_dirichlet)

        elif selection == "multinomial":
            mlehits[mlehits == 0] = 1e-10
            select_multi = rng.multinomial(1, mlehits / sum(mlehits))
            new_clear[read] = np.argmax(select_multi)

        else:
            raise ValueError("Must select a selection mode")

    assert len(ambig_hits) == len(new_clear) + len(new_ambig)
    return new_clear, new_ambig


def resolve_ambig_hits(ambig_hits):  # TODO - not implemented
    """Replace numpy array with index"""
    return {k: int(v[0]) for k, v in ambig_hits.items()}


def build_na_dict(na_hits):  # TODO - not implemented
    return {k: None for k in na_hits}


def write_abundance_file(strain_names, idx_relab, outfile):  # TODO - not implemented
    """
    For each strain in the database,
    grab the relab gathered from classification, else print 0.0
    """
    full_relab = collections.defaultdict(float)

    for idx, name in enumerate(strain_names):
        full_relab[name] = idx_relab[idx]
    if idx_relab["NA"]:
        full_relab["NA"] = idx_relab["NA"]

    with outfile.open(mode="w") as fh:
        for strain, relab in sorted(full_relab.items()):
            fh.write(f"{strain}\t{relab:.9f}\n")
    return


def raw_to_dict(raw_classified):  # TODO: Not currently implemented
    """
    Go from list of tuples (SeqRecord,hits)
    to dict {ReadID:hits}
    """
    return {
        read.id: hits for read, hits in raw_classified if isinstance(hits, np.ndarray)
    }


def single_core_classify(fasta, kmerlen, kmerdb):  # TODO - not currently working
    record_index = SeqIO.index(fasta, "fastq")
    records = (record_index[id] for id in record_index.keys())
    full_results = []
    for seqrecord in records:
        max_index = len(seqrecord.seq) - kmerlen + 1
        matched_kmer_strains = []
        s = seqrecord.seq
        with memoryview(bytes(s)) as seqview:
            for i in range(max_index):
                returned_strains = kmerdb.get(seqview[i : i + kmerlen])
                if returned_strains is not None:
                    matched_kmer_strains.append(returned_strains)
        res = (seqrecord, sum(matched_kmer_strains))
        full_results.append(res)
    return full_results

def init_pool(database, strain_list, klength):
    """ A function useful for multiprocess.set_context("spawn") mode in order to assign global variables 
     However overall slower than previous way when using fork 
     """
    print(f"Initializing process {os.getpid()}")
    global db, strains, kmerlen
    db = database
    strains = strain_list
    kmerlen = klength