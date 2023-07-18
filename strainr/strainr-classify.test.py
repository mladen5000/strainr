# BEGIN: abpxx6d04wxr
import functools
import multiprocessing as mp
import pathlib
import pickle
import random
import time
from collections import Counter, defaultdict
from typing import Any, Generator

import numpy as np
import pandas as pd
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from strainr.utils import _open

from strainr.parameter_config import process_arguments


SETCHUNKSIZE = 10000
# END: abpxx6d04wxr


def fast_count_kmers(seq_id: str, seq: bytes) -> tuple[str, np.ndarray]:
    """Main function to assign strain hits to reads
    
    Args:
    seq_id (str): Sequence ID
    seq (bytes): Sequence
    
    Returns:
    tuple[str, np.ndarray]: Tuple containing sequence ID and numpy array of matched kmer strains
    """
    matched_kmer_strains = []
    na_zeros = np.zeros(len(strains), dtype=np.uint8)
    max_index = len(seq) - kmerlen + 1
    with memoryview(seq) as seqview:
        for i in range(max_index):
            returned_strains = db.get(seqview[i : i + kmerlen])
            if returned_strains is not None:
                matched_kmer_strains.append(returned_strains)
    final_tally = sum(matched_kmer_strains)
    if isinstance(final_tally, np.ndarray):
        return seq_id, final_tally
    else:
        return seq_id, na_zeros


def fast_count_kmers_mpire(db: dict, seq_id: str, seq: bytes) -> tuple[str, np.ndarray]:
    """Main function to assign strain hits to reads
    
    Args:
    db (dict): Dictionary containing kmer strains
    seq_id (str): Sequence ID
    seq (bytes): Sequence
    
    Returns:
    tuple[str, np.ndarray]: Tuple containing sequence ID and numpy array of matched kmer strains
    """
    matched_kmer_strains = []
    na_zeros = np.full(len(strains), 0)
    max_index = len(str(seq)) - kmerlen + 1
    with memoryview(seq) as seqview:
        for i in range(max_index):
            returned_strains = db.get(seqview[i : i + kmerlen])
            if returned_strains is not None:
                matched_kmer_strains.append(returned_strains)
    final_tally = sum(matched_kmer_strains)
    if isinstance(final_tally, np.ndarray):
        return seq_id, final_tally
    else:
        return seq_id, na_zeros


def fast_count_kmers_helper(seqtuple: tuple[str, bytes]):
    """Helper function for fast_count_kmers
    
    Args:
    seqtuple (tuple[str, bytes]): Tuple containing sequence ID and sequence
    
    Returns:
    tuple[str, np.ndarray]: Tuple containing sequence ID and numpy array of matched kmer strains
    """
    return fast_count_kmers(*seqtuple)


def fastq_encoded_generator(inputfile: pathlib.Path) -> Generator:
    """Generate an updated generator expression, but without quality scores, and encodes sequences.
    
    Args:
    inputfile (pathlib.Path): Path to input file
    
    Yields:
    Generator: Generator yielding sequence ID and encoded sequence
    """
    with _open(inputfile) as f:
        for seq_id, seq, _ in FastqGeneralIterator(f):
            yield seq_id, bytes(seq, "utf-8")


def classify(input_file: pathlib.Path) -> list[tuple[str, np.ndarray]]:
    """Call multiprocessing library to lookup k-mers.
    
    Args:
    input_file (pathlib.Path): Path to input file
    
    Returns:
    list[tuple[str, np.ndarray]]: List of tuples containing sequence ID and numpy array of matched kmer strains
    """
    t0 = time.time()
    nreads = int(sum(1 for i in open(input_file, "rb")) / 4)
    print(f"Reads: {nreads}")

    # From 3-item generator to 2-item generator
    record_iter = (r for r in fastq_encoded_generator(input_file))

    # Generate k-mers, lookup strain spectra in db, return sequence scores
    with mp.Pool(processes=args.procs) as pool:
        results = list(
            pool.imap_unordered(
                fast_count_kmers_helper, record_iter, chunksize=SETCHUNKSIZE
            )
        )

    print(f"Ending classification: {time.time() - t0}s")
    return results


# FILEPATH: /home/bladen/work/strainr_all/strainr/strainr/strainr-classify.py
# BEGIN: ed8c6549bwf9
def count_kmers(seqrecord):
    """Main function to assign strain hits to reads
    
    Args:
    seqrecord (SeqRecord): Sequence record
    
    Returns:
    tuple[SeqRecord, int]: Tuple containing sequence record and number of matched kmer strains
    """
    max_index = len(seqrecord.seq) - kmerlen + 1
    matched_kmer_strains = []
    s = seqrecord.seq
    with memoryview(bytes(s)) as seqview:
        for i in range(max_index):
            returned_strains = db.get(seqview[i : i + kmerlen])
            if returned_strains is not None:
                matched_kmer_strains.append(returned_strains)
    return seqrecord, sum(matched_kmer_strains)
# END: ed8c6549bwf9


# Tests

def test_fast_count_kmers():
    seq_id = "test_seq"
    seq = b"ATCG"
    expected_output = (seq_id, np.array([0, 0, 0, 0]))
    assert fast_count_kmers(seq_id, seq) == expected_output

def test_fast_count_kmers_mpire():
    db = {"ATCG": np.array([1, 0, 0, 0])}
    seq_id = "test_seq"
    seq = b"ATCG"
    expected_output = (seq_id, np.array([1, 0, 0, 0]))
    assert fast_count_kmers_mpire(db, seq_id, seq) == expected_output

def test_fast_count_kmers_helper():
    seqtuple = ("test_seq", b"ATCG")
    expected_output = (seqtuple[0], np.array([0, 0, 0, 0]))
    assert fast_count_kmers_helper(seqtuple) == expected_output

def test_fastq_encoded_generator():
    inputfile = pathlib.Path("test.fastq")
    expected_output = [("test_seq", b"ATCG")]
    assert list(fastq_encoded_generator(inputfile)) == expected_output

def test_classify():
    input_file = pathlib.Path("test.fastq")
    expected_output = [("test_seq", np.array([0, 0, 0, 0]))]
    assert classify(input_file) == expected_output

def test_count_kmers():
    seqrecord = SeqRecord(Seq("ATCG"), id="test_seq")
    expected_output = (seqrecord, 0)
    assert count_kmers(seqrecord) == expected_output