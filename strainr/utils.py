#!/usr/bin/env python

import functools
import gzip
import mimetypes
import pickle

import pandas as pd


def _open(infile):
    """Handle unknown file for gzip and non-gzip alike"""
    encoding = mimetypes.guess_type(str(infile))[1]  # uses file extension
    _open = functools.partial(gzip.open, mode="rt") if encoding == "gzip" else open
    file_object = _open(infile)
    return file_object


def get_rc(kmer):
    rc_kmer = kmer.reverse_complement()
    return kmer if kmer < rc_kmer else rc_kmer


def pickle_results(outfile, results_raw, total_hits, strains):
    with open((outfile / "raw_results.pkl"), "wb") as fh:
        pickle.dump(results_raw, fh)
    with open((outfile / "total_hits.pkl"), "wb") as fh:
        pickle.dump(total_hits, fh)

    # save_results(results_raw, total_hits, strains) #TODO - missing arg
    return


def save_results(outfile, intermediate_scores, results, strains):
    """
    Take a dict of readid:hits and convert to a dataframe
    then save the output
    """
    df = pd.DataFrame.from_dict(
        dict(intermediate_scores), orient="index", columns=strains
    ).astype(int)
    final_names = {k: strains[int(v)] for k, v in results.items() if v != "NA"}
    assigned = pd.Series(final_names).rename("final")
    df = df.join(assigned)
    # savepath = outdir / "results_table.csv"
    # df.to_csv(savepath)
    picklepath = outfile / "results_table.pkl"
    df.to_pickle(picklepath)
    return


call_pickle = None
# if call_pickle:
#     pickle_results(results_raw, total_hits, strains)
