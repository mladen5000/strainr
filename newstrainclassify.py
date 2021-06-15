# new
import time
import pandas as pd
import pickle
import pathlib
from Bio import SeqIO

# kmerset = {bytes(seqview[i : i + KMERLEN]) for i in range(max_index)}
def count_kmers(seqrecord):
    max_index = len(seqrecord.seq) - KMERLEN + 1
    matched_kmer_strains = []
    with memoryview(bytes(seqrecord.seq)) as seqview:
        for i in range(max_index):
            returned_strains = db.get(seqview[i : i + KMERLEN])
            if returned_strains is not None:
                matched_kmer_strains.append(returned_strains)
    return sum(matched_kmer_strains)


def get_hits(read):
    subset = df[df.index.isin(kset)].sum(axis=1)
    max_val = subset.max()
    result = subset[subset == max_val].index.to_list()
    print(result)
    return result


def final(results):
    return results


def load_database(dbfile):
    df = pd.read_pickle(dbfile)
    return df


def df_to_dict(df):
    strain_array = list(df.to_numpy())
    strain_ids = df.columns
    kmers = df.index.to_list()
    db = dict(zip(kmers, larrays))
    return strain_ids, db


def get_args():
    """
    # Examples
    parser.add_argument('--source_file', type=open)
    parser.add_argument('--dest_file', type=argparse.FileType('w', encoding='latin-1'))
    parser.add_argument('--datapath', type=pathlib.Path)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="input file",
        type=str,
    )
    parser.add_argument(
        "-j",
        "--reverse-pair",
        help="reverse reads",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--db",
        help="Database file",
        type=int,
        default=31,
    )
    parser.add_argument(
        "-p",
        "--procs",
        type=int,
        default=1,
        help="Number of cores to use (default: 1)",
    )
    parser.add_argument(
        "-o", "--out", type=pathlib.Path, help="Output folder", default="strainr_out"
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


if __name__ == "__main__":
    p = pathlib.Path().cwd()
    params = vars(get_args().parse_args())
    main()

def main():
    df = load_database("new_method.sdb")
    strains, db = df_to_dict(df)

    results = []
    for i, read in enumerate(SeqIO.parse("inputs/test_R1.fastq", "fastq")):
        res = count_kmers(read)
        results.append(res)
    print(len(results))

    # for i,read in enumerate(SeqIO.parse("inputs/short_R1.fastq", "fastq")):
    #     print(i)
    #     kset = delayed(count_kmers)(read)
    #     res = delayed(get_hits)(kset)
    #     results.append(res)
    # results = delayed(final)(results)
    # resultsf = results.compute()
    # print(resultsf)
