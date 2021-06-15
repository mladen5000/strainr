# new
import dask
import dask.dataframe as dpd
from Bio import SeqIO
import time
import pandas as pd
from dask import delayed
from dask.distributed import Client
import pickle
# kmerset = {bytes(seqview[i : i + KMERLEN]) for i in range(max_index)}
def count_kmers(seqrecord):
    max_index = len(seqrecord.seq) - KMERLEN + 1
    matched_kmer_strains = []
    with memoryview(bytes(seqrecord.seq)) as seqview:
        for i in range(max_index):
            returned_strains = db.get(seqview[i:i+KMERLEN])
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

def load_pdatabase(dbfile):
    df = pd.read_pickle(dbfile)
    return df

def df_to_dict(df):
    strain_array = list(df.to_numpy())
    strain_ids = df.columns
    kmers = df.index.to_list()
    db = dict(zip(kmers,larrays))
    return strain_ids,db

if __name__ == "__main__":
    p = pathlib.Path().cwd()
    params = get_args()
    main()

    KMERLEN = 31
    t0 = time.time()
    df = load_pdatabase("new_method.sdb")
    strains,db = df_to_dict(df)

    results = []
    for i,read in enumerate(SeqIO.parse("inputs/test_R1.fastq", "fastq")):
        res = count_kmers(read)
        results.append(res)
    print(len(results))

    #     res = delayed(get_hits)(kset)
    # print((db))
    # client = Client()
    # t1 = time.time() - t0
    # print(f"df loaded in {t1} seconds")
    # # df = dpd.from_pandas(df, npartitions=16).persist()
    # # df = df.compute()
    # print(f"dask df in {time.time() - t1} seconds")


    # results = []
    # for i,read in enumerate(SeqIO.parse("inputs/short_R1.fastq", "fastq")):
    #     print(i)
    #     kset = delayed(count_kmers)(read)
    #     res = delayed(get_hits)(kset)
    #     results.append(res)
    # results = delayed(final)(results)
    # resultsf = results.compute()
    # print(resultsf)

# t0 = time.time()
# for i,read in enumerate(read_generator):
#     print(i)
#     kset = count_kmers(read)
#     subset = df[df.index.isin(kset)].sum(axis=1)
#     max_val= subset.max()
#     result = subset[subset == max_val].index
#     print(result)
# print(f"Total time is {time.time() - t0}")
# print(df[subset == subset_val])
# print(subset[== subset_val].index.to_list())
