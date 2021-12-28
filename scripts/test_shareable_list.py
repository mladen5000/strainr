from multiprocessing.managers import SharedMemoryManager
from multiprocessing import shared_memory as sm
from multiprocessing import Pool
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from statistics import multimode

def count_kmers(seq):
    # Shareable list
    KMERLEN=31
    max_index = len(seq) - KMERLEN + 1
    kmerset = multimode( seq[i:i+KMERLEN] for i in range(max_index) if i < max_index)
    return kmerset

def count_kmers_rl(seq):
    # Regular list
    KMERLEN=31
    max_index = len(seq) - KMERLEN + 1
    with memoryview(seq) as seq_buffer:
        kmerset = multimode( seq_buffer[i:i+KMERLEN].tobytes() for i in range(len(seq_buffer) - KMERLEN + 1) )
    return kmerset

# Generators
fastq_generator = (j.strip().encode() for (i,j,k) in FastqGeneralIterator('test_R1.fastq'))
fastq_readid_generator = (i for (i,j,k) in FastqGeneralIterator('test_R1.fastq'))

# Call shared
with Pool(processes=48) as p:
    with SharedMemoryManager() as smm:
        results = p.imap_unordered(count_kmers,smm.ShareableList(list(fastq_generator)))
#     # sl = smm.ShareableList(list(fastq_generator))
#     # sl = sm.ShareableList(list(fastq_generator))

# Call regular
# with Pool(processes=48) as p:
#     results = p.map(count_kmers_rl,fastq_generator)
    
results2 = list(zip(fastq_readid_generator,results))
# print((results2))
# [kmer.tobytes() for kmer in count_kmers(sl[0])]