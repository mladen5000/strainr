
def prefilter_database():
    return {k:v for k,v in kmer_database if k in kmer_set}

def count_kmers_input(allseqs):
    kmer_set = set()
    for rid,seq in allseqs:
        max_index = len(seq) - KMERLEN + 1
        with memoryview(bytes(seq)) as seq_buffer:
            kset = {bytes(seq_buffer[i:i+KMERLEN]) for i in range(max_index)}
        kmer_set |= kset
    return kmer_set

def fetch_kmers2(x):
    # Fetch records
    with memoryview(bytes(seq_records[x].seq)) as seq_buffer:
        max_index = len(seq_buffer) - KMERLEN + 1
        matched_kmer_strains = []
        for i in range(max_index):
            if i < max_index:
                returned_strains = kmer_database.get(
                    seq_buffer[i : i + KMERLEN], tuple()
                )
                matched_kmer_strains.append(returned_strains)

    # Flatten list of tuples
    flat_strain_list = []
    for sublist in matched_kmer_strains:
        for i in sublist:
            if sublist:
                flat_strain_list.append(i)

    # Grab most frequent strains
    top_hits = multimode(flat_strain_list)

    if top_hits:
        return seq_records[x].id, tuple(top_hits)
    else:
        return seq_records[x].id, tuple(tuple("X"))

def wvs_paired(
    parser1: Generator, parser2: Generator = None, fasta: bool = False
) -> Iterator[Tuple[str, bytes]]:
    """
    Takes 2 generators (SimpleFasta or FastqGeneral) and Combines them
    Does not work with SeqIO parse i think, which is fine cause it's slow
    """
    if parser2:
        if fasta:
            for (read_id, seq1), (_, seq2) in zip(parser1, parser2):
                seq = "NNN".join([seq1, seq2])
                yield (
                    read_id,
                    seq,
                )
        else:
            for (read_id, seq1, _), (_, seq2, _) in zip(parser1, parser2):
                seq = "NNN".join([seq1, seq2])
                yield (
                    read_id,
                    seq,
                )
    else:
        for (read_id, seq, _) in parser1:
            yield (
                read_id,
                seq,
            )

def preprocess_reads2(infile, input_ext, reverse_file):
    """Allows for fastq v fasta, paired v unpaired, and zipped vs unzipped"""

    # Either FastqGeneralIterator or SimpleFastaParser
    SeqParser = select_generator(infile, input_ext, reverse_file)
    print(SeqParser)

    # Identify compression and create fileobjects
    f1 = omni_open(infile)
    # Create Generators from file objects
    g1 = SeqParser(f1)

    if not reverse_file:
        print(f"Detected unpaired fasta/q file")
        records = ((i, j) for (i, j) in SeqParser(f1))
    else:
        print(f"Detected paired fasta/q files")
        f2 = (
            _open(
                infile,
            )
            if reverse_file
            else None
        )
        g2 = SeqParser(f2)
        records = wvs_paired(g1, g2)
    return records

def select_generator(infile, input_ext, reverse_file):
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
        file_parser = FastqGeneralIterator
    elif fasta_condition:
        file_parser = SimpleFastaParser
    else:
        raise ValueError("File does not end in standard fasta or fastq format")

    return file_parser
    # if f2: # Paired fastq
    #     if fastq_condition:
    #         print("\tDetected Paired-Fastq file")
    #         g1 = FastqGeneralIterator(f1)
    #         g2 = FastqGeneralIterator(f2)
    #         records = wvs_paired(g1, g2)  # infile, reverse_file))
    #     else: # Paired fasta
    #         g1 = SimpleFastaParser(f1)
    #         g2 = SimpleFastaParser(f2)
    #         records = wvs_paired(g1, g2, fasta=True)  # infile, reverse_file))
    # else: # Unpaired
    #     if fasta_condition:
    #     print("\tDetected Paired-Fasta file")
    #     f1 = _open( infile,)
    #     f2 = _open( reverse_file,)
    #     g1 = SimpleFastaParser(f1)
    #     g2 = SimpleFastaParser(f2)
    #     records = wvs_paired(g1, g2, fasta=True)  # infile, reverse_file))

    # elif fasta_condition and not reverse_file:
    #     f1 = _open(
    #         infile,
    #     )
    #     print("\tDetected unpaired Fasta file")
    #     records = ((i, j) for (i, j) in SimpleFastaParser(f1))

    # if fastq_condition and reverse_file:
    #     print("\tDetected Paired-Fastq file")
    #     f1 = _open(
    #         infile,
    #     )
    #     f2 = _open(
    #         reverse_file,
    #     )
    #     g1 = FastqGeneralIterator(f1)
    #     g2 = FastqGeneralIterator(f2)
    #     records = wvs_paired(g1, g2)  # infile, reverse_file))

    # elif fastq_condition and not reverse_file:
    #     print("\tDetected unpaired Fastq file")
    #     f1 = _open(
    #         infile,
    #     )
    #     records = ((i, j) for (i, j, k) in FastqGeneralIterator(f1))

    # elif fasta_condition and reverse_file:
    #     print("\tDetected Paired-Fasta file")
    #     f1 = _open(
    #         infile,
    #     )
    #     f2 = _open(
    #         reverse_file,
    #     )
    #     g1 = SimpleFastaParser(f1)
    #     g2 = SimpleFastaParser(f2)
    #     records = wvs_paired(g1, g2, fasta=True)  # infile, reverse_file))

    # elif fasta_condition and not reverse_file:
    #     f1 = _open(
    #         infile,
    #     )
    #     print("\tDetected unpaired Fasta file")
    #     records = ((i, j) for (i, j) in SimpleFastaParser(f1))

    # else:
    #     records = None
    #     print(infile, input_ext, reverse_file)
    #     print("womp")
    #     sys.exit()

    # return records

def fetch_kmers(rid, seq):
    max_index = len(seq) - KMERLEN + 1
    # Fetch records
    matched_kmer_strains = []
    with memoryview(bytes(seq)) as seq_buffer:
        for i in range(max_index):
            if i < max_index:
                returned_strains = kmer_database.get(
                    seq_buffer[i : i + KMERLEN], tuple()
                )
                matched_kmer_strains.append(returned_strains)

def full_sort(db):
    """Sorts each list as well as the dict itself"""
    logger.info("Optimizing The Dictionary")
    return {key: sorted(db[key]) for key in sorted(db)}


def convert_to_presence_absence(db):
    """
    Convert database so each entry only has a strain appearing up to 1 time
    """
    return {k: set(v) for k, v in db.items()}


def filter_most(db, num_strains):
    """Remove kmers that have hits for most strains"""
    return {k: v for k, v in db.items() if len(v) > num_strains // 2}


def filter_by_length(db, max_len):
    """
    Remove kmers that have more than n hits
    eg) max_len = 1 for unique kmers only
    """
    return {k: v for k, v in db.items() if len(v) > max_len}


def extract_acc(filepath):
    fname = filepath.stem
    return fname[:15]


def analyze_genome(genome_file):
    kmers = count_kmers(genome_file)
    acc = extract_acc(genome_file)
    metadata_dict = dict(filepath=genome_file, kmer_set=kmers, accession=acc)
    return metadata_dict


def count_kmers(genome_file):
    encoding = guess_type(genome_file)[1]  # uses file extension
    _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open
    with _open(genome_file) as g:
        record = SeqIO.read(g, "fasta")
    kmerlen = params["kmerlen"]
    max_index = len(record.seq) - kmerlen + 1
    with memoryview(bytes(record.seq)) as seq_buffer:
        kmerset = {bytes(seq_buffer[i : i + kmerlen]) for i in range(max_index)}
    logger.info(len(kmerset))
    return kmerset

def pickle_genome(metadata, kmerdir):
    ppath = kmerdir / metadata["accession"] + ".pkl"
    with ppath.open("wb") as ph:
        pickle.dump(metadata, ph)
    return
