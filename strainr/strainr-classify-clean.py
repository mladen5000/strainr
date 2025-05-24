#!/usr/bin/env python
import argparse
import gzip
import logging
import multiprocessing as mp
import pathlib
import sys
from collections import Counter
from math import log2
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from pydantic import BaseModel, Field, field_validator, model_validator

SETCHUNKSIZE = 10000

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("strainr.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class Args(BaseModel):
    input_forward: Union[pathlib.Path, List[pathlib.Path]] = Field(
        alias="input_forward"
    )
    input_reverse: Optional[Union[pathlib.Path, List[pathlib.Path]]] = Field(
        default=None, alias="input_reverse"
    )
    db: pathlib.Path
    procs: int = Field(default=4, ge=1, alias="procs")
    out: pathlib.Path = Field(default="strainr_out", alias="out")
    mode: str = Field(
        default="max",
        pattern="^(random|max|multinomial|dirichlet)$",
        alias="mode",
    )
    thresh: float = Field(default=0.001, ge=0.0, alias="thresh")
    bin: bool = Field(default=False, alias="bin")
    save_raw_hits: bool = Field(default=False, alias="save_raw_hits")

    @field_validator("input_forward", "input_reverse", mode="before")
    @classmethod
    def validate_file_paths(
        cls, v: Union[str, List[str]]
    ) -> Union[pathlib.Path, List[pathlib.Path]]:
        logger.debug(f"Validating file paths: {v}")
        if isinstance(v, list):
            return [cls.validate_single_path(path) for path in v]
        return cls.validate_single_path(v)

    @classmethod
    def validate_single_path(cls, v: str) -> pathlib.Path:
        logger.debug(f"Validating single path: {v}")
        path = pathlib.Path(v)
        if not path.exists():
            raise ValueError(f"File does not exist: {v}")
        return path

    @model_validator(mode="after")
    def check_paired_reads(self):
        logger.debug("Checking paired reads")
        if self.input_reverse:
            if isinstance(self.input_forward, list) and isinstance(
                self.input_reverse, list
            ):
                if len(self.input_reverse) != len(self.input_forward):
                    raise ValueError(
                        "The number of reverse reads must match the number of forward reads."
                    )
            elif isinstance(self.input_forward, pathlib.Path) and isinstance(
                self.input_reverse, pathlib.Path
            ):
                pass
            else:
                raise ValueError(
                    "Both input_forward and input_reverse should be either single paths or lists of paths."
                )
        return self


class KmerProcessor:
    def __init__(self, args: Args):
        logger.info("Initializing KmerProcessor")
        self.args = args
        self.kmerlen: int
        self.strains: List[str]
        self.db: Dict[bytes, np.ndarray]

    def open_file(self, inputfile: pathlib.Path):
        logger.debug(f"Opening file: {inputfile}")
        if inputfile.suffix.endswith(".gz"):
            return gzip.open(inputfile, "rt")
        else:
            return inputfile.open("rt")

    def detect_file_format(self, inputfile: pathlib.Path) -> str:
        logger.debug(f"Detecting file format for: {inputfile}")
        with self.open_file(inputfile) as f:
            first_line = f.readline().strip()
            if first_line.startswith(">"):
                return "fasta"
            elif first_line.startswith("@"):
                return "fastq"
            else:
                raise ValueError(f"Unknown file format for file: {inputfile}")

    def parse_file(
        self, input_forward: pathlib.Path, input_reverse: pathlib.Path = None
    ) -> Generator[Tuple[str, bytes, bytes], None, None]:
        logger.info(f"Parsing files: {input_forward}, {input_reverse}")
        format_forward = self.detect_file_format(input_forward)
        format_reverse = (
            self.detect_file_format(input_reverse) if input_reverse else None
        )

        with self.open_file(input_forward) as fwd:
            rev = self.open_file(input_reverse) if input_reverse else None
            try:
                if format_forward == "fasta":
                    forward_iter = SeqIO.parse(fwd, "fasta")
                    reverse_iter = SeqIO.parse(rev, "fasta") if rev else None
                    for forward_record in forward_iter:
                        if reverse_iter:
                            reverse_record = next(reverse_iter)
                            yield (
                                forward_record.id,
                                bytes(str(forward_record.seq), "utf-8"),
                                bytes(str(reverse_record.seq), "utf-8"),
                            )
                        else:
                            yield (
                                forward_record.id,
                                bytes(str(forward_record.seq), "utf-8"),
                                b"",
                            )
                elif format_forward == "fastq":
                    forward_iter = FastqGeneralIterator(fwd)
                    reverse_iter = FastqGeneralIterator(rev) if rev else None
                    for fwd_id, fwd_seq, _ in forward_iter:
                        if reverse_iter:
                            rev_id, rev_seq, _ = next(reverse_iter)
                            yield (
                                fwd_id,
                                bytes(fwd_seq, "utf-8"),
                                bytes(rev_seq, "utf-8"),
                            )
                        else:
                            yield fwd_id, bytes(fwd_seq, "utf-8"), b""
            finally:
                if rev:
                    rev.close()

    def build_database(self, dbpath: pathlib.Path):
        logger.info(f"Building database from: {dbpath}")
        df = pd.read_pickle(dbpath)
        self.kmerlen = len(df.index[0])
        self.strains = list(df.columns)
        self.db = dict(zip(df.index, df.to_numpy()))
        logger.info(
            f"Database loaded with {len(self.strains)} strains and k-mer length {self.kmerlen}."
        )

    def fast_count_kmers(
        self, record: Tuple[str, bytes, bytes]
    ) -> Tuple[str, np.ndarray]:
        logger.debug(f"Counting k-mers for record: {record[0]}")
        seq_id, fwd_seq, rev_seq = record
        matched_kmer_strains: List[np.ndarray] = []
        na_zeros = np.full(len(self.strains), 0)

        # Process forward read
        for seq in [fwd_seq, rev_seq]:
            if len(seq) < self.kmerlen:
                continue
            with memoryview(seq) as seqview:
                for i in range(len(seq) - self.kmerlen + 1):
                    kmer = seqview[i : i + self.kmerlen].tobytes()
                    returned_strains = self.db.get(kmer)
                    if returned_strains is not None:
                        matched_kmer_strains.append(returned_strains)

        if matched_kmer_strains:
            final_tally = sum(matched_kmer_strains)
            return seq_id, final_tally
        else:
            return seq_id, na_zeros

    def classify(
        self, input_forward: pathlib.Path, input_reverse: pathlib.Path = None
    ) -> List[Tuple[str, np.ndarray]]:
        logger.info(f"Classifying reads from: {input_forward}, {input_reverse}")
        record_iter = self.parse_file(input_forward, input_reverse)
        with mp.Pool(processes=self.args.procs) as pool:
            results = pool.map(
                self.fast_count_kmers, record_iter, chunksize=SETCHUNKSIZE
            )
        return results

    def save_read_spectra(self, results_raw: List[Tuple[str, np.ndarray]]):
        logger.info("Saving raw read spectra")
        df = pd.DataFrame(results_raw, columns=["read_id", "strain_counts"])
        outdir = self.args.out / "raw_hits"
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "raw_scores.csv", index=False)
        logger.info(f"Raw read spectra saved to {outdir / 'raw_scores.csv'}")

    def separate_hits(
        self, hitcounts: List[Tuple[str, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
        logger.info("Separating hits")
        clear_hits, ambig_hits, none_hits = {}, {}, []
        for read, hit_array in hitcounts:
            if np.all(hit_array == 0):
                none_hits.append(read)
            else:
                max_indices = np.argwhere(hit_array == np.max(hit_array)).flatten()
                if len(max_indices) == 1:
                    clear_hits[read] = hit_array
                elif len(max_indices) > 1:
                    ambig_hits[read] = hit_array
        return clear_hits, ambig_hits, none_hits

    def resolve_clear_hits(self, clear_hits: Dict[str, np.ndarray]) -> Dict[str, int]:
        logger.info("Resolving clear hits")
        return {k: int(np.argmax(v)) for k, v in clear_hits.items()}

    def resolve_ambiguous_hits(
        self, ambig_hits: Dict[str, np.ndarray]
    ) -> Dict[str, int]:
        logger.info("Resolving ambiguous hits")
        resolved_hits = {}
        for read, hit_array in ambig_hits.items():
            max_val = np.max(hit_array)
            hit_array_thresholded = np.where(hit_array == max_val, hit_array, 0)
            mlehits = hit_array_thresholded * self.prior

            if self.args.mode == "random":
                resolved_strain = np.random.choice(
                    np.flatnonzero(mlehits),
                    p=mlehits[np.flatnonzero(mlehits)]
                    / mlehits[np.flatnonzero(mlehits)].sum(),
                )
            elif self.args.mode == "max":
                resolved_strain = np.argmax(mlehits)
            elif self.args.mode == "dirichlet":
                mlehits[mlehits == 0] = 1e-10
                resolved_strain = np.argmax(np.random.dirichlet(mlehits))
            elif self.args.mode == "multinomial":
                mlehits[mlehits == 0] = 1e-10
                resolved_strain = np.argmax(
                    np.random.multinomial(1, mlehits / mlehits.sum())
                )
            else:
                raise ValueError("Invalid selection mode.")

            resolved_hits[read] = resolved_strain
        return resolved_hits

    def normalize_counter(
        self, acounter: Counter, remove_na: bool = False
    ) -> Counter[Any]:
        logger.info("Normalizing counter")
        if remove_na and "NA" in acounter:
            del acounter["NA"]
        total_counts = sum(acounter.values())
        return Counter({k: v / total_counts for k, v in acounter.items()})

    def threshold_by_relab(
        self, norm_counter_all: Counter, threshold: float = 0.02
    ) -> Counter:
        logger.info("Applying relative abundance threshold")
        thresh_counter = Counter(
            {k: v if v > threshold else 0.0 for k, v in norm_counter_all.items()}
        )
        return self.normalize_counter(thresh_counter, remove_na=True)

    def display_relab(
        self,
        acounter: Counter,
        nstrains: int = 10,
        template_string: str = "",
        display_na: bool = True,
    ):
        logger.info("Displaying relative abundances")
        print(f"\n\n{template_string}\n")
        for strain, abund in acounter.most_common(n=nstrains):
            if abund > 0.0 and strain != "NA":
                print(f"{abund:.4f}\t{strain}")
        if display_na and acounter.get("NA", 0.0) > 0.0:
            print(f"{acounter['NA']:.4f}\tNA\n")

    def translate_strain_indices_to_names(
        self, counter_indices: Counter, strain_names: List[str]
    ) -> Counter[str]:
        logger.info("Translating strain indices to names")
        name_to_hits = {}
        for k_idx, v_hits in counter_indices.items():
            if k_idx != "NA" and isinstance(k_idx, int):
                name_to_hits[strain_names[k_idx]] = v_hits
            elif k_idx == "NA":
                name_to_hits["NA"] = v_hits
            else:
                try:
                    name_to_hits[strain_names[k_idx]] = v_hits
                except (IndexError, TypeError):
                    logger.error(f"Invalid strain index: {k_idx}")
        return Counter(name_to_hits)

    def add_missing_strains(
        self, strain_names: List[str], final_hits: Counter[str]
    ) -> Counter[str]:
        logger.info("Adding missing strains")
        full_strain_relab = Counter(
            {strain: final_hits.get(strain, 0.0) for strain in strain_names}
        )
        full_strain_relab["NA"] = final_hits.get("NA", 0.0)
        return full_strain_relab

    def compute_summary_statistics(self, results: Dict[str, int]) -> Dict[str, Any]:
        logger.info("Computing summary statistics")
        total_reads = sum(results.values())
        total_strains = len(results)
        return {
            "total_reads": total_reads,
            "total_strains": total_strains,
        }

    def compute_shannon_diversity(self, results: Dict[str, int]) -> float:
        logger.info("Computing Shannon diversity index")
        total_reads = sum(results.values())
        if total_reads == 0:
            return 0.0
        shannon_index = -sum(
            (count / total_reads) * log2(count / total_reads)
            for count in results.values()
            if count > 0
        )
        return shannon_index

    def top_n_strains(
        self, results: Dict[str, int], n: int = 10
    ) -> List[Tuple[str, int]]:
        logger.info("Getting top N strains")
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[:n]

    def strain_coverage(self, results: Dict[str, int], n: int = 10) -> float:
        logger.info("Calculating strain coverage")
        total_reads = sum(results.values())
        top_strains = self.top_n_strains(results, n)
        top_reads = sum([count for _, count in top_strains])
        return (top_reads / total_reads) * 100 if total_reads > 0 else 0.0

    def output_results(self, results: Dict[str, int]):
        logger.info("Outputting results")
        outdir = self.args.out
        outdir.mkdir(parents=True, exist_ok=True)

        # Translate indices to strain names
        name_hits = self.translate_strain_indices_to_names(
            Counter(results.values()), self.strains
        )
        full_name_hits = self.add_missing_strains(self.strains, name_hits)

        # Display overall hits
        self.display_relab(full_name_hits, template_string="Overall Hits")

        # Normalize to get relative abundances
        final_relab = self.normalize_counter(full_name_hits, remove_na=False)
        self.display_relab(final_relab, template_string="Initial Relative Abundance")

        # Apply threshold
        final_threshab = self.threshold_by_relab(
            final_relab, threshold=self.args.thresh
        )
        self.display_relab(
            final_threshab, template_string="Post-Thresholding Relative Abundance"
        )

        # Create DataFrame
        relab_columns = [
            pd.DataFrame(
                list(full_name_hits.items()), columns=["strain", "sample_hits"]
            ).set_index("strain"),
            pd.DataFrame(
                list(final_relab.items()), columns=["strain", "sample_relab"]
            ).set_index("strain"),
            pd.DataFrame(
                list(final_threshab.items()), columns=["strain", "intra_relab"]
            ).set_index("strain"),
        ]
        results_table = pd.concat(relab_columns, axis=1).sort_values(
            by="sample_hits", ascending=False
        )
        results_table.to_csv(outdir / "abundance.tsv", sep="\t")
        logger.info(f"Abundance results saved to {outdir / 'abundance.tsv'}")

    def main(self):
        logger.info("Starting main processing")
        self.build_database(self.args.db)
        all_results = {}

        for idx, input_forward in enumerate(self.args.input_forward):
            input_reverse = (
                self.args.input_reverse[idx] if self.args.input_reverse else None
            )
            logger.info(
                f"Processing {'paired' if input_reverse else 'single'} read: {input_forward}"
            )
            results_raw = self.classify(input_forward, input_reverse)
            if self.args.save_raw_hits:
                self.save_read_spectra(results_raw)
            clear_hits, ambig_hits, na_hits = self.separate_hits(results_raw)
            assigned_clear = self.resolve_clear_hits(clear_hits)

            # Aggregate results
            for read_id, strain_idx in assigned_clear.items():
                all_results[read_id] = strain_idx

        # Perform additional analyses
        summary_stats = self.compute_summary_statistics(all_results)
        shannon_index = self.compute_shannon_diversity(all_results)
        top_strains = self.top_n_strains(all_results, n=5)
        coverage = self.strain_coverage(all_results, n=5)

        logger.info(f"Summary Statistics: {summary_stats}")
        logger.info(f"Shannon Diversity Index: {shannon_index:.4f}")
        logger.info(f"Top 5 Strains: {top_strains}")
        logger.info(f"Coverage of Top 5 Strains: {coverage:.2f}%")

        # Output the results
        self.output_results(all_results)


def parse_arguments() -> Args:
    logger.info("Parsing command-line arguments")
    parser = argparse.ArgumentParser(
        description="K-mer based strain classification tool."
    )
    parser.add_argument(
        "--input_forward",
        help="Input forward read file(s) (FASTA/FASTQ, possibly gzipped).",
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input_reverse",
        help="Input reverse read file(s) for paired-end data (FASTA/FASTQ, possibly gzipped).",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--db",
        help="Path to the k-mer database file (pickle format).",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--procs",
        help="Number of processor cores to use (default: 4).",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--out",
        help="Output directory (default: strainr_out).",
        type=str,
        default="strainr_out",
    )
    parser.add_argument(
        "--mode",
        help="Selection mode for disambiguation.",
        choices=["random", "max", "multinomial", "dirichlet"],
        default="max",
    )
    parser.add_argument(
        "--thresh",
        help="Threshold for relative abundance filtering (default: 0.001).",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--bin",
        help="Perform binning.",
        action="store_true",
    )
    parser.add_argument(
        "--save_raw_hits",
        help="Save intermediate results as a CSV.",
        action="store_true",
    )

    raw_args = parser.parse_args()
    return Args(
        input_forward=raw_args.input_forward,
        input_reverse=raw_args.input_reverse or [],
        db=pathlib.Path(raw_args.db),
        procs=raw_args.procs,
        out=pathlib.Path(raw_args.out),
        mode=raw_args.mode,
        thresh=raw_args.thresh,
        bin=raw_args.bin,
        save_raw_hits=raw_args.save_raw_hits,
    )


if __name__ == "__main__":
    logger.info("Starting Strainr classification")
    args = parse_arguments()
    processor = KmerProcessor(args)
    try:
        processor.main()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    logger.info("Strainr classification completed")
