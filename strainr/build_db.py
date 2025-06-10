#!/usr/bin/env python
"""
strainr Database Builder: Downloads genomes and creates a k-mer presence/absence database.

This script automates the process of:
1. Downloading bacterial genomes from NCBI (RefSeq or GenBank) based on
   taxonomic ID, a list of assembly accessions, or genus name.
2. Filtering downloaded genomes, for example, to include only those with
   unique strain-level taxonomic IDs.
3. Extracting k-mers from the genomic FASTA files (optimized with NumPy).
4. Constructing a presence/absence matrix where rows are unique k-mers,
   columns are strains (genomes), and values indicate if a k-mer is present.
5. Saving this matrix as a Parquet file.

K-mer Strategy:
- Canonical K-mers: The database stores canonical k-mers (the lexicographically
  smaller of a k-mer and its reverse complement) to ensure strand-insensitivity
  during classification.
- Ambiguous Bases ('N'): By default, k-mers containing 'N' bases are included.
  The `--skip-n-kmers` flag can be used to exclude such k-mers.
- K-mer Length: The default k-mer length is 31 (configurable via `--kmerlen`),
  a common choice for bacterial genomes balancing specificity and sensitivity.

K-mer Strategy:
- Canonical K-mers: The database stores canonical k-mers (the lexicographically
  smaller of a k-mer and its reverse complement) to ensure strand-insensitivity
  during classification.
- Ambiguous Bases ('N'): By default, k-mers containing 'N' bases are included.
  The `--skip-n-kmers` flag can be used to exclude such k-mers.
- K-mer Length: The default k-mer length is 31 (configurable via `--kmerlen`),
  a common choice for bacterial genomes balancing specificity and sensitivity.
"""

import argparse
import logging
import multiprocessing as mp  # Added

import pathlib
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ncbi_genome_download as ngd
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from .utils import open_file_transparently

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("strainr_db_creation.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

try:
    from kmer_counter_rs import extract_kmers_rs

    _extract_kmers_func = extract_kmers_rs
    _RUST_KMER_COUNTER_AVAILABLE = True
    logger.info(
        "Successfully imported Rust k-mer counter. Using Rust implementation for k-mer extraction."
    )
except Exception as e:  # pragma: no cover - rust module optional
    _extract_kmers_func = None
    _RUST_KMER_COUNTER_AVAILABLE = False
    logger.warning(
        f"Rust k-mer counter not available ({e}). Falling back to Python implementation."
    )


class DatabaseBuilder:
    """
    Manages the k-mer database creation workflow.

    This class encapsulates all steps from genome download to k-mer matrix
    generation and saving.
    """

    PY_RC_TRANSLATE_TABLE = bytes.maketrans(b"ACGTN", b"TGCAN")

    def __init__(self, args: argparse.Namespace) -> None:
        logger.info("Initializing DatabaseBuilder.")
        self.args: argparse.Namespace = args
        self.base_path: pathlib.Path = pathlib.Path().cwd()
        self.output_db_name: str = self.args.out
        self.output_db_dir: pathlib.Path = self.base_path / (
            self.output_db_name + "_files"
        )
        self.output_db_dir.mkdir(parents=True, exist_ok=True)

    def _parse_assembly_level(self) -> str:
        """
        Determines the assembly level string for ncbi-genome-download.
        """
        logger.debug(f"Parsing assembly level: {self.args.assembly_levels}")
        level_choice = self.args.assembly_levels
        if self.args.assembly_accessions:
            return "all"

        if level_choice == "complete":
            return "complete"
        elif level_choice == "chromosome":
            return "complete,chromosome"
        elif level_choice == "scaffold":
            return "complete,chromosome,scaffold"
        elif level_choice == "contig":
            return "complete,chromosome,scaffold,contig"
        else:
            raise ValueError(f"Invalid assembly level selected: {level_choice}")

    def _download_genomes_from_ncbi(self) -> Tuple[pathlib.Path, pathlib.Path]:
        """
        Downloads genomes from NCBI using `ncbi-genome-download`.
        """
        logger.info("Preparing to download genomes from NCBI.")
        assembly_level_str = self._parse_assembly_level()

        ncbi_kwargs: Dict[str, Any] = {
            "flat_output": True,
            "groups": "bacteria",
            "file_formats": "fasta",
            "section": self.args.source,
            "parallel": self.args.procs,
            "assembly_levels": assembly_level_str,
        }

        genome_target_dir_suffix: str = ""
        if self.args.taxid:
            if self.args.assembly_accessions:
                raise ValueError(
                    "Cannot specify both --taxid and --assembly_accessions."
                )
            ncbi_kwargs["species_taxids"] = self.args.taxid
            genome_target_dir_suffix = f"s{self.args.taxid}"
        elif self.args.assembly_accessions:
            accessions_file = pathlib.Path(self.args.assembly_accessions)
            if not accessions_file.is_file():
                raise FileNotFoundError(
                    f"Assembly accessions file not found: {accessions_file}"
                )
            ncbi_kwargs["assembly_accessions"] = str(accessions_file)
            genome_target_dir_suffix = f"acc_{accessions_file.stem}"
        elif self.args.genus:
            ncbi_kwargs["genera"] = self.args.genus
            genome_target_dir_suffix = f"g{self.args.genus.replace(' ', '_')}"
        else:
            raise ValueError(
                "Must specify one of --taxid, --assembly_accessions, or --genus for NCBI download."
            )

        genome_output_dir = (
            self.output_db_dir / f"ncbi_genomes_{genome_target_dir_suffix}"
        )
        metadata_table_path = (
            genome_output_dir / f"metadata_summary_{genome_target_dir_suffix}.tsv"
        )

        ncbi_kwargs.update(
            {
                "output": genome_output_dir,
                "metadata_table": metadata_table_path,
            }
        )

        logger.info(
            f"Downloading genomes to: {genome_output_dir} with metadata to {metadata_table_path}"
        )
        logger.debug(f"ncbi-genome-download arguments: {ncbi_kwargs}")

        exit_code: int = ngd.download(**ncbi_kwargs)
        if exit_code != 0:
            logger.error(f"ncbi-genome-download failed with exit code {exit_code}.")
            raise ConnectionError(
                "ncbi-genome-download did not successfully download the genomes."
            )

        logger.info("Genome download completed successfully.")
        return genome_output_dir, metadata_table_path

    def _filter_genomes_by_unique_taxid(
        self, genome_dir: pathlib.Path, metadata_table_path: pathlib.Path
    ) -> List[pathlib.Path]:
        """
        Filters downloaded genomes to include only those with unique strain-level taxIDs.
        """
        logger.info(
            f"Filtering genomes for unique strain taxIDs using metadata: {metadata_table_path}"
        )
        if not Path.exists(metadata_table_path):
            logger.warning(
                f"Metadata file not found at {metadata_table_path}, cannot filter by unique taxID. Using all genomes."
            )
            return list(genome_dir.glob("*fna.gz"))

        accessions_df = pd.read_csv(metadata_table_path, sep="\t")
        if "assembly_accession" in accessions_df.columns:
            accessions_df = accessions_df.set_index("assembly_accession")

        has_taxid: pd.DataFrame | pd.Series | Any = accessions_df["taxid"].notna()
        is_unique_strain_taxid = (
            accessions_df["taxid"] != accessions_df["species_taxid"]
        )
        filtered_accessions_df = accessions_df[has_taxid & is_unique_strain_taxid]

        filtered_metadata_path = (
            metadata_table_path.parent / f"filtered_{metadata_table_path.name}"
        )
        filtered_accessions_df.to_csv(filtered_metadata_path, sep="\t")
        logger.info(f"Filtered metadata saved to: {filtered_metadata_path}")

        if "local_filename" not in filtered_accessions_df.columns:
            logger.error(
                "'local_filename' column not found in metadata. Cannot determine file paths for unique taxID genomes."
            )
            return []

        filtered_genome_files: List[pathlib.Path] = []
        for rel_path_str in filtered_accessions_df["local_filename"]:
            potential_path = genome_dir / pathlib.Path(rel_path_str).name
            if Path.exists(potential_path):
                filtered_genome_files.append(potential_path)
            else:
                logger.warning(
                    f"Genome file listed in metadata not found: {potential_path} (original rel_path: {rel_path_str})"
                )

        logger.info(
            f"Found {len(filtered_genome_files)} genomes after filtering for unique strain taxIDs."
        )
        return filtered_genome_files

    def _get_genome_file_list(
        self,
    ) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        """
        Determines the list of genome FASTA files to process.
        """
        if self.args.custom:
            custom_path = pathlib.Path(self.args.custom)
            if not custom_path.is_dir():
                raise NotADirectoryError(
                    f"Custom genome directory not found: {custom_path}"
                )
            logger.info(f"Using custom genome files from: {custom_path}")
            genome_files = list(custom_path.glob("*.fna")) + list(
                custom_path.glob("*.fna.gz")
            )
            return genome_files, None
        else:
            genome_download_dir, metadata_file = self._download_genomes_from_ncbi()
            all_downloaded_files = list(genome_download_dir.glob("*fna.gz"))
            if self.args.unique_taxid:
                return self._filter_genomes_by_unique_taxid(
                    genome_download_dir, metadata_file
                ), metadata_file
            else:
                return all_downloaded_files, metadata_file

    def _extract_strain_name_from_metadata(
        self, genome_file_path: pathlib.Path, metadata_df: Optional[pd.DataFrame]
    ) -> str:
        """
        Extracts a descriptive strain name using the metadata DataFrame.
        """
        base_name = genome_file_path.name.replace(".fna.gz", "").replace(".fna", "")
        if metadata_df is None or metadata_df.empty:
            return base_name

        parts = base_name.split("_")
        if len(parts) >= 2:
            accession_match = "_".join(parts[:2])
        else:
            accession_match = base_name

        try:
            if accession_match in metadata_df.index:
                record = metadata_df.loc[accession_match]
                org_name = record.get("organism_name", base_name)
                strain_info = record.get("infraspecific_name", "")
                if pd.isna(strain_info) or strain_info == "nan":
                    strain_info = ""
                strain_info = str(strain_info).replace("strain=", "").strip()
                full_name = org_name
                if strain_info:
                    full_name += f" strain={strain_info}"
                full_name += f" ({accession_match})"
                return full_name
            else:
                logger.warning(
                    f"Accession '{accession_match}' (derived from {base_name}) not found in metadata. Using filename as identifier."
                )
                return base_name
        except KeyError:
            logger.warning(
                f"Metadata lookup failed for {accession_match}. Using filename {base_name} as identifier."
            )
            return base_name

    def _get_genome_names_for_files(
        self,
        genome_files: List[pathlib.Path],
        metadata_file_path: Optional[pathlib.Path],
    ) -> List[str]:
        """
        Generates descriptive names for each genome file.
        """
        logger.info("Generating names for genome files.")
        if not genome_files:
            return []

        metadata_df: Optional[pd.DataFrame] = None
        if metadata_file_path and Path.exists(metadata_file_path):
            try:
                metadata_df = pd.read_csv(metadata_file_path, sep="\t")
                if "assembly_accession" in metadata_df.columns:
                    metadata_df = metadata_df.set_index("assembly_accession")
                else:
                    logger.warning(
                        f"'assembly_accession' column not found in {metadata_file_path}. Cannot use it for name generation effectively."
                    )
                    metadata_df = None
            except Exception as e:
                logger.error(
                    f"Error reading metadata file {metadata_file_path}: {e}. Proceeding with filenames."
                )
                metadata_df = None

        genome_names: List[str] = []
        for gf_path in genome_files:
            genome_names.append(
                self._extract_strain_name_from_metadata(gf_path, metadata_df)
            )

        if len(genome_files) != len(genome_names):
            raise RuntimeError(
                "Mismatch between number of genome files and generated names."
            )

        if len(set(genome_names)) < len(genome_names):
            logger.warning(
                "Duplicate strain names detected. Appending suffixes to ensure uniqueness."
            )
            counts = Counter(genome_names)
            name_map: Dict[str, int] = {name: 0 for name in counts if counts[name] > 1}
            unique_genome_names = []
            for name in genome_names:
                if counts[name] > 1:
                    name_map[name] += 1
                    unique_genome_names.append(f"{name}_{name_map[name]}")
                else:
                    unique_genome_names.append(name)
            genome_names = unique_genome_names

        return genome_names

    @staticmethod
    def fast_kmers_numpy(seq: str, k: int) -> Set[bytes]:
        """
        Returns a set of unique k-mers as np.void blocks. Handles any sequence length.
        """
        seq = seq.upper()
        arr = np.frombuffer(seq.encode("ascii"), dtype="S1")
        n = arr.shape[0] - k + 1
        if n <= 0:
            return set()

        windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=k)
        void_windows = windows.view(np.dtype((np.void, k))).ravel()
        return {vw.tobytes() for vw in void_windows}

    def _py_reverse_complement_db(self, kmer_bytes: bytes) -> bytes:
        """Computes the reverse complement of a DNA sequence."""
        return kmer_bytes.translate(self.PY_RC_TRANSLATE_TABLE)[::-1]

    def _extract_kmers_from_bytes(self, sequence_bytes: bytes, k: int) -> List[bytes]:
        """Extract k-mers using Rust implementation if available, else Python (canonical)."""
        # Ensure sequence is uppercase before kmer extraction
        # Rust implementation is expected to handle this, or ensure input is uppercase
        upper_sequence_bytes = sequence_bytes.upper()

        # DEV NOTE / BIOINFORMATICS INTEGRITY:
        # The Rust k-mer extraction function (`kmer_counter_rs.extract_kmers_rs`)
        # is assumed to return CANONICAL k-mers (the lexicographically smaller of
        # a k-mer and its reverse complement).
        # This is crucial for consistency with the classification step, which typically
        # processes reads by generating canonical k-mers to ensure strand-insensitivity.
        # If the Rust implementation does not provide canonical k-mers, or provides
        # raw k-mers, this could lead to mismatches during classification unless
        # the k-mers from reads are also processed in the exact same (raw) way.
        # The Python fallback in this script (`_py_extract_canonical_kmers` logic within
        # `_extract_kmers_from_bytes`) *does* generate canonical k-mers.
        # If kmer_counter_rs behavior is uncertain or needs to be raw, ensure downstream
        # classification tools are aligned, or add an explicit canonicalization step here
        # (though that would preferably be handled by the Rust library itself for performance).

        if _extract_kmers_func is not None:  # Rust path
            try:
                # Assuming Rust version already returns canonical k-mers or raw if specified by its own logic
                kmers_from_rust = _extract_kmers_func(upper_sequence_bytes, k)
                if self.args.skip_n_kmers:
                    # Assuming kmer_counter_rs returns a list of bytes objects
                    kmers_from_rust = [
                        kmer for kmer in kmers_from_rust if b"N" not in kmer
                    ]  # N is already uppercase due to upper_sequence_bytes
                return kmers_from_rust

            except Exception as e:  # pragma: no cover - runtime fallback
                logger.error(
                    f"Rust k-mer extraction failed: {e}. Falling back to Python canonical implementation."
                )

        # Python fallback: extract canonical k-mers
        kmers_list: List[bytes] = []
        if len(upper_sequence_bytes) < k:
            return kmers_list

        with memoryview(upper_sequence_bytes) as seq_view:
            for i in range(len(upper_sequence_bytes) - k + 1):
                kmer_candidate_bytes = seq_view[
                    i : i + k
                ].tobytes()  # kmer candidate before canonicalization

                if (
                    self.args.skip_n_kmers and b"N" in kmer_candidate_bytes
                ):  # Check for 'N'
                    continue

                rc_kmer = self._py_reverse_complement_db(kmer_candidate_bytes)
                kmers_list.append(
                    kmer_candidate_bytes if kmer_candidate_bytes <= rc_kmer else rc_kmer
                )

        return kmers_list

    def _process_single_fasta_for_kmers(
        self,
        genome_file_info: Tuple[
            pathlib.Path, str, int
        ],  # (file_path, strain_name, strain_idx)
        kmer_length: int,
        # num_total_strains: int, # No longer needed for this worker's direct task
    ) -> Tuple[str, int, Set[bytes]]:  # (strain_name, strain_idx, strain_kmers_set)
        """
        Extracts k-mers from a single FASTA file for one strain and returns them as a set.

        Args:
            genome_file_info: Tuple containing the path to the FASTA file,
                              the assigned strain name, and its column index.
            kmer_length: The length of k-mers to extract.

        Returns:
            A tuple: (strain_name, strain_idx, set_of_unique_kmer_bytes).
        """
        genome_file, strain_name, strain_idx = genome_file_info
        logger.debug(
            f"Processing {genome_file} for strain '{strain_name}' (index {strain_idx})."
        )

        strain_kmers: Set[bytes] = set()
        try:
            with open_file_transparently(genome_file) as f_handle:
                for record in SeqIO.parse(f_handle, "fasta"):
                    seq_str = str(record.seq)
                    # Basic validation for sequence characters if necessary, e.g., ensure they are in ACGTN
                    # For performance, this might be skipped if input data is trusted
                    # or handled by the k-mer extraction function itself.
                    seq_bytes = seq_str.encode(
                        "utf-8"
                    )  # Make sure this is compatible with _extract_kmers_from_bytes

                    # _extract_kmers_from_bytes now handles making sequence uppercase and canonical if Python fallback
                    kmers_from_seq = self._extract_kmers_from_bytes(
                        seq_bytes, kmer_length
                    )
                    strain_kmers.update(kmers_from_seq)  # update with list of bytes

        except Exception as e:
            logger.error(
                f"Error processing FASTA file {genome_file} for strain {strain_name}: {e}"
            )

            # Depending on desired robustness, could return empty set or re-raise
            # For now, return empty set for this genome on error to not halt entire batch
            return strain_name, strain_idx, set()

        logger.debug(
            f"Extracted {len(strain_kmers)} unique k-mers for strain '{strain_name}'."
        )

        return strain_name, strain_idx, strain_kmers

    def _build_kmer_database_parallel(
        self, genome_files: List[pathlib.Path], strain_names: List[str]
    ) -> pd.DataFrame:
        """
        Extracts k-mers from genomes in parallel and aggregates them in memory.
        """
        if not genome_files:
            logger.warning("No genome files provided to build database.")
            return pd.DataFrame()
        if len(genome_files) != len(strain_names):
            raise ValueError(
                "Mismatch between number of genome files and strain names."
            )

        num_genomes = len(genome_files)
        logger.info(
            f"Starting k-mer extraction for {num_genomes} genomes using {self.args.procs} processes."
        )

        # Rough estimate for memory warning
        # Assuming average 2 million unique k-mers per bacterial genome (highly variable)
        # and kmer_length around 30 (30 bytes per kmer).
        # This doesn't account for Python object overhead or the master_kmer_dict structure itself.
        estimated_total_unique_kmers_approx = num_genomes * 2_000_000
        # A very rough threshold, e.g. 1 billion k-mers might take ~30GB for just kmer bytes
        if estimated_total_unique_kmers_approx > 500_000_000:  # 500 million k-mers
            logger.warning(
                f"Processing {num_genomes} genomes. Estimated total unique k-mers could be very large, "
                "potentially leading to high memory usage during in-memory aggregation. "
                "Monitor memory closely."
            )

        tasks = []
        for i in range(num_genomes):
            tasks.append((genome_files[i], strain_names[i], i))

        worker_function = partial(
            self._process_single_fasta_for_kmers,
            kmer_length=self.args.kmerlen,
            # num_total_strains is no longer passed to the worker
        )

        extraction_results: List[Tuple[str, int, Set[bytes]]] = []
        with mp.Pool(processes=self.args.procs) as pool:
            logger.info(
                "Starting parallel k-mer extraction (results collected in memory)."
            )

            # Results from worker: (strain_name, strain_idx, strain_kmers_set)
            for result in tqdm(
                pool.imap_unordered(worker_function, tasks),
                total=len(tasks),
                desc="Extracting k-mers",
            ):
                extraction_results.append(result)

        logger.info("Aggregating k-mers into database matrix (in-memory).")
        master_kmer_dict: Dict[bytes, np.ndarray] = defaultdict(
            lambda: np.zeros(len(strain_names), dtype=bool)
        )

        # Sum of lengths of all kmer sets for a more accurate progress bar for aggregation
        total_kmers_to_aggregate = sum(
            len(s_kmers) for _, _, s_kmers in extraction_results
        )

        for res_strain_name, res_strain_idx, strain_kmers_set in tqdm(
            extraction_results,
            total=len(extraction_results),  # This total is for number of genomes
            desc="Aggregating k-mers (genome by genome)",
        ):
            # Could make inner loop tqdm for kmer-level progress, but might be too verbose
            # for kmer_bytes in tqdm(strain_kmers_set, desc=f"Aggregating {res_strain_name}", leave=False):
            for kmer_bytes in strain_kmers_set:
                master_kmer_dict[kmer_bytes][res_strain_idx] = True

        # No temporary directory or files to clean up.

        if not master_kmer_dict:
            logger.warning(
                "No k-mers were extracted or aggregated. Resulting database will be empty."
            )
            return pd.DataFrame(columns=strain_names)

        logger.info(f"Total unique k-mers found: {len(master_kmer_dict)}")

        # Convert the master dictionary to a DataFrame
        # Sort k-mers (index) for consistent database output, though this can be slow for many k-mers
        # sorted_kmer_keys = sorted(master_kmer_dict.keys())
        # Using dict directly is faster if order doesn't strictly matter or can be handled later
        kmer_matrix_df = pd.DataFrame.from_dict(
            master_kmer_dict,
            orient="index",  # k-mers as rows
            columns=strain_names,  # strains as columns
            dtype=bool,  # Ensure boolean type
        )
        # kmer_matrix_df = kmer_matrix_df.reindex(sorted_kmer_keys) # Apply sort if using sorted_kmer_keys

        return kmer_matrix_df

    def _save_database_to_parquet(self, database_df: pd.DataFrame) -> None:
        """
        Saves the k-mer database DataFrame to a Parquet file.
        """
        output_path = self.base_path / (self.output_db_name + ".db.parquet")
        logger.info(f"Saving k-mer database to (Parquet format): {output_path}")
        try:
            database_df.to_parquet(output_path, index=True)
            logger.info("Database saved successfully to Parquet.")
        except Exception as e:
            logger.error(
                f"Failed to save database to {output_path} (Parquet format): {e}"
            )
            raise

    def create_database(self) -> None:
        """
        Main public method to orchestrate the entire database creation process.
        """
        logger.info("Starting k-mer database creation workflow.")

        genome_files, metadata_file_path = self._get_genome_file_list()
        if not genome_files:
            logger.error("No genome files found or specified. Cannot build database.")
            return

        logger.info(f"Found {len(genome_files)} genome files to process.")

        strain_identifiers = self._get_genome_names_for_files(
            genome_files, metadata_file_path
        )
        if not strain_identifiers:
            logger.error("Could not determine strain identifiers for genome files.")
            return

        kmer_database_df = self._build_kmer_database_parallel(
            genome_files, strain_identifiers
        )

        if kmer_database_df.empty:
            logger.warning(
                "K-mer database construction resulted in an empty DataFrame."
            )
        else:
            logger.info(
                f"Constructed k-mer database DataFrame with shape: {kmer_database_df.shape} "
                f"(kmers x strains)."
            )

        self._save_database_to_parquet(kmer_database_df)
        logger.info("K-mer database creation workflow finished.")


def get_cli_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the ArgumentParser for the script.
    """
    parser = argparse.ArgumentParser(
        description="StrainR Database Builder: Downloads genomes and creates a k-mer presence/absence database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    genome_source_group = parser.add_mutually_exclusive_group(required=True)
    genome_source_group.add_argument(
        "-t",
        "--taxid",
        type=str,
        help="Species taxonomic ID from NCBI from which all strains will be downloaded.",
    )
    genome_source_group.add_argument(
        "-f",
        "--assembly_accessions",
        type=str,
        help="Path to a file listing assembly accessions (one per line) to download from NCBI.",
    )
    genome_source_group.add_argument(
        "-g",
        "--genus",
        type=str,
        help="Genus name for which to download genomes from NCBI.",
    )
    genome_source_group.add_argument(
        "--custom",
        type=str,
        help="Path to a folder containing custom genome FASTA files (.fna, .fna.gz) for database creation.",
    )

    parser.add_argument(
        "-k",
        "--kmerlen",
        type=int,
        default=31,
        help="Length of k-mers to extract. Default: 31, suitable for bacterial genomes.",
    )
    parser.add_argument(
        "-l",
        "--assembly_levels",
        choices=["complete", "chromosome", "scaffold", "contig"],
        type=str,
        default="complete",
        help="Assembly level(s) of genomes to download from NCBI (e.g., 'contig' includes 'complete', 'chromosome', 'scaffold').",
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["refseq", "genbank"],
        type=str,
        default="refseq",
        help="NCBI database source for downloads (refseq or genbank).",
    )
    parser.add_argument(
        "-p",
        "--procs",
        type=int,
        default=4,
        help="Number of processor cores to use for parallel tasks.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="strainr_kmer_database",
        help="Output name prefix for the database file (e.g., 'my_db' -> 'my_db.db.parquet').",
    )
    parser.add_argument(
        "--unique-taxid",
        action="store_true",
        help="Flag to only include genomes from NCBI that have a unique strain-level taxonomic ID.",
    )
    parser.add_argument(
        "--skip-n-kmers",
        action="store_true",
        help="If set, k-mers containing 'N' (ambiguous) bases will be excluded from the database.",
    )

    return parser


if __name__ == "__main__":
    logger.info("Strainr Database Building Script Started.")

    arg_parser = get_cli_parser()
    cli_args = arg_parser.parse_args()

    try:
        builder = DatabaseBuilder(args=cli_args)
        builder.create_database()
    except Exception as e:
        logger.critical(
            f"A critical error occurred during database creation: {e}", exc_info=True
        )
        sys.exit(1)

    logger.info("StrainR Database Building Script Finished Successfully.")
