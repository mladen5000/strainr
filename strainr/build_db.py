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
"""

import contextlib
import logging
import multiprocessing as mp
import pathlib
import shutil
import subprocess
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import ncbi_genome_download as ngd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from Bio import SeqIO
from pydantic import BaseModel, DirectoryPath, Field, FilePath, model_validator
from tqdm import tqdm
from typing_extensions import Annotated

from .utils import check_external_commands, open_file_transparently

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
    from kmer_counter_rs import extract_kmer_rs

    _extract_kmers_func = extract_kmer_rs
    _RUST_KMER_COUNTER_AVAILABLE = True
    # _extract_kmers_func = None
    # _RUST_KMER_COUNTER_AVAILABLE = False
    logger.info(
        "Successfully imported Rust k-mer counter. Using Rust implementation for k-mer extraction."
    )
except Exception as e:  # pragma: no cover - rust module optional
    _extract_kmers_func = None  # Fallback if Rust is not available
    _RUST_KMER_COUNTER_AVAILABLE = False  # Fallback if Rust is not available
    logger.warning(
        f"Rust k-mer counter not available ({e}). Falling back to Python implementation."
    )


# --- Pydantic Model for CLI Arguments ---
class BuildDBArgs(BaseModel):
    taxid: Optional[str] = Field(None, description="Species taxonomic ID from NCBI.")
    assembly_accessions: Optional[FilePath] = Field(
        None, description="Path to a file listing assembly accessions."
    )
    genus: Optional[str] = Field(None, description="Genus name for NCBI download.")
    custom: Optional[DirectoryPath] = Field(
        None, description="Path to folder with custom genome FASTA files."
    )

    kmerlen: int = Field(31, description="Length of k-mers.", gt=0)
    assembly_levels: Literal["complete", "chromosome", "scaffold", "contig"] = Field(
        "complete", description="Assembly level(s) for NCBI download."
    )
    source: Literal["refseq", "genbank"] = Field(
        "refseq", description="NCBI database source."
    )
    procs: int = Field(4, description="Number of processor cores.", gt=0)
    out: str = Field(
        "strainr_kmer_database", description="Output name prefix for the database file."
    )
    unique_taxid: bool = Field(
        False, description="Filter NCBI genomes for unique strain-level taxID."
    )
    skip_n_kmers: bool = Field(False, description="Exclude k-mers with 'N' bases.")

    @model_validator(mode="after")
    def check_genome_source(cls, values):
        sources = [
            values.taxid,
            values.assembly_accessions,
            values.genus,
            values.custom,
        ]
        if sum(s is not None for s in sources) != 1:
            raise ValueError(
                "Exactly one of --taxid, --assembly_accessions, --genus, or --custom must be specified."
            )
        if values.assembly_accessions and not values.assembly_accessions.is_file():
            raise ValueError(
                f"Assembly accessions file not found: {values.assembly_accessions}"
            )
        if values.custom and not values.custom.is_dir():
            raise ValueError(f"Custom genome directory not found: {values.custom}")
        return values


class DatabaseBuilder:
    """
    Manages the k-mer database creation workflow.

    This class encapsulates all steps from genome download to k-mer matrix
    generation and saving.
    """

    PY_RC_TRANSLATE_TABLE = bytes.maketrans(b"ACGTN", b"TGCAN")

    def __init__(self, args: BuildDBArgs) -> None:
        logger.info("Initializing DatabaseBuilder.")
        self.args: BuildDBArgs = args
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
        # For ncbi-genome-download, if assembly_accessions are provided,
        # the assembly_levels parameter is effectively ignored by ngd,
        # and it downloads whatever accessions are specified.
        # So, we can simplify this.
        if self.args.assembly_accessions:
            return "all"

        # For other cases (taxid, genus), map the Pydantic literal to ngd's expected string.
        level_mapping = {
            "complete": "complete",
            "chromosome": "complete,chromosome",
            "scaffold": "complete,chromosome,scaffold",
            "contig": "complete,chromosome,scaffold,contig",
        }
        selected_level = self.args.assembly_levels
        if selected_level in level_mapping:
            return level_mapping[selected_level]
        else:
            # This case should ideally not be reached if Pydantic validation is correct
            raise ValueError(f"Invalid assembly level selected: {selected_level}")

    def _download_genomes_from_ncbi(self) -> Tuple[pathlib.Path, pathlib.Path]:
        """
        Downloads genomes from NCBI using `ncbi-genome-download`.
        """
        logger.info("Preparing to download genomes from NCBI.")
        assembly_level_str = self._parse_assembly_level()

        ncbi_kwargs: Dict[str, Any] = {
            "flat_output": True,  # Keep downloaded files in a flat directory structure
            "groups": "bacteria",  # Assuming we are always downloading bacteria
            "file_formats": "fasta",
            "section": self.args.source,  # 'refseq' or 'genbank'
            "parallel": self.args.procs,
            "assembly_levels": assembly_level_str,
        }

        genome_target_dir_suffix: str = ""
        # Pydantic model validation ensures exactly one of these is set
        if self.args.taxid:
            ncbi_kwargs["species_taxids"] = self.args.taxid
            genome_target_dir_suffix = f"s{self.args.taxid}"
        elif self.args.assembly_accessions:
            # Pydantic ensures this is a FilePath and exists
            ncbi_kwargs["assembly_accessions"] = str(self.args.assembly_accessions)
            genome_target_dir_suffix = f"acc_{self.args.assembly_accessions.stem}"
        elif self.args.genus:
            ncbi_kwargs["genera"] = self.args.genus
            genome_target_dir_suffix = f"g{self.args.genus.replace(' ', '_')}"
        # No 'else' needed due to Pydantic validation

        genome_output_dir = (
            self.output_db_dir / f"ncbi_genomes_{genome_target_dir_suffix}"
        )
        metadata_table_path = (
            genome_output_dir / f"metadata_summary_{genome_target_dir_suffix}.tsv"
        )

        ncbi_kwargs.update({
            "output": genome_output_dir,
            "metadata_table": metadata_table_path,
        })

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
            # Pydantic ensures this is a DirectoryPath and exists
            logger.info(f"Using custom genome files from: {self.args.custom}")
            genome_files = list(self.args.custom.glob("*.fna")) + list(
                self.args.custom.glob("*.fna.gz")
            )
            return genome_files, None
        else:
            genome_download_dir, metadata_file = self._download_genomes_from_ncbi()
            all_downloaded_files = list(genome_download_dir.glob("*fna.gz"))
            if self.args.unique_taxid:  # This attribute comes from BuildDBArgs
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
        """
        Extracts canonical k-mers from a sequence.

        This method prioritizes using the Rust implementation (`kmer_counter_rs.extract_kmers_rs`)
        if it's available (_extract_kmers_func is not None). The Rust function is expected
        to return canonical k-mers and handle the `skip_n_kmers` logic internally via its
        `perform_strict_dna_check` argument, which is passed from `_process_and_write_kmers_worker`.

        If the Rust implementation is unavailable or fails, this method falls back to a
        Python-based canonical k-mer extraction.

        Args:
            sequence_bytes: The DNA sequence as bytes.
            k: The k-mer length.

        Returns:
            A list of canonical k-mers (bytes).
        """
        # Ensure sequence is uppercase before kmer extraction
        # Rust implementation also expects uppercase, or handles it internally.
        upper_sequence_bytes = sequence_bytes.upper()

        # DEV NOTE / BIOINFORMATICS INTEGRITY & skip_n_kmers:
        # 1. Canonical K-mers: The Rust k-mer extraction function (`kmer_counter_rs.extract_kmers_rs`)
        #    is responsible for returning CANONICAL k-mers. This is vital for consistency
        #    with downstream classification tools.
        # 2. skip_n_kmers: The `skip_n_kmers` logic (handling k-mers with 'N' or other non-ACGT bases)
        #    is primarily delegated to the Rust function via its `perform_strict_dna_check`
        #    parameter. This parameter is set based on `self.args.skip_n_kmers` in the
        #    `_process_and_write_kmers_worker` method, which calls this extraction logic.
        #    The Python-side filter for 'N's on `kmers_from_rust` below is a secondary check.
        #    If the Rust function *guarantees* that `perform_strict_dna_check=True` means
        #    no k-mers with 'N' (or any non-ACGT) are returned, this Python filter is redundant
        #    for the Rust path. Current tests suggest consistency, but this is an area
        #    to watch if Rust library behavior changes.

        if _extract_kmers_func is not None:  # Rust path is available
            try:
                # Call Rust function with three arguments: (sequence_bytes, k, skip_n_kmers)
                skip_n_kmers = getattr(self.args, "skip_n_kmers", False)
                kmers_from_rust = _extract_kmers_func(
                    upper_sequence_bytes, k, skip_n_kmers
                )
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
                    getattr(self.args, "skip_n_kmers", False)
                    and b"N" in kmer_candidate_bytes
                ):
                    continue

                rc_kmer = self._py_reverse_complement_db(kmer_candidate_bytes)
                kmers_list.append(
                    kmer_candidate_bytes if kmer_candidate_bytes <= rc_kmer else rc_kmer
                )

        return kmers_list

    def _process_single_fasta_for_kmers(
        self,
        genome_file_info: Tuple[
            pathlib.Path, str, int, Optional[pathlib.Path]
        ],  # (file_path, strain_name, strain_idx, optional_output_path)
        kmer_length: int,  # This comes from self.args.kmerlen
        skip_n_kmers: bool,  # This comes from self.args.skip_n_kmers
        num_total_strains: Optional[int] = None,
    ) -> Union[
        Tuple[str, int, Set[bytes]],
        Tuple[str, int, int, pathlib.Path],
        Tuple[str, int, pathlib.Path],
    ]:
        """
        Extracts k-mers from a single FASTA file.

        Args:
            genome_file_info: Tuple containing the path to the FASTA file,
                the assigned strain name, its column index and optionally a
                path where k-mers should be written.
            kmer_length: The length of k-mers to extract.
            skip_n_kmers: Boolean, whether to skip k-mers with 'N'.

        Returns:
            * ``(strain_name, strain_idx, strain_kmers_set)`` when no output
              path is provided.
            * ``(strain_name, strain_idx, written_kmer_count, output_path)`` or
              ``(strain_name, written_kmer_count, output_path)`` when an output
              path is provided depending on whether ``num_total_strains`` is
              given.
        """
        genome_file, strain_name, strain_idx, output_path = genome_file_info
        logger.debug(
            f"Processing {genome_file} for strain '{strain_name}' (index {strain_idx})."
        )

        strain_kmers: Set[bytes] = set()
        written_kmer_count = 0
        try:
            with open_file_transparently(genome_file) as f_handle:
                if output_path:
                    out_f = open(output_path, "w", encoding="utf-8")
                else:
                    out_f = None
                with out_f if out_f else contextlib.nullcontext():
                    for record in SeqIO.parse(f_handle, "fasta"):
                        if record is None:
                            continue
                        if record.seq is None:
                            continue
                        seq_str = str(record.seq).upper()
                        for i in range(0, len(seq_str) - kmer_length + 1):
                            kmer = seq_str[i : i + kmer_length]
                            if skip_n_kmers and "N" in kmer:
                                continue
                            if out_f:
                                out_f.write(f"{kmer}\n")
                                written_kmer_count += 1
                            else:
                                strain_kmers.add(kmer.encode("utf-8"))

        except Exception as e:
            logger.error(
                f"Error processing FASTA file {genome_file} for strain {strain_name}: {e}"
            )

            if output_path:
                return (
                    strain_name,
                    strain_idx,
                    0,
                    output_path,
                )
            return strain_name, strain_idx, set()

        if output_path:
            logger.debug(
                f"Extracted {written_kmer_count} k-mers for strain '{strain_name}'."
            )
            if num_total_strains is not None:
                return (
                    strain_name,
                    strain_idx,
                    written_kmer_count,
                    output_path,
                )
            return strain_name, written_kmer_count, output_path

        logger.debug(
            f"Extracted {len(strain_kmers)} unique k-mers for strain '{strain_name}'."
        )

        return strain_name, strain_idx, strain_kmers

    def _build_kmer_database_parallel(
        self, genome_files: List[pathlib.Path], strain_names: List[str]
    ) -> Optional[pathlib.Path]:
        """
        Extracts k-mers from genomes in parallel and builds the Parquet database
        using a memory-scalable disk-based sort-reduce strategy.

        This method avoids loading all k-mers into memory at once.
        1.  Map: Each worker process extracts k-mers from one genome and writes
            (k-mer, strain_id) pairs to a temporary file.
        2.  Sort: All temporary files are concatenated and sorted on disk using
            the system's `sort` command.
        3.  Reduce: The large sorted file is read line-by-line, and k-mers are
            grouped to build presence/absence vectors, which are written
            incrementally to the final Parquet file.

        Args:
            genome_files: List of paths to genome FASTA files.
            strain_names: List of corresponding strain names for columns.

        Returns:
            The path to the newly created Parquet database file.
        """
        if not genome_files:
            logger.warning("No genome files provided to build database.")
            return None

        if len(genome_files) != len(strain_names):
            raise ValueError(
                "Mismatch between number of genome files and strain names."
            )

        num_genomes = len(genome_files)
        logger.info(
            f"Starting scalable k-mer extraction for {num_genomes} genomes using {self.args.procs} processes."
        )

        # --- Setup Temporary Directory ---
        temp_dir = self.output_db_dir / "tmp_kmer_parts"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        # --- 1. MAP PHASE ---
        # Each worker extracts k-mers and writes to a unique temp file.
        tasks = [
            (
                genome_files[i],  # genome_file path
                strain_names[i],  # strain_name string
                i,  # strain_idx integer
                self.args.kmerlen,  # kmer_length integer
                self.args.skip_n_kmers,  # skip_n_kmers boolean (passed to worker)
                temp_dir,  # temp_dir path
            )
            for i in range(num_genomes)
        ]

        logger.info("Starting parallel k-mer extraction (writing to disk)...")

        # Collect results and fail fast if any worker fails
        worker_results = []
        with mp.Pool(processes=self.args.procs) as pool:
            for result in tqdm(
                pool.imap_unordered(self._process_and_write_kmers_worker, tasks),
                total=len(tasks),
                desc="Extracting k-mers (Map)",
            ):
                worker_results.append(result)
        # If any worker failed, an exception will be raised and execution will stop here

        logger.info("Map phase completed. All k-mers written to temporary files.")

        # --- Check that all expected .part.gz files exist ---
        expected_part_files = [temp_dir / f"{i}.part.gz" for i in range(num_genomes)]
        missing_files = [str(f) for f in expected_part_files if not f.exists()]
        if missing_files:
            logger.error(
                f"Missing expected k-mer part files: {missing_files}. Aborting before sort/concat phase."
            )
            raise FileNotFoundError(
                f"Missing expected k-mer part files: {missing_files}"
            )

        # --- Check that all expected .part.gz files exist ---
        expected_part_files = [temp_dir / f"{i}.part.gz" for i in range(num_genomes)]
        missing_files = [str(f) for f in expected_part_files if not f.exists()]
        if missing_files:
            logger.error(
                f"Missing expected k-mer part files: {missing_files}. Aborting before sort/concat phase."
            )
            raise FileNotFoundError(
                f"Missing expected k-mer part files: {missing_files}"
            )

        try:
            # --- 2. SORT PHASE ---
            # Concatenate and sort the temporary files on disk.
            all_parts_file = temp_dir / "all_kmer_parts.tsv"
            sorted_parts_file = temp_dir / "all_kmer_parts.sorted.tsv"

            logger.info(f"Fast concatenating temporary files into {all_parts_file}...")
            # Optimized concatenation using system commands (much faster than Python shutil)
            part_files = list(temp_dir.glob("*.part.gz"))
            if len(part_files) > 100:  # For many files, use find + zcat
                concat_command = f"find {temp_dir} -name '*.part.gz' -exec zcat {{}} + > {all_parts_file}"
            else:  # For fewer files, direct zcat
                part_paths = " ".join(str(f) for f in part_files)
                concat_command = f"zcat {part_paths} > {all_parts_file}"

            subprocess.run(concat_command, shell=True, check=True)

            logger.info("Optimized sorting all k-mer parts on disk (parallel sort)...")
            # Use parallel sort with memory buffer and compression for much better performance
            sort_command = (
                f"LC_ALL=C sort -k1,1 --parallel={self.args.procs} "
                f"-S 2G --compress-program=gzip {all_parts_file} -o {sorted_parts_file}"
            )
            process = subprocess.run(
                sort_command, shell=True, check=True, capture_output=True, text=True
            )
            if process.returncode != 0:
                logger.error(f"Sorting failed. Stderr: {process.stderr}")
                raise RuntimeError("Disk-based sort command failed.")
            logger.info("Sort phase completed.")

            # --- 3. REDUCE PHASE (OPTIMIZED) ---
            # Group sorted k-mers and write to Parquet file in batches.
            output_path = self.base_path / (self.output_db_name + ".db.parquet")
            logger.info(f"Aggregating sorted k-mers into Parquet file: {output_path}")

            # Define the schema for the output Parquet file.
            schema_fields = [pa.field("kmer", pa.binary())]
            for name in strain_names:
                schema_fields.append(pa.field(name, pa.bool_()))

            # Add custom metadata
            custom_metadata = {
                b"strainr_kmerlen": str(self.args.kmerlen).encode("utf-8"),
                b"strainr_skip_n_kmers": str(self.args.skip_n_kmers).encode("utf-8"),
            }
            schema = pa.schema(schema_fields, metadata=custom_metadata)
            logger.info(f"Parquet schema metadata set with: {custom_metadata}")

            # Batch processing for much better performance
            BATCH_SIZE = 10000  # Process 10k k-mers at a time
            batch_data = {field.name: [] for field in schema_fields}

            with pq.ParquetWriter(output_path, schema) as writer:
                with open(sorted_parts_file, "r") as f:
                    grouped_iterator = groupby(
                        f, key=lambda line: line.split("\t", 1)[0]
                    )

                    batch_count = 0
                    for kmer_hex, group in tqdm(
                        grouped_iterator, desc="Aggregating k-mers (Reduce)"
                    ):
                        presence_vector = np.zeros(num_genomes, dtype=bool)

                        # Process all lines for this k-mer
                        for line in group:
                            try:
                                strain_idx = int(line.strip().split("\t")[1])
                                presence_vector[strain_idx] = True
                            except (IndexError, ValueError) as e:
                                logger.warning(
                                    f"Skipping malformed line in sorted file: '{line.strip()}'. Error: {e}"
                                )
                                continue

                        # Add to batch
                        batch_data["kmer"].append(bytes.fromhex(kmer_hex))
                        for i, name in enumerate(strain_names):
                            batch_data[name].append(presence_vector[i])

                        batch_count += 1

                        # Write batch when full
                        if batch_count >= BATCH_SIZE:
                            table = pa.Table.from_pydict(batch_data, schema=schema)
                            writer.write_table(table)
                            # Clear batch
                            batch_data = {field.name: [] for field in schema_fields}
                            batch_count = 0

                    # Write final batch if any remaining
                    if batch_count > 0:
                        table = pa.Table.from_pydict(batch_data, schema=schema)
                        writer.write_table(table)

            logger.info(
                "Reduce phase completed. Parquet database created successfully at: "
                + str(output_path)
            )
            # The 'kmer' column is written as a regular column.
            # The StrainKmerDatabase class will handle setting this as index upon loading.
            # Metadata (kmerlen, skip_n_kmers) is already written with the schema.
            # No re-loading and re-saving needed here, which saves memory.

            return output_path

        finally:
            # --- Cleanup ---
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

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

        # Check for required external commands first
        try:
            check_external_commands(["sort", "zcat"])
            logger.info("External commands 'sort' and 'zcat' found.")
        except FileNotFoundError as e:
            logger.error(f"Missing required external command: {e}")
            # Depending on desired behavior, you might re-raise or sys.exit
            raise  # Re-raise to stop execution if commands are critical

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

        # This method now handles file creation and returns the path.
        # It no longer returns a DataFrame.
        db_path = self._build_kmer_database_parallel(genome_files, strain_identifiers)

        if not db_path or not db_path.exists():
            logger.error(
                "K-mer database construction failed to produce an output file."
            )
        else:
            # The logging about saving is now handled inside the build method.
            # We can just log the final success message.
            df_info = pd.read_parquet(
                db_path, columns=[]
            )  # Read only metadata for shape
            logger.info(
                f"Successfully constructed and saved k-mer database to {db_path} "
                f"with shape: ({len(df_info.index)}, {len(df_info.columns)}) "
                f"(kmers x strains)."
            )

        logger.info("K-mer database creation workflow finished.")

    @staticmethod
    def _process_and_write_kmers_worker(task):
        logger = logging.getLogger(__name__)
        logger.info(f"Worker starting for {task}")
        genome_file, strain_name, strain_idx, kmer_length, skip_n_kmers, temp_dir = task

        # Initialize a local extractor
        try:
            from kmer_counter_rs import extract_kmer_rs

            extract_func = extract_kmer_rs
        except ImportError:
            extract_func = None

        # Process the genome file
        strain_kmers = set()

        try:
            from Bio import SeqIO
            from .utils import open_file_transparently

            with open_file_transparently(genome_file) as f_handle:
                for record in SeqIO.parse(f_handle, "fasta"):
                    if record is None:
                        continue
                    if record.seq is None:
                        continue
                    seq_str = str(record.seq)
                    seq_bytes = seq_str.encode("utf-8")

                    if extract_func is not None:
                        try:
                            # Call Rust function with three arguments: (sequence_bytes, k, skip_n_kmers)
                            kmers_from_seq = extract_func(
                                seq_bytes.upper(), kmer_length
                            )
                            logger.debug(
                                f"Received {len(kmers_from_seq)} k-mers from Rust for {genome_file} (record: {record.id})"
                            )
                            strain_kmers.update(kmers_from_seq)
                        except Exception as rust_exc:  # pragma: no cover
                            logger.warning(
                                f"Rust k-mer extraction failed for {genome_file} (record: {record.id}): {rust_exc}. Falling back to Python for this sequence."
                            )
                            # Fallback to Python implementation
                            kmers_from_seq = (
                                DatabaseBuilder._py_extract_canonical_kmers_static(
                                    seq_bytes, kmer_length, skip_n_kmers
                                )
                            )
                            logger.debug(
                                f"Received {len(kmers_from_seq)} k-mers from Python fallback for {genome_file} (record: {record.id})"
                            )
                            strain_kmers.update(kmers_from_seq)
                    else:
                        # Python only path (Rust not available at all)
                        kmers_from_seq = (
                            DatabaseBuilder._py_extract_canonical_kmers_static(
                                seq_bytes, kmer_length, skip_n_kmers
                            )
                        )
                        logger.debug(
                            f"Received {len(kmers_from_seq)} k-mers from Python for {genome_file} (record: {record.id})"
                        )
                        strain_kmers.update(kmers_from_seq)

        except Exception as e:
            logger.error(f"Error processing {genome_file}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            # Raise exception so the main process knows this worker failed
            raise

        # Write to compressed temp file for better I/O performance
        temp_file_path = temp_dir / f"{strain_idx}.part.gz"
        import gzip

        with gzip.open(temp_file_path, "wt") as f_out:
            for kmer_bytes in strain_kmers:
                f_out.write(f"{kmer_bytes.hex()}\t{strain_idx}\n")

        logger.info(f"Worker finished for {task}")
        return temp_file_path

    @staticmethod
    def _py_extract_canonical_kmers_static(
        sequence_bytes: bytes, k: int, skip_n_kmers: bool
    ) -> List[bytes]:
        """Static version of canonical k-mer extraction for multiprocessing."""
        PY_RC_TRANSLATE_TABLE = bytes.maketrans(b"ACGTN", b"TGCAN")

        def _py_reverse_complement_static(kmer_bytes: bytes) -> bytes:
            return kmer_bytes.translate(PY_RC_TRANSLATE_TABLE)[::-1]

        kmers_list = []
        upper_sequence_bytes = sequence_bytes.upper()

        if len(upper_sequence_bytes) < k:
            return kmers_list

        with memoryview(upper_sequence_bytes) as seq_view:
            for i in range(len(upper_sequence_bytes) - k + 1):
                kmer_candidate_bytes = seq_view[i : i + k].tobytes()

                if skip_n_kmers and b"N" in kmer_candidate_bytes:
                    continue

                rc_kmer = _py_reverse_complement_static(kmer_candidate_bytes)
                kmers_list.append(
                    kmer_candidate_bytes if kmer_candidate_bytes <= rc_kmer else rc_kmer
                )

        return kmers_list

    def _process_and_write_kmers(
        self, genome_file_info, kmer_length, temp_dir
    ) -> pathlib.Path:
        """
        Legacy method - now redirects to static worker for consistency.
        """
        # Reconstruct task tuple for worker
        genome_file, strain_name, strain_idx = genome_file_info[:3]
        task_tuple = (
            genome_file,
            strain_name,
            strain_idx,
            kmer_length,  # kmer_length is passed directly
            self.args.skip_n_kmers,  # skip_n_kmers from pydantic model
            temp_dir,
        )
        return self._process_and_write_kmers_worker(task_tuple)


# Typer application
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command(
    name="build-db",
    help="StrainR Database Builder: Downloads genomes and creates a k-mer presence/absence database.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def main(
    taxid: Annotated[
        Optional[str], typer.Option(help="Species taxonomic ID from NCBI.")
    ] = None,
    assembly_accessions: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to a file listing assembly accessions (one per line).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    genus: Annotated[
        Optional[str], typer.Option(help="Genus name for NCBI download.")
    ] = None,
    custom: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to a folder containing custom genome FASTA files.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    kmerlen: Annotated[int, typer.Option(help="Length of k-mers.")] = 31,
    assembly_levels: Annotated[
        str,
        typer.Option(
            help="Assembly level(s) for NCBI download (complete, chromosome, scaffold, contig)."
        ),
    ] = "complete",
    source: Annotated[
        str, typer.Option(help="NCBI database source (refseq, genbank).")
    ] = "refseq",
    procs: Annotated[int, typer.Option(help="Number of processor cores.")] = 4,
    out: Annotated[
        str, typer.Option(help="Output name prefix for the database file.")
    ] = "strainr_kmer_database",
    unique_taxid: Annotated[
        bool,
        typer.Option(
            "--unique-taxid", help="Filter NCBI genomes for unique strain-level taxID."
        ),
    ] = False,
    skip_n_kmers: Annotated[
        bool, typer.Option("--skip-n-kmers", help="Exclude k-mers with 'N' bases.")
    ] = False,
):
    """
    Main CLI entry point for building the StrainR database.
    Uses Typer for argument parsing and Pydantic for validation.
    """
    logger.info("StrainR Database Building Script Started.")

    try:
        # Convert Typer args to Pydantic model
        # Typer ensures file/dir existence for paths, Pydantic model will re-validate logic
        build_args = BuildDBArgs(
            taxid=taxid,
            assembly_accessions=assembly_accessions,
            genus=genus,
            custom=custom,
            kmerlen=kmerlen,
            assembly_levels=assembly_levels,
            source=source,
            procs=procs,
            out=out,
            unique_taxid=unique_taxid,
            skip_n_kmers=skip_n_kmers,
        )

        builder = DatabaseBuilder(args=build_args)
        builder.create_database()

    except ValueError as ve:  # Catch Pydantic validation errors specifically
        logger.critical(
            f"Configuration error: {ve}", exc_info=False
        )  # No need for full stack trace
        # Attempt to provide a more user-friendly message from Typer/Pydantic if possible
        # For now, just printing the error is fine.
        print(f"Error: {ve}", file=sys.stderr)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.critical(
            f"A critical error occurred during database creation: {e}", exc_info=True
        )
        # Consider if specific error types should have different exit codes
        raise typer.Exit(code=1)

    logger.info("StrainR Database Building Script Finished Successfully.")


if __name__ == "__main__":
    app()
