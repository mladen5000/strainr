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

import logging
import multiprocessing as mp
import pathlib
import shutil
import shlex
import subprocess
import sys
import gzip
from collections import Counter, defaultdict
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
    in_memory_processing: bool = Field(
        False, description="Use in-memory k-mer processing instead of disk-based."
    )

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

        # --- Enhanced check for expected .part.txt files ---
        expected_part_files = [temp_dir / f"{i}.part.txt" for i in range(num_genomes)]
        problematic_files = []
        missing_files_list = []
        empty_files_list = []

        for f_path in expected_part_files:
            if not f_path.exists():
                missing_files_list.append(str(f_path))
                problematic_files.append(str(f_path)) # Also add to general problematic list
            elif f_path.stat().st_size == 0:
                empty_files_list.append(str(f_path))
                problematic_files.append(str(f_path)) # Also add to general problematic list

        if missing_files_list or empty_files_list:
            error_messages = []
            if missing_files_list:
                error_messages.append(f"Missing expected k-mer part files: {missing_files_list}")
            if empty_files_list:
                error_messages.append(f"Empty k-mer part files (suggesting worker error): {empty_files_list}")

            full_error_message = ". ".join(error_messages) + ". Aborting before sort/concat phase."
            logger.error(full_error_message)
            raise RuntimeError(full_error_message) # Changed from FileNotFoundError to RuntimeError for broader scope

        logger.info("All expected k-mer part files exist and are non-empty.")
        try:
            # --- 2. SORT PHASE ---
            # Concatenate and sort the temporary files on disk.
            all_parts_file = temp_dir / "all_kmer_parts.tsv"
            sorted_parts_file = temp_dir / "all_kmer_parts.sorted.tsv"

            logger.info(f"Concatenating temporary part files into {all_parts_file}...")
            part_files = list(temp_dir.glob("*.part.txt"))
            if len(part_files) > 100:  # For many files, use find + cat (robustly)
                # Using print0 and xargs -0 for robustness with filenames containing special characters or spaces.
                # The '--' ensures that even if a filename starts with '-', cat doesn't interpret it as an option.
                concat_command = f"find {temp_dir} -name '*.part.txt' -print0 | xargs -0 cat -- > {all_parts_file}"
            else:  # For fewer files, direct cat
                if not part_files: # Handle case where there are no files to avoid error with empty command
                    concat_command = f"touch {all_parts_file}" # Create an empty file if no parts exist
                else:
                    part_paths_str = " ".join(shlex.quote(str(f)) for f in part_files) # Use shlex.quote for safety
                    concat_command = f"cat {part_paths_str} > {all_parts_file}"

            logger.info(f"Executing concatenation command: {concat_command}")
            concat_process = subprocess.run(
                concat_command, shell=True, capture_output=True, text=True
            )
            if concat_process.returncode != 0:
                logger.error(
                    f"Concatenation command ('{concat_command}') failed with exit code {concat_process.returncode}."
                )
                logger.error(f"Stdout: {concat_process.stdout.strip()}")
                logger.error(f"Stderr: {concat_process.stderr.strip()}")
                raise RuntimeError(f"Concatenation of k-mer part files failed. Command: {concat_command}")
            logger.info(f"Concatenation command finished with return code {concat_process.returncode}.")
            logger.info("Concatenation completed successfully.")

            logger.info("Optimized sorting all k-mer parts on disk (parallel sort)...")
            # Use parallel sort with memory buffer. Compression is removed as inputs are plain text.
            sort_command = (
                f"LC_ALL=C sort -k1,1 --parallel={self.args.procs} "
                f"-S 2G {all_parts_file} -o {sorted_parts_file}"
            )
            logger.info(f"Executing sort command: {sort_command}")
            sort_process = subprocess.run(
                sort_command, shell=True, capture_output=True, text=True
            )
            if sort_process.returncode != 0:
                logger.error(
                    f"Sort command ('{sort_command}') failed with exit code {sort_process.returncode}."
                )
                logger.error(f"Stdout: {sort_process.stdout.strip()}")
                logger.error(f"Stderr: {sort_process.stderr.strip()}")
                raise RuntimeError(f"Disk-based sort command failed. Command: {sort_command}")
            logger.info(f"Sort command finished with return code {sort_process.returncode}.")
            logger.info("Sort phase completed.")

            # --- 3. REDUCE PHASE (OPTIMIZED) ---
            # Group sorted k-mers and write to Parquet file in batches.
            output_path = self.base_path / (self.output_db_name + ".db.parquet")
            logger.info(f"Starting Reduce phase: Aggregating sorted k-mers from {sorted_parts_file}")
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
                    processed_kmer_groups_count = 0
                    for kmer_hex, group in tqdm(
                        grouped_iterator, desc="Aggregating k-mers (Reduce)"
                    ):
                        processed_kmer_groups_count += 1
                        if processed_kmer_groups_count % 100000 == 0 and processed_kmer_groups_count > 0:
                            logger.info(f"Reduce phase: Aggregated {processed_kmer_groups_count} unique k-mer groups so far.")
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

    @staticmethod
    def _extract_kmers_from_file(
        genome_file: pathlib.Path, kmerlen: int, skip_n_kmers: bool, use_rust_if_available: bool
    ) -> Set[bytes]:
        """
        Extracts canonical k-mers from a single genome FASTA/FASTA.gz file.
        Chooses between Rust and Python implementation based on availability and arguments.
        """
        logger_func = logging.getLogger(__name__) # Use a local logger or pass one if static
        kmers_set: Set[bytes] = set()

        rust_available = _RUST_KMER_COUNTER_AVAILABLE and _extract_kmers_func is not None
        attempt_rust = use_rust_if_available and rust_available

        processed_with_rust_successfully = False
        if attempt_rust:
            try:
                # _extract_kmers_func is extract_kmer_rs(file_path, k, process_n_kmers)
                # process_n_kmers is true if we WANT to process N-containing kmers (i.e., NOT skip them)
                rust_kmers_map = _extract_kmers_func(
                    str(genome_file), kmerlen, not skip_n_kmers
                )
                kmers_set.update(rust_kmers_map.keys())
                processed_with_rust_successfully = True
                logger_func.debug(f"Extracted {len(kmers_set)} unique k-mers from {genome_file} using Rust.")
            except Exception as e:
                logger_func.warning(
                    f"Rust k-mer extraction failed for {genome_file}: {e}. "
                    f"Falling back to Python for this file."
                )

        if not processed_with_rust_successfully:
            if attempt_rust: # Log if this is a fallback
                 logger_func.info(f"Executing Python fallback for {genome_file} due to Rust failure or Rust not being selected.")
            else: # Log if Python was the primary choice
                 logger_func.info(f"Using Python k-mer extraction for {genome_file}.")

            try:
                with open_file_transparently(genome_file) as f_handle:
                    for record in SeqIO.parse(f_handle, "fasta"):
                        if record and record.seq:
                            seq_bytes = str(record.seq).encode("utf-8")
                            kmers_from_record = (
                                DatabaseBuilder._py_extract_canonical_kmers_static(
                                    seq_bytes,
                                    kmerlen,
                                    skip_n_kmers,
                                )
                            )
                            kmers_set.update(kmers_from_record)
                logger_func.debug(f"Extracted {len(kmers_set)} unique k-mers (after Python processing) from {genome_file}.")
            except Exception as e:
                logger_func.error(f"Python k-mer extraction failed critically for {genome_file}: {e}", exc_info=True)
                # Re-raise or handle as appropriate for the caller.
                # For now, let the caller decide. If this helper is used by the worker,
                # the worker might raise to stop its process. If used by in-memory, it might skip.
                raise  # Or return an empty set / specific error indicator
        return kmers_set

    def _build_kmer_database_in_memory(
        self, genome_files: List[pathlib.Path], strain_names: List[str]
    ) -> Optional[pathlib.Path]:
        """
        Builds the k-mer database entirely in memory using defaultdict and NumPy arrays.
        """
        if not genome_files:
            logger.warning("No genome files provided to build in-memory database.")
            return None

        if len(genome_files) != len(strain_names):
            raise ValueError(
                "Mismatch between number of genome files and strain names for in-memory build."
            )

        num_genomes = len(genome_files)
        logger.info(
            f"Starting in-memory k-mer database construction for {num_genomes} genomes."
        )

        # Initialize defaultdict: kmer (bytes) -> np.array (bool, size num_genomes)
        kmer_dict: Dict[bytes, np.ndarray] = defaultdict(
            lambda: np.zeros(num_genomes, dtype=bool)
        )

        # Determine if Rust should be attempted based on global availability
        # The helper function `_extract_kmers_from_file` will also check availability
        # but this top-level check can inform general logging.
        can_attempt_rust = _RUST_KMER_COUNTER_AVAILABLE and _extract_kmers_func is not None
        if can_attempt_rust:
            logger.info("Rust k-mer extractor is available and will be attempted for in-memory build.")
        else:
            logger.info("Rust k-mer extractor not available. Using Python for in-memory build.")

        for i, (genome_file, strain_name) in tqdm(
            enumerate(zip(genome_files, strain_names)),
            total=num_genomes,
            desc="Processing genomes (In-memory)",
        ):
            logger.debug(f"Processing {genome_file} for strain {strain_name} (index {i})")
            try:
                current_strain_kmers = DatabaseBuilder._extract_kmers_from_file(
                    genome_file,
                    self.args.kmerlen,
                    self.args.skip_n_kmers,
                    use_rust_if_available=True # Always try Rust if available globally
                )

                # Populate the main kmer_dict
                for kmer_bytes in current_strain_kmers:
                    kmer_dict[kmer_bytes][i] = True

                logger.info(f"Processed {i+1}/{num_genomes}: {strain_name} - {len(current_strain_kmers)} unique kmers found and added to main dictionary.")

            except Exception as e:
                # The _extract_kmers_from_file method logs specifics, here we log the consequence.
                logger.error(f"Critical error extracting k-mers from {genome_file} for strain {strain_name}. Skipping this file. Error: {e}", exc_info=True)
                # Optionally, mark this strain's columns as all False or handle differently if needed,
                # but for now, its k-mers simply won't be added.
                continue # Skip to the next genome file

        if not kmer_dict:
            logger.warning("No k-mers found in any of the provided (and successfully processed) genomes.")
            return None

        logger.info(f"Total unique k-mers found across all genomes: {len(kmer_dict)}")

        output_path = self.base_path / (self.output_db_name + ".db.parquet")
        logger.info(f"Converting k-mer data to Parquet format at {output_path}")

        # Prepare data for PyArrow Table
        # This approach is more memory-efficient than creating intermediate lists of all k-mers/vectors
        kmer_column_data = list(kmer_dict.keys()) # Still need all k-mers once for the column

        # Strain columns: each is a list of booleans
        strain_columns_pa_data = []
        for strain_idx in range(num_genomes):
            # For each strain, create its boolean column based on the presence vectors in kmer_dict
            strain_col = [kmer_dict[kmer][strain_idx] for kmer in kmer_column_data]
            strain_columns_pa_data.append(pa.array(strain_col, type=pa.bool_()))

        pa_arrays = [pa.array(kmer_column_data, type=pa.binary())] + strain_columns_pa_data
        pa_names = ["kmer"] + strain_names

        # Define schema with metadata
        schema_fields = [pa.field("kmer", pa.binary())] + [
            pa.field(name, pa.bool_()) for name in strain_names
        ]
        custom_metadata = {
            b"strainr_kmerlen": str(self.args.kmerlen).encode("utf-8"),
            b"strainr_skip_n_kmers": str(self.args.skip_n_kmers).encode("utf-8"),
        }
        schema = pa.schema(schema_fields, metadata=custom_metadata)

        try:
            # If schema is provided, it defines the names and types.
            # Do not pass pa_names separately if schema is passed.
            table = pa.Table.from_arrays(pa_arrays, schema=schema)
            pq.write_table(table, output_path)
            logger.info(f"In-memory k-mer database successfully saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write Parquet file for in-memory database: {e}", exc_info=True)
            return None # Indicate failure

        return output_path

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

        # This method now handles file creation and returns the path.
        # It no longer returns a DataFrame.
        if self.args.in_memory_processing:
            logger.info("Using in-memory k-mer database construction strategy.")
            db_path = self._build_kmer_database_in_memory(
                genome_files, strain_identifiers
            )
        else:
            logger.info("Using disk-based (parallel) k-mer database construction strategy.")
            # Check for external commands only if using the parallel disk-based method
            try:
                check_external_commands(["sort", "cat", "find", "xargs", "touch"])
                logger.info("External commands for disk-based processing found.")
            except FileNotFoundError as e:
                logger.error(
                    f"Missing required external command for disk-based processing: {e}. "
                    "Consider using --in-memory-processing if system resources allow, "
                    "or ensure the command is installed and in PATH."
                )
                raise # Re-raise to stop execution

            db_path = self._build_kmer_database_parallel(
                genome_files, strain_identifiers
            )

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
        # Ensure logger is available in static context
        worker_logger = logging.getLogger(f"{__name__}._process_and_write_kmers_worker")

        genome_file, strain_name, strain_idx, kmer_length, skip_n_kmers, temp_dir = task
        worker_logger.info(f"Worker starting for strain {strain_name} (index {strain_idx}), file {genome_file}")

        try:
            # Use the centralized k-mer extraction method.
            # For workers in the parallel disk-based method, always try to use Rust if available.
            extracted_kmers = DatabaseBuilder._extract_kmers_from_file(
                genome_file,
                kmer_length,
                skip_n_kmers,
                use_rust_if_available=True
            )
            worker_logger.info(f"Worker for strain {strain_name}: Successfully extracted {len(extracted_kmers)} k-mers.")

        except Exception as e:
            worker_logger.error(
                f"Worker for strain {strain_name} (file {genome_file}) failed during k-mer extraction: {e}",
                exc_info=True
            )
            # Re-raise the exception. This will be caught by the main process if it's observing worker results,
            # allowing for a fail-fast mechanism.
            raise

        # Writing k-mers to file
        temp_file_path = temp_dir / f"{strain_idx}.part.txt"
        num_kmers_to_write = len(extracted_kmers)

        if num_kmers_to_write == 0:
            worker_logger.info(f"Worker for strain_idx {strain_idx} ({strain_name}): No k-mers to write for {genome_file}.")
            with open(temp_file_path, "w", encoding='utf-8') as f_out:
                pass
        else:
            worker_logger.info(f"Worker for strain_idx {strain_idx} ({strain_name}): Preparing to write {num_kmers_to_write} k-mers to {temp_file_path}")
            batch_size = 10000
            batch = []
            total_written_count = 0
            with open(temp_file_path, "w", encoding='utf-8') as f_out:
                for kmer_bytes in extracted_kmers:
                    line = f"{kmer_bytes.hex()}\t{strain_idx}\n"
                    batch.append(line)
                    total_written_count += 1
                    if len(batch) >= batch_size:
                        f_out.writelines(batch)
                        batch = []

                    if total_written_count > 0 and total_written_count % (batch_size * 50) == 0:
                        worker_logger.debug(f"Worker for strain_idx {strain_idx} ({strain_name}): Written {total_written_count}/{num_kmers_to_write} k-mers so far to {temp_file_path}")

                if batch:
                    f_out.writelines(batch)

        worker_logger.info(f"Worker for strain_idx {strain_idx} ({strain_name}): Finished writing {total_written_count} k-mers to {temp_file_path}")
        worker_logger.info(f"Worker finished for task processing {genome_file.name} for strain {strain_name}")
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
    in_memory_processing: Annotated[
        bool,
        typer.Option(
            "--in-memory-processing",
            help="Use in-memory k-mer processing (faster but requires more RAM).",
        ),
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
            in_memory_processing=in_memory_processing,
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
