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
"""

import argparse
import logging
import multiprocessing as mp
import pathlib
import sys
from collections import Counter
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

import ncbi_genome_download as ngd
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# from utils import open_file_transparently
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


class DatabaseBuilder:
    """
    Manages the k-mer database creation workflow.

    This class encapsulates all steps from genome download to k-mer matrix
    generation and saving.
    """

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
        if not metadata_table_path.exists():
            logger.warning(
                f"Metadata file not found at {metadata_table_path}, cannot filter by unique taxID. Using all genomes."
            )
            return list(genome_dir.glob("*fna.gz"))

        accessions_df = pd.read_csv(metadata_table_path, sep="\t")
        if "assembly_accession" in accessions_df.columns:
            accessions_df = accessions_df.set_index("assembly_accession")

        has_taxid: DataFrame | Series | Unknown = accessions_df["taxid"].notna()
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
            if potential_path.exists():
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

        accession_match = base_name.split("_")[0]
        if len(accession_match) < 10:
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
        if metadata_file_path and metadata_file_path.exists():
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

    def _process_single_fasta_for_kmers(
        self,
        genome_file_info: Tuple[pathlib.Path, str, int],
        kmer_length: int,
        num_total_strains: int,
    ) -> Tuple[str, int, Set[bytes]]:
        """
        Extracts unique k-mers (as bytes) from one FASTA file.
        """
        genome_file, strain_name, strain_idx = genome_file_info
        strain_kmers: Set[bytes] = set()
        with open_file_transparently(genome_file) as f_handle:
            for record in SeqIO.parse(f_handle, "fasta"):
                # Note: staticmethod, so no extra `self` is passed
                strain_kmers |= DatabaseBuilder.fast_kmers_numpy(
                    str(record.seq), kmer_length
                )
        return strain_name, strain_idx, strain_kmers

    def _build_kmer_database_parallel(
        self, genome_files: List[pathlib.Path], strain_names: List[str]
    ) -> pd.DataFrame:
        """
        (Unchanged logic, except now fast_kmers_numpy yields bytes instead of void)
        """
        if not genome_files:
            logger.warning("No genome files provided to build database.")
            return pd.DataFrame()
        if len(genome_files) != len(strain_names):
            raise ValueError(
                "Mismatch between number of genome files and strain names."
            )

        logger.info(
            f"Starting k-mer extraction for {len(genome_files)} genomes using {self.args.procs} processes."
        )

        tasks = [
            (genome_files[i], strain_names[i], i) for i in range(len(genome_files))
        ]
        worker = partial(
            self._process_single_fasta_for_kmers,
            kmer_length=self.args.kmerlen,
            num_total_strains=len(strain_names),
        )

        strain_kmer_sets = []
        with mp.Pool(processes=self.args.procs) as pool:
            for strain_name, strain_idx, strain_kmers in tqdm(
                pool.imap_unordered(worker, tasks),
                total=len(tasks),
                desc="Extracting k-mers",
            ):
                strain_kmer_sets.append((strain_idx, strain_kmers))

        # Sort so that the column order matches strain_names
        strain_kmer_sets.sort()
        only_kmer_sets = [x[1] for x in strain_kmer_sets]

        logger.info("Building global k-mer set.")
        all_kmers = set().union(*only_kmer_sets)  # union of all sets
        all_kmers = sorted(all_kmers)
        kmer_index = {kmer: i for i, kmer in enumerate(all_kmers)}

        matrix = np.zeros((len(all_kmers), len(strain_names)), dtype=bool)
        for col, kset in enumerate(only_kmer_sets):
            idxs = [kmer_index[k] for k in kset]
            matrix[idxs, col] = True

        # all_kmers is already a list of Python bytes, so just decode if you want strings
        try:
            kmer_labels = [k.decode("ascii") for k in all_kmers]
        except:
            kmer_labels = all_kmers  # leave as bytes if non-ASCII appear

        df = pd.DataFrame(matrix, index=kmer_labels, columns=strain_names)
        return df

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
        "-k", "--kmerlen", type=int, default=31, help="Length of k-mers to extract."
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
    return parser


if __name__ == "__main__":
    logger.info("StrainR Database Building Script Started.")

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
