#!/usr/bin/env python
"""
StrainR Database Builder: Downloads genomes and creates a k-mer presence/absence database.

This script automates the process of:
1. Downloading bacterial genomes from NCBI (RefSeq or GenBank) based on
   taxonomic ID, a list of assembly accessions, or genus name.
   It uses `ncbi-genome-download` for this purpose.
2. Filtering downloaded genomes, for example, to include only those with
   unique strain-level taxonomic IDs.
3. Extracting k-mers from the genomic FASTA files.
4. Constructing a presence/absence matrix where rows are unique k-mers,
   columns are strains (genomes), and values indicate if a k-mer is present
   in a strain.
5. Saving this matrix as a pickled Pandas DataFrame, which serves as the
   k-mer database for StrainR classification tools.
"""

import argparse
import gzip
import logging
import pathlib
import pickle
import sys
from collections import defaultdict
from functools import partial
from mimetypes import guess_type
from typing import Dict, List, Tuple, Union, Set, Any, Optional, TextIO

import ncbi_genome_download as ngd
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm # For progress bars

# Assuming strainr.utils is in PYTHONPATH or installed
from strainr.utils import open_file_transparently


# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Configure logging if not already configured by another part of a larger application
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, # Default to INFO, can be made configurable
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("strainr_db_creation.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


class DatabaseBuilder:
    """
    Manages the k-mer database creation workflow.

    This class encapsulates all steps from genome download to k-mer matrix
    generation and saving.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initializes the DatabaseBuilder.

        Args:
            args: An `argparse.Namespace` object containing parsed command-line arguments.
                  Expected attributes include `taxid`, `assembly_accessions`, `kmerlen`,
                  `assembly_levels`, `source`, `procs`, `genus`, `out` (output name),
                  `custom` (custom genome folder), and `unique_taxid`.
        """
        logger.info("Initializing DatabaseBuilder.")
        self.args: argparse.Namespace = args
        self.base_path: pathlib.Path = pathlib.Path().cwd() # Base path for outputs

        # Ensure output directory for the database file exists
        self.output_db_name: str = self.args.out
        self.output_db_dir: pathlib.Path = self.base_path / (self.output_db_name + "_files") # Dir for genomes etc.
        self.output_db_dir.mkdir(parents=True, exist_ok=True)


    def _parse_assembly_level(self) -> str:
        """
        Determines the assembly level string for ncbi-genome-download.

        Based on the `assembly_levels` argument, it constructs the appropriate
        comma-separated string for desired assembly levels (e.g., "complete",
        "complete,chromosome").

        Returns:
            A string representing the assembly levels for `ncbi-genome-download`.

        Raises:
            ValueError: If an invalid assembly level choice is provided.
        """
        logger.debug(f"Parsing assembly level: {self.args.assembly_levels}")
        level_choice = self.args.assembly_levels
        if self.args.assembly_accessions: # If specific accessions are given, download all levels for them
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
            # This should ideally be caught by argparse choices, but good for safety
            raise ValueError(f"Invalid assembly level selected: {level_choice}")

    def _download_genomes_from_ncbi(self) -> Tuple[pathlib.Path, pathlib.Path]:
        """
        Downloads genomes from NCBI using `ncbi-genome-download`.

        Constructs arguments for `ngd.download` based on user input (taxid,
        accessions, genus, etc.) and executes the download.

        Returns:
            A tuple containing:
                - `genome_output_dir`: Path to the directory where genomes were downloaded.
                - `metadata_table_path`: Path to the metadata summary file.

        Raises:
            ValueError: If required arguments for downloading (taxid, accessions, or genus)
                        are not appropriately provided or are conflicting.
            ConnectionError: If `ncbi-genome-download` fails to download genomes.
        """
        logger.info("Preparing to download genomes from NCBI.")
        assembly_level_str = self._parse_assembly_level()

        ncbi_kwargs: Dict[str, Any] = {
            "flat_output": True, # Keep downloaded files in a flat directory structure
            "groups": "bacteria", # Assuming bacteria, could be made configurable
            "file_formats": "fasta",
            "section": self.args.source, # 'refseq' or 'genbank'
            "parallel": self.args.procs,
            "assembly_levels": assembly_level_str,
        }

        # Determine download target (taxid, accessions, or genus)
        genome_target_dir_suffix: str = ""
        if self.args.taxid:
            if self.args.assembly_accessions: # Ensure mutual exclusivity if not handled by argparse group
                raise ValueError("Cannot specify both --taxid and --assembly_accessions.")
            ncbi_kwargs["species_taxids"] = self.args.taxid
            genome_target_dir_suffix = f"s{self.args.taxid}"
        elif self.args.assembly_accessions:
            # Assuming assembly_accessions is a path to a file listing accessions
            accessions_file = pathlib.Path(self.args.assembly_accessions)
            if not accessions_file.is_file():
                raise FileNotFoundError(f"Assembly accessions file not found: {accessions_file}")
            ncbi_kwargs["assembly_accessions"] = str(accessions_file) # ngd expects string path
            # Create a suffix from the file name, e.g., "file_accs"
            genome_target_dir_suffix = f"acc_{accessions_file.stem}"
        elif self.args.genus:
            ncbi_kwargs["genera"] = self.args.genus
            genome_target_dir_suffix = f"g{self.args.genus.replace(' ', '_')}"
        else:
            raise ValueError("Must specify one of --taxid, --assembly_accessions, or --genus for NCBI download.")

        # Define output paths
        genome_output_dir = self.output_db_dir / f"ncbi_genomes_{genome_target_dir_suffix}"
        metadata_table_path = genome_output_dir / f"metadata_summary_{genome_target_dir_suffix}.tsv"
        
        ncbi_kwargs.update({
            "output_folder": genome_output_dir, # Corrected from 'output' to 'output_folder' for ngd
            "metadata_table": metadata_table_path,
        })
        
        logger.info(f"Downloading genomes to: {genome_output_dir} with metadata to {metadata_table_path}")
        logger.debug(f"ncbi-genome-download arguments: {ncbi_kwargs}")

        exit_code: int = ngd.download(**ncbi_kwargs)
        if exit_code != 0:
            logger.error(f"ncbi-genome-download failed with exit code {exit_code}.")
            raise ConnectionError("ncbi-genome-download did not successfully download the genomes.")
        
        logger.info("Genome download completed successfully.")
        return genome_output_dir, metadata_table_path

    def _filter_genomes_by_unique_taxid(
            self, genome_dir: pathlib.Path, metadata_table_path: pathlib.Path
    ) -> List[pathlib.Path]:
        """
        Filters downloaded genomes to include only those with unique strain-level taxIDs.

        Args:
            genome_dir: The directory containing the downloaded genome files.
            metadata_table_path: Path to the NCBI metadata summary TSV file.

        Returns:
            A list of `pathlib.Path` objects for genome files that meet the
            unique strain taxID criteria.
        """
        logger.info(f"Filtering genomes for unique strain taxIDs using metadata: {metadata_table_path}")
        if not metadata_table_path.exists():
            logger.warning(f"Metadata file not found at {metadata_table_path}, cannot filter by unique taxID. Using all genomes.")
            return list(genome_dir.glob("*fna.gz")) # Fallback: use all downloaded .fna.gz files

        accessions_df = pd.read_csv(metadata_table_path, sep="\t")
        # Ensure 'assembly_accession' is the index for consistent access
        if 'assembly_accession' in accessions_df.columns:
            accessions_df = accessions_df.set_index("assembly_accession")
        
        # Filter conditions
        has_taxid = accessions_df["taxid"].notna()
        is_unique_strain_taxid = accessions_df["taxid"] != accessions_df["species_taxid"]
        
        filtered_accessions_df = accessions_df[has_taxid & is_unique_strain_taxid]
        
        # Update metadata file with filtered entries (optional, good for record keeping)
        filtered_metadata_path = metadata_table_path.parent / f"filtered_{metadata_table_path.name}"
        filtered_accessions_df.to_csv(filtered_metadata_path, sep="\t")
        logger.info(f"Filtered metadata saved to: {filtered_metadata_path}")

        # Construct paths to the filtered genome files
        # 'local_filename' column in ngd metadata usually points to the file within its output structure
        if "local_filename" not in filtered_accessions_df.columns:
            logger.error("'local_filename' column not found in metadata. Cannot determine file paths for unique taxID genomes.")
            return [] # Or raise error

        filtered_genome_files: List[pathlib.Path] = []
        for rel_path_str in filtered_accessions_df["local_filename"]:
            # local_filename might be relative to the root of the section (e.g. refseq/bacteria/...)
            # or already include the genome_dir path. Check common ngd output structure.
            # ngd's `flat_output=True` means files are directly in genome_output_dir
            potential_path = genome_dir / pathlib.Path(rel_path_str).name # Use only filename part
            if potential_path.exists():
                filtered_genome_files.append(potential_path)
            else:
                logger.warning(f"Genome file listed in metadata not found: {potential_path} (original rel_path: {rel_path_str})")

        logger.info(f"Found {len(filtered_genome_files)} genomes after filtering for unique strain taxIDs.")
        return filtered_genome_files

    def _get_genome_file_list(self) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        """
        Determines the list of genome FASTA files to process.

        Handles three cases:
        1. Custom folder of genomes provided via `--custom`.
        2. Genomes downloaded from NCBI, potentially filtered by unique taxID.
        3. No genomes found or specified.

        Returns:
            A tuple containing:
                - A list of `pathlib.Path` objects for the genome FASTA files.
                - An optional `pathlib.Path` to the metadata (accession summary) file if downloaded.
        """
        if self.args.custom:
            custom_path = pathlib.Path(self.args.custom)
            if not custom_path.is_dir():
                raise NotADirectoryError(f"Custom genome directory not found: {custom_path}")
            logger.info(f"Using custom genome files from: {custom_path}")
            # Assuming .fna or .fna.gz, common extensions for NCBI FASTA files
            genome_files = list(custom_path.glob("*.fna")) + list(custom_path.glob("*.fna.gz"))
            return genome_files, None # No NCBI metadata file for custom genomes
        else:
            genome_download_dir, metadata_file = self._download_genomes_from_ncbi()
            all_downloaded_files = list(genome_download_dir.glob("*fna.gz")) # ngd typically downloads as .fna.gz

            if self.args.unique_taxid:
                return self._filter_genomes_by_unique_taxid(genome_download_dir, metadata_file), metadata_file
            else:
                return all_downloaded_files, metadata_file

    def _extract_strain_name_from_metadata(
            self, genome_file_path: pathlib.Path, metadata_df: Optional[pd.DataFrame]
    ) -> str:
        """
        Extracts a descriptive strain name using the metadata DataFrame.
        Falls back to filename if metadata is unavailable or accession is not found.

        Args:
            genome_file_path: Path to the genome FASTA file.
            metadata_df: Optional Pandas DataFrame containing genome metadata
                         (e.g., from NCBI download summary), indexed by assembly accession.

        Returns:
            A descriptive name for the strain/genome.
        """
        # Default name from file
        base_name = genome_file_path.name.replace(".fna.gz", "").replace(".fna", "")
        
        if metadata_df is None or metadata_df.empty:
            return base_name

        # NCBI FASTA files often have accession like GCF_xxxxxxxxxxxxxxx.1_assembly_name in filename
        # Attempt to extract accession from filename (first part before first '_')
        # This might need adjustment based on actual filename patterns from ngd
        accession_match = base_name.split('_')[0] 
        if len(accession_match) < 10 : # Heuristic for typical accession like GCF_... or GCA_...
             accession_match = base_name # Fallback if no underscore or first part is short

        try:
            if accession_match in metadata_df.index:
                record = metadata_df.loc[accession_match]
                org_name = record.get("organism_name", base_name)
                strain_info = record.get("infraspecific_name", "")
                if pd.isna(strain_info) or strain_info == "nan": strain_info = "" # Handle actual NaN or "nan" string
                strain_info = str(strain_info).replace("strain=", "").strip()
                
                # Combine parts for a descriptive name
                full_name = org_name
                if strain_info:
                    full_name += f" strain={strain_info}"
                full_name += f" ({accession_match})" # Add accession for uniqueness
                return full_name
            else: # Accession not found in metadata
                logger.warning(f"Accession '{accession_match}' (derived from {base_name}) not found in metadata. Using filename as identifier.")
                return base_name
        except KeyError: # If .loc fails
            logger.warning(f"Metadata lookup failed for {accession_match}. Using filename {base_name} as identifier.")
            return base_name


    def _get_genome_names_for_files(
            self, genome_files: List[pathlib.Path], metadata_file_path: Optional[pathlib.Path]
    ) -> List[str]:
        """
        Generates descriptive names for each genome file.

        If a metadata file from NCBI download is available, it's used to extract
        organism and strain information. Otherwise, filenames are used.

        Args:
            genome_files: A list of paths to genome FASTA files.
            metadata_file_path: Optional path to the metadata summary file from NCBI.

        Returns:
            A list of string names, corresponding to each genome file.
        """
        logger.info("Generating names for genome files.")
        if not genome_files:
            return []

        metadata_df: Optional[pd.DataFrame] = None
        if metadata_file_path and metadata_file_path.exists():
            try:
                metadata_df = pd.read_csv(metadata_file_path, sep="\t")
                if 'assembly_accession' in metadata_df.columns:
                     metadata_df = metadata_df.set_index("assembly_accession")
                else:
                    logger.warning(f"'assembly_accession' column not found in {metadata_file_path}. Cannot use it for name generation effectively.")
                    metadata_df = None # Treat as if no metadata if key column missing
            except Exception as e:
                logger.error(f"Error reading metadata file {metadata_file_path}: {e}. Proceeding with filenames.")
                metadata_df = None
        
        genome_names: List[str] = []
        for gf_path in genome_files:
            genome_names.append(self._extract_strain_name_from_metadata(gf_path, metadata_df))
        
        if len(genome_files) != len(genome_names): # Should not happen if logic is correct
            raise RuntimeError("Mismatch between number of genome files and generated names.")
        
        # Check for duplicate names and append numbers if necessary
        if len(set(genome_names)) < len(genome_names):
            logger.warning("Duplicate strain names detected. Appending suffixes to ensure uniqueness.")
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

    def _process_single_fasta_for_kmers(
        self,
        genome_file_info: Tuple[pathlib.Path, str, int], # (file_path, strain_name, strain_idx)
        kmer_length: int,
        num_total_strains: int
    ) -> Tuple[str, int, Dict[bytes, bool]]: # (strain_name, strain_idx, {kmer: presence})
        """
        Extracts k-mers from a single FASTA file for one strain.

        Args:
            genome_file_info: Tuple containing the path to the FASTA file,
                              the assigned strain name, and the strain's column index.
            kmer_length: The length of k-mers to extract.
            num_total_strains: Total number of strains (for context, though not directly used here).

        Returns:
            A tuple: (strain_name, strain_idx, unique_kmers_for_strain_dict).
            The dict maps k-mer bytes to True.
        """
        genome_file, strain_name, strain_idx = genome_file_info
        logger.debug(f"Processing {genome_file} for strain '{strain_name}' (index {strain_idx}).")
        
        # Using dict for kmer presence for this strain initially, then will populate matrix
        # Using set is more memory efficient if just checking presence for this one strain
        strain_kmers: Set[bytes] = set()

        with open_file_transparently(genome_file) as f_handle:
            for record in SeqIO.parse(f_handle, "fasta"):
                sequence_bytes = str(record.seq).upper().encode("utf-8") # Ensure bytes and upper case
                if len(sequence_bytes) < kmer_length:
                    continue
                
                with memoryview(sequence_bytes) as seq_view:
                    for i in range(len(sequence_bytes) - kmer_length + 1):
                        kmer = seq_view[i : i + kmer_length].tobytes()
                        strain_kmers.add(kmer)
        
        # Convert set to dict {kmer: True} for easier merging or consistency if needed later
        return strain_name, strain_idx, {kmer: True for kmer in strain_kmers}


    def _build_kmer_database_parallel(
        self, genome_files: List[pathlib.Path], strain_names: List[str]
    ) -> pd.DataFrame:
        """
        Builds the k-mer presence/absence matrix from genome files in parallel.

        Args:
            genome_files: List of paths to genome FASTA files.
            strain_names: List of corresponding strain names for column headers.

        Returns:
            A Pandas DataFrame where index is k-mer (bytes) and columns are
            strain names, values are boolean (presence/absence).
        """
        if not genome_files:
            logger.warning("No genome files provided to build database.")
            return pd.DataFrame()
        if len(genome_files) != len(strain_names):
            raise ValueError("Mismatch between number of genome files and strain names.")

        logger.info(f"Starting k-mer extraction for {len(genome_files)} genomes using {self.args.procs} processes.")

        # Prepare arguments for multiprocessing
        # Each task needs file path, strain name, its index, kmer_length, and total number of strains
        tasks = [
            (genome_files[i], strain_names[i], i) for i in range(len(genome_files))
        ]
        
        # Use functools.partial to pass fixed arguments to the worker function
        worker_function = partial(
            self._process_single_fasta_for_kmers,
            kmer_length=self.args.kmerlen,
            num_total_strains=len(strain_names)
        )
        
        # This dictionary will store all unique k-mers found across all genomes
        # and for each k-mer, a boolean numpy array indicating presence/absence in strains
        # {kmer_bytes: np.array([False, True, False, ...])}
        master_kmer_dict: Dict[bytes, np.ndarray] = defaultdict(
            lambda: np.zeros(len(strain_names), dtype=bool)
        )

        with mp.Pool(processes=self.args.procs) as pool:
            # Process files in parallel
            # Wrap with tqdm for a progress bar
            results = list(tqdm(pool.imap_unordered(worker_function, tasks), total=len(tasks), desc="Extracting k-mers"))

        logger.info("Aggregating k-mers into database matrix.")
        for strain_name, strain_idx, unique_kmers_for_strain_dict in tqdm(results, desc="Aggregating results"):
            for kmer_bytes in unique_kmers_for_strain_dict: # Iterate over keys (kmers)
                master_kmer_dict[kmer_bytes][strain_idx] = True
        
        if not master_kmer_dict:
            logger.warning("No k-mers were extracted from any genome. Resulting database will be empty.")
            return pd.DataFrame(columns=strain_names)

        logger.info(f"Total unique k-mers found: {len(master_kmer_dict)}")
        
        # Convert the master dictionary to a DataFrame
        # Sort k-mers (index) for consistent database output, though this can be slow for many k-mers
        # sorted_kmer_keys = sorted(master_kmer_dict.keys()) 
        # Using dict directly is faster if order doesn't strictly matter or can be handled later
        kmer_matrix_df = pd.DataFrame.from_dict(
            master_kmer_dict,
            orient="index", # k-mers as rows
            columns=strain_names, # strains as columns
            dtype=bool # Ensure boolean type
        )
        # kmer_matrix_df = kmer_matrix_df.reindex(sorted_kmer_keys) # Apply sort if using sorted_kmer_keys

        return kmer_matrix_df

    def _save_database_to_pickle(self, database_df: pd.DataFrame) -> None:
        """Saves the k-mer database DataFrame to a pickle file."""
        output_path = self.base_path / (self.output_db_name + ".db.pkl") # Changed extension for clarity
        logger.info(f"Saving k-mer database to: {output_path}")
        try:
            database_df.to_pickle(output_path)
            logger.info("Database saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save database to {output_path}: {e}")
            raise

    def create_database(self) -> None:
        """
        Main public method to orchestrate the entire database creation process.
        """
        logger.info("Starting k-mer database creation workflow.")
        
        # 1. Get list of genome files and optional metadata
        genome_files, metadata_file_path = self._get_genome_file_list()
        if not genome_files:
            logger.error("No genome files found or specified. Cannot build database.")
            return

        logger.info(f"Found {len(genome_files)} genome files to process.")

        # 2. Get strain/sequence names for these files
        strain_identifiers = self._get_genome_names_for_files(genome_files, metadata_file_path)
        if not strain_identifiers: # Should not happen if genome_files is not empty
            logger.error("Could not determine strain identifiers for genome files.")
            return

        # 3. Build the k-mer database (presence/absence matrix)
        kmer_database_df = self._build_kmer_database_parallel(genome_files, strain_identifiers)

        if kmer_database_df.empty:
            logger.warning("K-mer database construction resulted in an empty DataFrame.")
        else:
            logger.info(
                f"Constructed k-mer database DataFrame with shape: {kmer_database_df.shape} "
                f"(kmers x strains)."
            )
        
        # 4. Save the database
        self._save_database_to_pickle(kmer_database_df)
        
        logger.info("K-mer database creation workflow finished.")


def get_cli_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the ArgumentParser for the script.

    Returns:
        An `argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description="StrainR Database Builder: Downloads genomes and creates a k-mer presence/absence database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Group for specifying input genomes (mutually exclusive conceptually, but argparse needs care)
    genome_source_group = parser.add_mutually_exclusive_group(required=True)
    genome_source_group.add_argument(
        "-t", "--taxid", type=str,
        help="Species taxonomic ID from NCBI from which all strains will be downloaded."
    )
    genome_source_group.add_argument(
        "-f", "--assembly_accessions", type=str,
        help="Path to a file listing assembly accessions (one per line) to download from NCBI."
    )
    genome_source_group.add_argument(
        "-g", "--genus", type=str,
        help="Genus name for which to download genomes from NCBI."
    )
    genome_source_group.add_argument(
        "--custom", type=str,
        help="Path to a folder containing custom genome FASTA files (.fna, .fna.gz) for database creation."
    )

    parser.add_argument(
        "-k", "--kmerlen", type=int, default=31,
        help="Length of k-mers to extract."
    )
    parser.add_argument(
        "-l", "--assembly_levels",
        choices=["complete", "chromosome", "scaffold", "contig"], type=str, default="complete",
        help="Assembly level(s) of genomes to download from NCBI (e.g., 'contig' includes 'complete', 'chromosome', 'scaffold')."
    )
    parser.add_argument(
        "-s", "--source", choices=["refseq", "genbank"], type=str, default="refseq",
        help="NCBI database source for downloads (refseq or genbank)."
    )
    parser.add_argument(
        "-p", "--procs", type=int, default=4, # Increased default
        help="Number of processor cores to use for parallel tasks."
    )
    parser.add_argument(
        "-o", "--out", type=str, default="strainr_kmer_database", # More descriptive default
        help="Output name prefix for the database file (e.g., 'my_db' -> 'my_db.db.pkl')."
    )
    parser.add_argument(
        "--unique-taxid", action="store_true",
        help="Flag to only include genomes from NCBI that have a unique strain-level taxonomic ID."
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
        logger.critical(f"A critical error occurred during database creation: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info("StrainR Database Building Script Finished Successfully.")
