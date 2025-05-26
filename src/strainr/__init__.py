"""
Strainr: A Python Package for K-mer Based Strain Analysis.

This package provides tools for building k-mer databases, classifying sequences
against these databases, analyzing classification results, calculating strain
abundances, and performing sequence manipulation related to k-mers.
"""

__version__ = "0.1.0"

# Core classes and functions for easier access
from .kmer_database import StrainKmerDb # Updated to consolidated class name
from .analyze import ClassificationAnalyzer
from .output import AbundanceCalculator
from .sequence import GenomicSequence, extract_kmers_from_sequence
from .utils import open_file_transparently, get_canonical_kmer
from .parameter_config import process_arguments
from .running import main

__all__ = [
    "StrainKmerDb", # Updated to consolidated class name
    "ClassificationAnalyzer",
    "AbundanceCalculator",
    "GenomicSequence",
    "extract_kmers_from_sequence",
    "open_file_transparently",
    "get_canonical_kmer",
    "process_arguments",
    "main",
    "__version__",
]
