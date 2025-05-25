# =============================================================================
# strainr/output.py - Refactored output and reporting
# =============================================================================
"""
Output formatting and abundance calculation functionality.

CHANGES:
- Improved abundance calculation logic
- Better normalization and thresholding
- Enhanced output formatting
- More comprehensive statistics
"""

import pathlib
from collections import Counter, defaultdict
from typing import Dict, Union, List, Any
import pandas as pd


class AbundanceCalculator:
    """
    Calculator for strain abundance analysis and reporting.
    
    This class handles conversion of read assignments to relative abundances,
    normalization, thresholding, and output formatting.
    """
    
    def __init__(
        self, 
        strain_names: List[str],
        abundance_threshold: float = 0.001
    ):
        """
        Initialize the abundance calculator.
        
        Args:
            strain_names: List of all strain identifiers
            abundance_threshold: Minimum relative abundance for reporting
        """
        self.strain_names = strain_names
        self.abundance_threshold = abundance_threshold

    def convert_assignments_to_strain_names(