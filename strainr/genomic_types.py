"""
Type definitions for the Strainr package.

This module centralizes common type aliases used throughout the Strainr application
to ensure consistency and improve code readability.
"""

from typing import Dict, List, Tuple, Union  # Removed Optional, Generator, Any

import numpy as np
import numpy.typing as npt

# Type aliases for clarity
KmerString = str  # Represents a k-mer as a Python string.
StrainIndex = int  # Represents the integer index of a strain.
ReadId = str  # Represents a unique identifier for a sequence read.
CountVector = npt.NDArray[
    np.uint8
]  # A NumPy array storing k-mer counts (uint8) for each strain.
KmerCountDict = Dict[bytes, CountVector]  # Maps a k-mer (bytes) to its CountVector.
ReadHitResults = List[
    Tuple[ReadId, CountVector]
]  # A list of tuples, each pairing a ReadId with its CountVector of k-mer hits.
StrainAbundanceDict = Dict[
    Union[StrainIndex, str], float
]  # Maps a strain (by index or name) to its relative abundance (float).
FinalAssignmentsType = Dict[
    ReadId, Union[StrainIndex, str]
]  # Maps a ReadId to its final assignment, which can be a StrainIndex or an unassigned marker string.
