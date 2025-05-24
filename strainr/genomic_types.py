"""
Type definitions for the Strainr package.
"""

from typing import Dict, List, Tuple, Union, Optional, Generator, Any
import numpy as np
import numpy.typing as npt
from pathlib import Path

# Type aliases for clarity
KmerString = str
StrainIndex = int
ReadId = str
CountVector = npt.NDArray[np.uint8]
KmerCountDict = Dict[bytes, CountVector]
ReadHitResults = List[Tuple[ReadId, CountVector]]
StrainAbundanceDict = Dict[Union[StrainIndex, str], float]
