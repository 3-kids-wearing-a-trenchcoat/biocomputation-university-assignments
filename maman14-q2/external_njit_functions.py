"""
This module holds the functions implemented using njit and called by any other module
"""

from __future__ import annotations
import numpy as np
from numba import njit
from typing import Tuple, List
from numpy.typing import NDArray

rng = np.random.default_rng()
FLOAT_DTYPE = np.float32


