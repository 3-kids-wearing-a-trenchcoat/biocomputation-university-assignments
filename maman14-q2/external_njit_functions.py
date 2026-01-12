"""
This module holds the functions implemented using njit and called by any other module
"""

from __future__ import annotations
import numpy as np
from numba import njit
from typing import Tuple, List
from numpy.typing import NDArray

rng = np.random.default_rng()
FLOAT_DTYPE = np.float16

@njit
def choose_binding_by_strength(candidates:List[Tuple[int, int, int, float]]) -> Tuple[int, int, int, float]:
    """Choose one of the bindings in the input list randomly, weighted by strength."""
    # Normalize strength to be used as probability
    probs = np.asarray([entry[3] for entry in candidates], dtype=FLOAT_DTYPE)
    # TODO: If I wanted to apply some probability function like strength**2, this would be the place
    total = probs.sum()
    probs = probs / total
    chosen = int(rng.choice(len(candidates), p=probs))
    return candidates[chosen]
