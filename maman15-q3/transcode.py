"""This module contains the functions that convert between any of the following
1. binary representation
2. language representation (two-letter words in a 6-letter language)
3. DNA representation"""

from __future__ import annotations
from typing import List, Tuple, TypeAlias
import numpy as np
from numpy.typing import NDArray

# representations
five_bit: TypeAlias = tuple[int, int, int, int, int]
WORD_BIT_PAIRS: List[Tuple[str, five_bit]] = [("UU", (0,0,0,0,0)), ("UV", (0,0,0,0,1)), ("UW", (0,0,0,1,0)),
                                              ("UX", (0,0,0,1,1)), ("UY", (0,0,1,0,0)), ("UZ", (0,0,1,0,1)),
                                              ("VU", (0,0,1,1,0)), ("VV", (0,0,1,1,1)), ("VW", (0,1,0,0,0)),
                                              ("VX", (0,1,0,0,1)), ("VY", (0,1,0,1,0)), ("VZ", (0,1,0,1,1)),
                                              ("WU", (0,1,1,0,0)), ("WV", (0,1,1,0,1)), ("WW", (0,1,1,1,0)),
                                              ("WX", (0,1,1,1,1)), ("WY", (1,0,0,0,0)), ("WZ", (1,0,0,0,1)),
                                              ("XU", (1,0,0,1,0)), ("XV", (1,0,0,1,1)), ("XW", (1,0,1,0,0)),
                                              ("XX", (1,0,1,0,1)), ("XY", (1,0,1,1,0)), ("XZ", (1,0,1,1,1)),
                                              ("YU", (1,1,0,0,0)), ("YV", (1,1,0,0,1)), ("YW", (1,1,0,1,0)),
                                              ("YX", (1,1,0,1,1)), ("YY", (1,1,1,0,0)), ("YZ", (1,1,1,0,1)),
                                              ("ZU", (1,1,1,1,0)), ("ZV", (1,1,1,1,1))]

# TODO: Define word_base_pairs ("base" as in nucleotides)

# dictionaries - matching between representations
WORD_TO_BITS = {entry[0]: entry[1] for entry in WORD_BIT_PAIRS}
BITS_TO_WORD = {entry[1]: entry[0] for entry in WORD_BIT_PAIRS}