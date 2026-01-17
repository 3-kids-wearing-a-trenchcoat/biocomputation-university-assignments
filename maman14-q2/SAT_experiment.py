"""This module describes the run of an experiment for a given 3SAT formula"""

from __future__ import annotations
import binding, strand, anneal, unravel, magnetic, SATinit
from TwoBitArray import TwoBitArray
from typing import List, Tuple
from tqdm import tqdm, trange

# types
type Literal = Tuple[int, bool]                 # (variable number, literal is True)
type Clause = Tuple[Literal, Literal, Literal]  # (Clause[0] OR Clause[1] OR Clause[2])
type Formula = List[Clause]

# global variables


# functions

