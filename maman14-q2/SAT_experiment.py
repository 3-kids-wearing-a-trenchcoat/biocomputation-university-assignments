"""This module describes the run of an experiment for a given 3SAT formula"""

from __future__ import annotations
import binding, strand, anneal, unravel, magnetic, SATinit, SAT_routines
from SATinit import variable_rep_false, variable_rep_true, connector_rep, complement_rep
from PCR import PCR
from TwoBitArray import TwoBitArray
from typing import List, Tuple
from tqdm import tqdm, trange

# types
type Literal = Tuple[int, bool]                 # (variable number, literal is True)
type Clause = Tuple[Literal, Literal, Literal]  # (Clause[0] OR Clause[1] OR Clause[2])
type Formula = List[Clause]

# constants
DEAD_THRESHOLD = 0.5    # If dead strands/bindings fraction is above this threshold, reindex
ANNEALING_REPS = 20
SETTLE_DIFF = 5         # if difference (in number of strand or binds) is below this between steps, it's settled
SETTLE_ITER = 3         # If sample is settled (as defined above) for this many iterations, stop the step_until_settled run

# global variables
temperature = 55

# functions

# TODO: implement magnetic selection routine
# TODO: Implement magnetic strand removal routine, which heats the sample to unbind all and then removes magnetic strands

# TODO: use "step" in such a way that annealing depends on temperature

# TODO: implement "initial selection" routine which filters by strand size, start with a_0 x_0 and end with x_{n-1} a_n


def run(formula: Formula) -> None:

