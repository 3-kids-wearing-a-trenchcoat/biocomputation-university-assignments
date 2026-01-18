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
def generate_constraints(clause: Clause) -> Tuple[TwoBitArray, TwoBitArray, TwoBitArray]:
    """Generate 3 TwoBitArray objects representing the input clause."""
    var0, var1, var2 = clause[0][0], clause[1][0], clause[2][0]
    true0, true1, true2 = clause[0][1], clause[1][1], clause[2][1]
    lit0 = (variable_rep_true[var0] if true0 else variable_rep_false[var0]).concat(connector_rep[var0 + 1])
    lit1 = (variable_rep_true[var1] if true1 else variable_rep_false[var1]).concat(connector_rep[var1 + 1])
    lit2 = (variable_rep_true[var2] if true2 else variable_rep_false[var2]).concat(connector_rep[var2 + 1])
    if var0 > 0:
        lit0 = connector_rep[var0 - 1].concat(lit0)
    if var1 > 0:
        lit1 = connector_rep[var1 - 1].concat(lit1)
    if var2 > 0:
        lit2 = connector_rep[var2 - 1].concat(lit2)
    return lit0, lit1, lit2

# TODO: implement magnetic selection routine
# TODO: Implement magnetic strand removal routine, which heats the sample to unbind all and then removes magnetic strands

# TODO: use "step" in such a way that annealing depends on temperature

# TODO: implement "initial selection" routine which filters by strand size, start with a_0 x_0 and end with x_{n-1} a_n


def run(formula: Formula) -> None:

