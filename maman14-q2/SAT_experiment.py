"""This module describes the run of an experiment for a given 3SAT formula"""

from __future__ import annotations
import binding, strand, anneal, unravel, magnetic, SATinit
from SATinit import variable_rep_false, variable_rep_true, connector_rep, complement_rep
from TwoBitArray import TwoBitArray
from typing import List, Tuple
from tqdm import tqdm, trange

# types
type Literal = Tuple[int, bool]                 # (variable number, literal is True)
type Clause = Tuple[Literal, Literal, Literal]  # (Clause[0] OR Clause[1] OR Clause[2])
type Formula = List[Clause]

# constants
ANNEALING_REPS = 20

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
# TODO: Implement "cool" routine which binds, anneals and unbinds depending on temperature
# TODO: Implement "heat" routine which unbinds depending on temperature
# TODO: (probably?) Implement annealing routine which works until no significant change is made
# TODO: (maybe?) Implement bind routine which works until no significant change is made
# TODO: (maybe?) Implement unbind routine which works until no significant change is made
# TODO: Implement primer seeding
# TODO: Implement PCR routine, that is, throw in primers and let them bind
# TODO: implement "initial selection" routine which filters by strand size, start with a_0 x_0 and end with x_{n-1} a_n


def run(formula: Formula) -> None:

