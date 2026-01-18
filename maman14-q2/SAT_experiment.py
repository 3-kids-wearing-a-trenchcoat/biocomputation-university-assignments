"""This module describes the run of an experiment for a given 3SAT formula"""

from __future__ import annotations
import binding, strand, anneal, unravel, magnetic, SATinit, SAT_routines
from SATinit import variable_rep_false, variable_rep_true, connector_rep, complement_rep
from PCR import PCR
from TwoBitArray import TwoBitArray
from typing import List, Tuple
from tqdm import tqdm, trange
from math import ceil

# types
type Literal = Tuple[int, bool]                 # (variable number, literal is True)
type Clause = Tuple[Literal, Literal, Literal]  # (Clause[0] OR Clause[1] OR Clause[2])
type Formula = List[Clause]

# constants
DEAD_THRESHOLD = 0.5    # If dead strands/bindings fraction is above this threshold, reindex
PCR_REPS = 20
SETTLE_DIFF = 2         # if difference (in number of strand or binds) is below this between steps, it's settled
SETTLE_ITER = 3         # If sample is settled (as defined above) for this many iterations, stop the step_until_settled run

# global variables
temperature = 55

# functions
def settle() -> None:
    SAT_routines.step_until_settle(temperature, SETTLE_DIFF, SETTLE_ITER, temperature == 75)

def magnetic_selection(clause: Clause) -> None:
    """
    Complete magnetic selection routine.
    1. Generate magnetic strands according to the input clause
    2. Insert many copies of these strands into the sample
    3. Allow the magnetic strands to bind and settle
    4. Remove from the sample all strands which are neither magnetic nor bound (directly or indirectly) to a magnetic strand
    5. Undo all binds by heating the sample
    6. Remove all magnetic strands from sample
    """
    global temperature
    # remove all not selected magnetically
    SAT_routines.add_magnetic_strands_by_clause(clause)
    temperature = 55
    settle()
    magnetic.magnetic_selection(False)
    # clear sample of magnetic strands
    temperature = 95
    settle()
    magnetic.magnetic_selection(True)

def initial_selection(n:int) -> None:
    """Filter from sample all strands which do not satisfy the following three conditions:
    1. begin with a_0 x_0 a_1 or a_0 x'_0 a_1
    2. end with a_{n-1} x_{n-1} a_n or a_{n-1} x'_{n-1} a_n
    3. are exactly of length n+n+1 = 2n+1"""
    global temperature  # expecting temperature to be 95 and assuming no bindings (within margin of error)
    # filter by expected length
    SAT_routines.electrophoresis(2*n + 1)
    PCR(PCR_REPS)
    # filter by beginning and end
    copies_per_strand = ceil(SATinit.T / 2)
    mag_strands = SAT_routines.generate_init_selection_magnetic_strands(n)
    with tqdm(mag_strands, desc="adding magnetized strands", position=1, leave=False) as prog:
        for seq in prog:
            [strand.new_strand(seq, True) for _ in trange(copies_per_strand, position=2, leave=False,
                                                          desc="generating copies of strand", miniters=100)]
    temperature = 55
    settle()
    magnetic.magnetic_selection(False)  # remove all not matching criteria
    temperature = 95
    settle()
    magnetic.magnetic_selection(True)   # remove magnetic strands from sample

def run(formula: Formula) -> None:

