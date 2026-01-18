"""This module describes the run of an experiment for a given 3SAT formula"""

from __future__ import annotations
import strand, magnetic, SATinit, SAT_routines
from PCR import PCR
from typing import List, Tuple
from tqdm import tqdm, trange
from math import ceil

# types
type Literal = Tuple[int, bool]                 # (variable number, literal is True)
type Clause = Tuple[Literal, Literal, Literal]  # (Clause[0] OR Clause[1] OR Clause[2])
type Formula = List[Clause]

# constants
DEAD_THRESHOLD = 0.5    # If dead strands/bindings fraction is above this threshold, reindex
PCR_REPS = 5
SETTLE_DIFF = 100         # if difference (in number of strand or binds) is below this between steps, it's settled
SETTLE_ITER = 3         # If sample is settled (as defined above) for this many iterations, stop the step_until_settled run

# global variables
temperature = 55

# functions
def settle() -> None:
    SAT_routines.step_until_settle(temperature, SETTLE_DIFF, SETTLE_ITER, temperature == 75, temperature > 90)

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
    with tqdm(mag_strands, desc="adding magnetized strands", position=1, leave=False, dynamic_ncols=True) as prog:
        for seq in prog:
            [strand.new_strand(seq, True) for _ in trange(copies_per_strand, position=2, leave=False, dynamic_ncols=True,
                                                          desc="generating copies of strand", miniters=100)]
    temperature = 55
    settle()
    magnetic.magnetic_selection(False)  # remove all not matching criteria
    temperature = 95
    settle()
    magnetic.magnetic_selection(True)   # remove magnetic strands from sample

def run(formula: Formula) -> bool:
    """Main Experiment sequence. returns 'True' iff formula is satisfiable"""
    global temperature
    # initialize experiment
    n = SATinit.init_3sat(formula)  # n is the number of variables

    with tqdm(total=5, desc="Running", leave=True, position=0, dynamic_ncols=True) as p:
        p.set_postfix_str("Allowing strands to bind and anneal into candidate solutions")
        temperature = 75        # set for annealing
        settle()                # allow initial strands to bind into full strands representing solutions
        p.update(1)

        p.set_postfix_str("unraveling binds in preparation to initial filter")
        # unravel binds in preparation for initial selection
        temperature = 95
        settle()
        p.update(1)

        p.set_postfix_str("select by expected length and possible start/end sequences")
        # filter out all candidates that are not of the expected length or beginning/end possibilities
        initial_selection(n)
        PCR(PCR_REPS)
        p.update(1)

        p.set_postfix_str("Go over every clause in the formula and exclude all strands that do not satisfy it")
        with tqdm(formula, position=1, leave=False, desc="Filtering by clause", dynamic_ncols=True) as formula_prog:
            for clause in formula_prog:
                if strand.empty_sample():
                    return False    # if sample becomes empty, we can return False right away
                formula_prog.set_postfix_str("reindexing")
                SAT_routines.check_and_reindex(DEAD_THRESHOLD)

                formula_prog.set_postfix_str("magnetic selection")
                magnetic_selection(clause)

                PCR(PCR_REPS, 2)
                formula_prog.set_postfix_str("PCR")
        p.update(1)

        if strand.empty_sample():
            return False

        p.set_postfix_str("once again filtering by expected length and possible start/end sequences")
        initial_selection(n)
        p.update(1)

        return not strand.empty_sample()    # if sample is not empty, formula is satisfiable


