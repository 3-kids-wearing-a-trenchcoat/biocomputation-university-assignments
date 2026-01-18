from __future__ import annotations
import binding, strand, anneal, unravel, magnetic, SATinit, sequences, PCR
from SATinit import variable_rep_false, variable_rep_true, connector_rep, complement_rep
from TwoBitArray import TwoBitArray
from typing import List, Tuple
from tqdm import tqdm, trange
import numpy as np
from numpy.typing import NDArray

# constants
ELECTROPHORESIS_FAILURE_PROB = 1e-6 # probability of strand to be wrongly selected or not selected by length

# functions

# ========== reindex ==========
def check_and_reindex(threshold:float) -> None:
    """Check if there's a need to compactify the data, if so, reindex."""
    reindex_binding= binding.get_dead_fraction() > threshold
    # reindex strands (and compactify sequences) if either strands or sequences are above threshold
    reindex_strand = strand.get_dead_fraction() > threshold or sequences.get_dead_fraction() > threshold
    # if strands above threshold, reindex and get the new IDs
    new_ids = strand.reindex() if reindex_strand else None
    if reindex_binding or reindex_strand:   # if bindings are above threshold OR strands were reindexed
        binding.reindex(new_ids)            # Will also update IDs if strand was reindexed

# ========== step ==========
def step(temperature: int, perform_annealing:bool = False, report:bool = False, tqdm_pos:int = 1) -> None|Tuple[int, int]:
    """
    Perform a single "step" in the simulation that is analogous to letting the sample "do its thing" uninterrupted.
    These steps include stochastic binding, stochastic annealing and stochastic unbinding.
    :param temperature: temperature of the sample
    :param perform_annealing: Whether to perform annealing. Defaults to 'False'
                              A cheap approximation of adding enzymes in the right temperature.
    :param report: Whether to report the num of strands and binds after the step is done. Defaults to 'False'
                   May be useful for determining if the sample has settled.
    :param tqdm_pos: pos value for tqdm, defaults to 1.
    :return: None if report is 'False', otherwise return a tuple of two int containing (in order)
             1. Number of (living) strands in the sample after the step ran its course
             2. Number of (active) binds in the sample after the step ran its course
    """
    tqdm_total = 3 if perform_annealing else 2
    with tqdm(total=tqdm_total, leave=False, desc="simulation step", position=1) as p:
        if perform_annealing:
            p.set_postfix_str("Annealing")
            anneal.bulk_anneal()
            p.update(1)
        p.set_postfix_str("Binding")
        binding.bulk_bind(1)
        p.update(1)
        p.set_postfix_str("Unravelling")
        unravel.bulk_unravel(temperature)
        p.update(1)
    if report:
        return strand.get_active_num(), binding.get_active_num()
    else:
        return None

def step_until_settle(temperature: int, diff: int, stop_iter: int, with_annealing:bool = False) -> None:
    """
    Run 'step()' until the sample is "settled" for several consecutive iterations
    :param temperature: Temperature of the sample
    :param diff: If the difference between strand or binding number between `step` iterations is below this,
                 The iteration is considered "Settled".
    :param stop_iter: Stop the loop when we reach this many settled iterations.
    :param with_annealing: Whether to include annealing in step, defaults to False
    """
    settled_iter = 0
    strand_num, bind_num = strand.get_active_num(), binding.get_active_num()
    with tqdm(desc="Letting sample settle", position=1, leave=False, unit="iters") as p:
        while settled_iter < stop_iter:
            new_nums = step(temperature, with_annealing, True, 2)

            strand_diff, bind_diff = strand_num - new_nums[0], bind_num - new_nums[1]
            if strand_diff <= diff or bind_diff <= diff:
                settled_iter += 1
            else:
                settled_iter = 0

            p.set_postfix(strand_num_change=strand_diff,
                          bind_num_change=bind_diff,
                          settled_iterations=settled_iter,
                          refresh=False)
            p.update()
            strand_num, bind_num = new_nums

# ========== gel electrophoresis ==========
def get_ids_bound_to_length(length:int) -> NDArray[np.uint32]:
    """Get IDs of strands that are of the specified length, bound to a strand of the specified length
    or indirectly bound to such a strand.
    PROCESS HAS A SMALL CHANCE OF FAILURE FOR EACH STRAND"""
    # choose strands of right length
    length_mask = strand.get_length_mask(length)
    # small chance of wrongly selecting or not selecting each strand
    failure = np.random.default_rng().random(len(length_mask)) <= ELECTROPHORESIS_FAILURE_PROB  # failed selection
    length_mask[failure] = ~length_mask[failure]  # flip selection/rejection by failure

    # find all IDs bound (directly or indirectly) to previously selected strands
    selected = np.nonzero(length_mask)[0].astype(np.uint32)
    found = selected.copy()
    while True:
        found = binding.get_bound_ids(found)  # get ids of strands bound to those we found last time
        if found.size == 0 or np.isin(selected, found).all():  # if no new IDs found
            return selected  # no more strands to find, return output
        # add newly found IDs to output
        selected = np.union1d(selected, found)

def electrophoresis(length: int) -> None:
    """Remove from sample all strands that are not of the specified length or are bound (directly or indirectly)
    To a strand of the specified length."""
    with tqdm(total=3, desc="gel electrophoresis", position=1, dynamic_ncols=True, leave=False) as p:
        p.set_postfix_str("Selecting by length")
        # Select strands to discard
        idx = np.arange(strand.get_length().size, dtype=np.uint32)
        keep_mask = np.in1d(idx, get_ids_bound_to_length(length), assume_unique=True)
        discard_mask = ~keep_mask
        discard_ids = np.nonzero(discard_mask)[0].astype(np.uint32)
        p.update(1)

        p.set_postfix_str("remove bindings of wrong length")
        # delete binds with strands to discard
        binding.delete_all_with_strand_id(discard_ids)
        p.update(1)

        p.set_postfix_str("remove strands of wrong length")
        # delete all strands to discard
        strand.bulk_delete(discard_ids)
        p.update(1)

# ========== magnetic selection ==========
type Literal = Tuple[int, bool]                 # (variable number, literal is True)
type Clause = Tuple[Literal, Literal, Literal]  # (Clause[0] OR Clause[1] OR Clause[2])

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

def add_magnetic_strands_by_clause(clause: Clause, copies:int=strand.get_entry_num()/2):
    """
    Create magnetic strands out of the input clause and add them to the sample.
    For each literal in the clause, two nucleotide sequences are created, one for the desired expression and
    another for its complement.
    :param clause: Tuple of 3 literals, each literal is (variable index, whether it needs to be True of False)
    :param copies: Number of copies for each magnetic strand, defaults to half the number of living strands
    :return:
    """
    seq0, seq1, seq2 = generate_constraints(clause)
    for _ in trange(copies, desc="adding magnetized strands", position=1, leave=False):
        strand.new_strand(seq0, True)
        strand.new_strand(~seq0, True)
        strand.new_strand(seq1, True)
        strand.new_strand(~seq1, True)
        strand.new_strand(seq2, True)
        strand.new_strand(~seq2, True)

def generate_init_selection_magnetic_strands(n:int) -> List[TwoBitArray]:
    start_seq_t = connector_rep[0].concat(variable_rep_true[0]).concat(connector_rep[1])
    start_seq_f = connector_rep[0].concat(variable_rep_false[0]).concat(connector_rep[1])
    end_seq_t = connector_rep[n - 1].concat(variable_rep_true[n - 1]).concat(connector_rep[n])
    end_seq_f = connector_rep[n - 1].concat(variable_rep_false[n - 1]).concat(connector_rep[n])
    output = [start_seq_t, start_seq_f, end_seq_t, end_seq_f]
    # add complements of the above to output
    [output.append(~output[i]) for i in range(3)]
    return output

# actual selection and clearing should be done in SAT_experiment

