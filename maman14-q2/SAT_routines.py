from __future__ import annotations
import binding, strand, anneal, unravel, magnetic, SATinit, sequences
from SATinit import variable_rep_false, variable_rep_true, connector_rep, complement_rep
from TwoBitArray import TwoBitArray
from typing import List, Tuple
from tqdm import tqdm, trange


def check_and_reindex(threshold:float) -> None:
    """Check if there's a need to compactify the data, if so, reindex."""
    reindex_binding= binding.get_dead_fraction() > threshold
    # reindex strands (and compactify sequences) if either strands or sequences are above threshold
    reindex_strand = strand.get_dead_fraction() > threshold or sequences.get_dead_fraction() > threshold
    # if strands above threshold, reindex and get the new IDs
    new_ids = strand.reindex() if reindex_strand else None
    if reindex_binding or reindex_strand:   # if bindings are above threshold OR strands were reindexed
        binding.reindex(new_ids)            # Will also update IDs if strand was reindexed

def step(temperature: int, perform_annealing:bool = False, report:bool = False) -> None | Tuple[int, int]:
    """
    Perform a single "step" in the simulation that is analogous to letting the sample "do its thing" uninterrupted.
    These steps include stochastic binding, stochastic annealing and stochastic unbinding.
    :param temperature: temperature of the sample
    :param perform_annealing: Whether to perform annealing. Defaults to 'False'
                              A cheap approximation of adding enzymes in the right temperature.
    :param report: Whether to report the num of strands and binds after the step is done. Defaults to 'False'
                   May be useful for determining if the sample has settled.
    :return: None if report is 'False', otherwise return a tuple of two int containing (in order)
             1. Number of (living) strands in the sample after the step ran its course
             2. Number of (active) binds in the sample after the step ran its course
    """
    with tqdm(total=3, leave=False, desc="simulation step", position=1) as p:
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