from __future__ import annotations
import binding, strand
from TwoBitArray import TwoBitArray
import numpy as np
from numpy.typing import NDArray

FAILURE_PROB = 1e-6

def get_unbound_strand_ids() -> NDArray[np.uint32]:
    living_strands = strand.get_living_ids()
    bound_strands = binding.get_all_bound_strands()
    bound_living_mask = np.in1d(living_strands, bound_strands, assume_unique=True)
    return living_strands[~bound_living_mask]

def PCR() -> None:
    """
    PCR implementation that's computationally cheaper than faithfully simulating PCR.
    Duplicate all strands that are alive and not bound to another strand with a small probability
    of failure for each duplication
    """
    to_dup = get_unbound_strand_ids()
    prob = np.random.default_rng().random(len(to_dup))
    to_dup = to_dup[prob <= FAILURE_PROB]   # have some duplications randomly fail
    [strand.new_strand(strand.get_seq(idx)) for idx in to_dup]