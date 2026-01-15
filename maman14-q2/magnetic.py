from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import binding
import strand
from typing import List
from TwoBitArray import TwoBitArray
from concurrent.futures import ProcessPoolExecutor

# constants
MAG_SELECTION_FAILURE_PROB = 1e-4
PROCESSES = 8   # max number of ProcessPoolExecutor workers that can run simultaneously

# functions
def get_strands_with_pattern(patterns:List[TwoBitArray]) -> NDArray[np.uint32]:
    """
    Get all the living strands that contain at least one of the input patterns.
    Each selected strand has a small probability of not being selected, for simulation reasons.
    :param patterns: List of TwoBitArray representing a pattern to search for
    :return: NDArray of strand IDs that contain at least one of the patterns in `patterns`
    """
    # TODO: parallelize if I have time -- that is, run several `get_pattern_mask` instances at once
    # get a mask representing which strands contain one of the patterns in them
    mask_list = [strand.get_pattern_mask(pattern) for pattern in patterns]
    mask = np.logical_or.reduce(mask_list)
    # Each selection has a small probability of being discarded despite fitting
    rnd = binding.rng.random(len(mask))
    mask &= (rnd <= MAG_SELECTION_FAILURE_PROB)
    # translate mask to indexes (strand IDs)
    ids = np.nonzero(mask)[0].astype(np.uint32)

    # Of the patterned strands selected, select all the ones that are currently bound to some other strand
    bound_ids = binding.is_bound(ids)
    # TODO: find if any of those bound strands are bound in spots that would not allow a (virtual) magnetic strand to bind
    # TODO: Add to the output all IDs of strands that are bound to the previously selected strands

    return ids

def filter_by_patterns(patterns:List[TwoBitArray], remove_selection:bool = False) -> None:
    """
    Select all strands in the sample which contain one of the input patterns and either remove them or remove all
    strands except for those selected.
    Selection has a small chance of failing to select an otherwise fitting candidate.
    :param patterns: List of TwoBitArrays representing genetic patterns.
    :param remove_selection: Defaults to 'False' - if set to 'True', remove selected strands.
                             If set to 'False', remove all strands except for those selected by the patterns.
    """
    selected = get_strands_with_pattern(patterns)
    if remove_selection:    # remove all strands matching any of the patterns
        strand.bulk_delete(selected)
    else:                   # remove all strands EXCEPT for the strands matching any of the patterns
        living_ids = strand.get_living_ids()
        mask = np.isin(living_ids, selected)    # select all positions of living_id that contain selected strands
        ids_to_remove = living_ids[~mask]       # select all positions of living_id NOT containing selected strands
        strand.bulk_delete(ids_to_remove)       # remove those strands