from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import binding
import strand
from typing import List, Tuple
from TwoBitArray import TwoBitArray
from concurrent.futures import ProcessPoolExecutor

# constants
MAG_SELECTION_FAILURE_PROB = 1e-4
PROCESSES = 8   # max number of ProcessPoolExecutor workers that can run simultaneously

# functions
def find_all_attached_to_magnetic() -> NDArray[np.uint32]:
    """
    Get an id NDArray strands which are alive and are either magnetic or attached to a magnetic strand.
    A strand is considered attached to a magnetic strand if it bound directly to one or bound to a strand
    which is directly or indirectly bound to a magnetic strand.
    """
    output = strand.get_magnetic_id()
    found = output.copy()
    while True:
        found = binding.get_bound_ids(found)                # get ids of strands bound to those we found last time
        if found.size == 0 | np.isin(output, found).all():  # if no new IDs found
            return output                                   # no more strands to find, return output
        # add newly found IDs to output
        output = np.union1d(output, found)

def magnetic_selection(remove_selection:bool = False) -> None:
    """
    Filter the sample by magnetic selection.
    Magnetic selection selects all magnetic strands in the sample and all strands bound to a magnetic strand,
    directly or indirectly.
    It then either removes all selected strands from the sample, or removes all strands BUT the selected strands
    from the sample.
    :param remove_selection: If 'False' (default) -- Remove all strands BUT the selected strands from the sample
                             If 'True' -- Remove selected strands from the sample.
    """
    selected = find_all_attached_to_magnetic()
    # If selected are to be removed
    if remove_selection:
        strand.bulk_delete(selected)
        binding.delete_all_with_strand_id(selected)
        return
    # If everything BUT selected is to be removed
    # find strand IDs to remove
    N = strand.get_entry_num()
    mask = np.ones(N, dtype=np.bool)
    mask[selected] = False
    to_delete = np.nonzero(mask)[0].astype(np.uint32)
    strand.bulk_delete(to_delete)
    # delete binds containing the deleted strands
    binding.delete_all_with_strand_id(to_delete)



# def filter_blocking_binds_from_selection(ids: NDArray[np.uint32], patterns:List[TwoBitArray]) -> (
#                                                                         Tuple[NDArray[np.uint32], NDArray[np.uint32]]):
#     """
#     Of the strand IDs found to have matching patterns, check if any are bound in such a place that would prevent
#     this pattern selection from happening by blocking the section a (virtual) magnetic strand could bind to.
#     Any strand ID that is actually blocked from selection by this bind is filtered out of the selection
#     :param ids: Current selection of all strands matching a pattern (NDArray of uint32)
#     :return: Tuple of two uint32 NDArrays, they are in order:
#              1. Filtered selection of all strands matching a pattern   (subset of ids)
#              2. IDs of strands that are bound to some other strand but are still selected
#     """
#     # Of the patterned strands selected, select all the ones that are currently bound to some other strand
#     bound_ids = binding.is_bound(ids)
#     # find if any of those bound strands are bound in spots that would not allow a (virtual) magnetic strand to bind
#     bound_strands = [binding.get_bound_strands(bound_id) for bound_id in bound_ids]
#     # bound_strands (list) -- for every index `i`, bound_strands[i] == (ids_bound_to_i, bind_start_on_i, bind_length)
#
#
# def get_strands_with_pattern(patterns:List[TwoBitArray]) -> NDArray[np.uint32]:
#     """
#     Get all the living strands that contain at least one of the input patterns.
#     Each selected strand has a small probability of not being selected, for simulation reasons.
#     :param patterns: List of TwoBitArray representing a pattern to search for
#     :return: NDArray of strand IDs that contain at least one of the patterns in `patterns`
#     """
#     # TODO: parallelize if I have time -- that is, run several `get_pattern_mask` instances at once
#     # get a mask representing which strands contain one of the patterns in them
#     mask_list = [strand.get_pattern_mask(pattern) for pattern in patterns]
#     mask = np.logical_or.reduce(mask_list)
#     # Each selection has a small probability of being discarded despite fitting
#     rnd = binding.rng.random(len(mask))
#     mask &= (rnd <= MAG_SELECTION_FAILURE_PROB)
#     # translate mask to indexes (strand IDs)
#     ids = np.nonzero(mask)[0].astype(np.uint32)
#
#     # TODO: Add to the output all IDs of strands that are bound to the previously selected strands
#     ids_bound_to_selections =
#
#     return ids
#
# def filter_by_patterns(patterns:List[TwoBitArray], remove_selection:bool = False) -> None:
#     """
#     Select all strands in the sample which contain one of the input patterns and either remove them or remove all
#     strands except for those selected.
#     Selection has a small chance of failing to select an otherwise fitting candidate.
#     :param patterns: List of TwoBitArrays representing genetic patterns.
#     :param remove_selection: Defaults to 'False' - if set to 'True', remove selected strands.
#                              If set to 'False', remove all strands except for those selected by the patterns.
#     """
#     selected = get_strands_with_pattern(patterns)
#     if remove_selection:    # remove all strands matching any of the patterns
#         strand.bulk_delete(selected)
#     else:                   # remove all strands EXCEPT for the strands matching any of the patterns
#         living_ids = strand.get_living_ids()
#         mask = np.isin(living_ids, selected)    # select all positions of living_id that contain selected strands
#         ids_to_remove = living_ids[~mask]       # select all positions of living_id NOT containing selected strands
#         strand.bulk_delete(ids_to_remove)       # remove those strands