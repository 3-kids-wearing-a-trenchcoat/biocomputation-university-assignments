from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import binding
import strand

# constants
# MAG_SELECTION_FAILURE_PROB = 1e-6
# PROCESSES = 8   # max number of ProcessPoolExecutor workers that can run simultaneously

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