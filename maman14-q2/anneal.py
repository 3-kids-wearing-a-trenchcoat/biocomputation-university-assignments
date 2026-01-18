from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from threading import RLock
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Iterable
import strand
from LazyLock import LazyLock
from binding import (_A_id, _B_id, _A_start, _B_start, _length, _active, _strength, add_bind, calc_strength,
                     _lock, rng, get_bound_strands, _bulk_add)


ANNEALING_FAILURE_PROB = 1e-5   # probability that two strands wouldn't anneal despite being neighbors

# ==========ANNEALING FUNCTIONS==========

# @njit
def _keep_only_unique(pairs: NDArray, hosts: NDArray|None = None) -> NDArray|Tuple[NDArray, NDArray]:
    """
    Given a numpy array of shape (N,2) (effectively an array of tuples of size 2), remove from the array
    all tuples which contain a value that appeared in a previous tuple and return it.
    """
    if pairs.shape[1] != 2:
        raise ValueError("`pairs` must be an (N,2) shaped NDArray")
    N = pairs.shape[0]      # number of pairs
    flat = pairs.ravel()    # flatten pairs
    # unique_vals is a list of the unique values found
    # first_flat_idx matches - for every cell i in flat - the index in unique_vals containing its value
    unique_vals, first_flat_idx = np.unique(flat, return_index=True)
    first_row_of_unique = first_flat_idx // 2   # row of first appearance

    # map each flat value to index in unique_vals
    idx_into_unique = np.searchsorted(unique_vals, flat)
    # get for each flat position the row of first instance
    first_row_flat = first_row_of_unique[idx_into_unique]   # length 2*N
    # reshape back to (N,2) and require both elements' first_row == current_row
    first_row_pairs = first_row_flat.reshape(N,2)
    row_indices = np.arange(N)[:,None]
    keep_mask = np.all(first_row_pairs == row_indices, axis=1)  # shape (N,)

    if hosts is None:
        return pairs[keep_mask]
    return pairs[keep_mask], hosts[keep_mask]

def _replace_binds_of_annealing_host(id_a:int, id_b:int, id_host:int, id_new:int) -> List[int]:
    """
    the new strand C (id_new) has just been created by appending B (id_b) to the end of A (id_a).
    This function replaces the two binds `A <-> host` and `B <-> host` with a single `C <-> host` binding.
    :param id_a: id of strand making up the left part of the new strand
    :param id_b: id of strand making up the right part of the new strand
    :param id_host: id of the strand to which both A and B are bound adjacently
    :param id_new: id of the strand resulting from the annealing of A and B
    :return: List containing the two id values of the bindings that were just replaced (deleted)
    """
    # global _active
    host_in_A_mask = ((_B_id == id_a) | (_B_id == id_b)) & _active & (_A_id == id_host)
    host_in_B_mask = ((_A_id == id_a) | (_A_id == id_b)) & _active & (_B_id == id_host)
    # sanity check -- should be a total of two entries, one for id_a and one for id_b
    assert np.count_nonzero(host_in_A_mask) + np.count_nonzero(host_in_B_mask) == 2
    # end of sanity check

    # find IDs of old binds
    old_binds = np.nonzero(host_in_A_mask | host_in_B_mask)[0].tolist()
    # order old bind IDs so that index 0 stores the bind between host and A, and index 1 stores the bind between host and B
    if _A_id[old_binds[0]] == id_b or _B_id[old_binds[0]] == id_b:  # if id_b appears in bind 0
        old_binds = [old_binds[1], old_binds[0]]                    # flip order so id_b appears in bind 1
    # select correct length arrays for strands A and host
    start_of_a, start_of_host = (_A_start, _B_start) if _A_id[old_binds[0]] == id_a else (_B_start, _A_start)
    # get start and length values for new bind
    start_new = start_of_a[old_binds[0]]
    new_length = _length[old_binds[0]] + _length[old_binds[1]]
    start_host = start_of_host[old_binds[0]]
    # calculate strength
    sum_of_bounds_nucs = _get_num_of_bound_nucs_in_bind(old_binds).sum()
    lengths = strand.get_length()
    total_nucs = lengths[id_new] + lengths[id_host]
    strength = sum_of_bounds_nucs / total_nucs
    # (soft) delete old binds
    _active[old_binds] = False
    # create new bind
    add_bind(id_new, start_new, id_host, start_host, new_length, strength)

    return old_binds

def _get_num_of_bound_nucs_in_bind(bind_id:int|NDArray|List) -> int|NDArray:
    """
    Determine the number of bound nucleotides in one or several binds.
    Binding strength is defined as the fraction of the number of total nucleotides in the bind (total nucs in A
    plus total nucs in B) which are bound to a complementary nucleotide.
    Since we know the final strength, we can extract it by multiplying strength by sum of lengths of A and B.
    :param bind_id: int or NDArray of ints -- bind ID we want to check
    :return: number of bound nucleotides -- if NDArray was given, return NDArray where index `i` represents the number
                                            of bound nucleotides for the binding `bind_id[i]`.
                                            If the specified bind is not active, return 0.
    """
    if type(bind_id) == List:
        bind_id = np.array(bind_id)
    length = strand.get_length()[bind_id]
    strength = _strength[bind_id]
    strength[~_active] = 0  # strength of inactive binds is 0
    id_a, id_b = _A_id[bind_id], _B_id[bind_id]
    return (strength * (length[id_a] + length[id_b])).round().astype(np.uint16)

def _merge_strands(id_a:int, id_b:int, host_id:int) -> None:
    """
    Unite strand A and strand B by appending strand B to strand A.
    Will update the strand data as well as the binding data.
    For the sake of the simulation, this function has a small probability of not doing anything
    :param id_a: id of strand A
    :param id_b: id of strand B, which will be appended to the end of strand A
    :param host_id: id of host strand. The host strand is the strand to which A and B are connected in such a way
                    that they can be merged.
    """
    if rng.random() < ANNEALING_FAILURE_PROB:
        return  # annealing has a small chance (equal to ANNEALING_FAILURE_PROB) of not happening
    # TODO: If this were part of a parallelized process, this should use locks of some kind.

    new_id = strand.merge_strands(id_a, id_b)       # create merged strand and delete the two input strands
    # initialize arrays that will be used by several portions of this function
    strand_lens = strand.get_length()
    a_len, b_len = strand_lens[id_a], strand_lens[id_b]

    # TODO: Add strength calculation for everything after host

    # create binding to replace host binds
    old_host_binds: List[int] = _replace_binds_of_annealing_host(id_a, id_b, host_id, new_id)

    # get Indexes for which rows reference id_a / id_b on each side (excluding inactive binds)
    a_in_A_mask = (_A_id == id_a) & _active
    a_in_B_mask = (_B_id == id_a) & _active
    b_in_A_mask = (_A_id == id_b) & _active
    b_in_B_mask = (_B_id == id_b) & _active
    a_in_A, a_in_B = np.nonzero(a_in_A_mask)[0], np.nonzero(a_in_B_mask)[0]
    b_in_A, b_in_B = np.nonzero(b_in_A_mask)[0], np.nonzero(b_in_B_mask)[0]

    # id_a in field A (host was bound to end of A, so all bind here end before the end of a)
    id_other = _B_id[a_in_A]        # a is in field A, so necessarily the other is in field B
    start_other = _B_start[a_in_A]  # start for other remains unchanged
    start_new = _A_start[a_in_A]    # the start in the new strand is the same as the old a
    length = _length[a_in_A]        # length unchanged
    new_id_arr = np.full(len(length), new_id, dtype=np.uint32)
    # calc strength
    other_strand_length = strand_lens[id_other]
    total_strand_length = other_strand_length + a_len
    comp_num = _get_num_of_bound_nucs_in_bind(a_in_A)
    strength = (comp_num / total_strand_length).round()
    # add new and delete old
    _bulk_add(new_id_arr, start_new, id_other, start_other, length, strength)
    _active[a_in_A] = False

    # id_a in field B               (same as id_a in A, but with fields switched)
    id_other = _A_id[a_in_B]
    start_other = _A_start[a_in_B]
    start_new = _B_start[a_in_B]
    length = _length[a_in_B]
    new_id_arr = np.full(len(length), new_id, dtype=np.uint32)
    # calc strength
    other_strand_length = strand_lens[id_other]
    total_strand_length = other_strand_length + b_len
    comp_num = _get_num_of_bound_nucs_in_bind(a_in_B)
    strength = (comp_num / total_strand_length).round()
    # add new and delete old
    _bulk_add(new_id_arr, start_new, id_other, start_other, length, strength)
    _active[a_in_B] = False

    # id_b in field A (host was bound to the start B, so all binds here start after the start of B)
    id_other = _B_id[b_in_A]
    start_other = _B_start[b_in_A]
    start_new = _A_start[b_in_A] + a_len
    length = _length[b_in_A]
    new_id_arr = np.full(len(length), new_id, dtype=np.uint32)
    # calc strength
    other_strand_length = strand_lens[id_other]
    total_strand_length = other_strand_length + a_len
    comp_num = _get_num_of_bound_nucs_in_bind(b_in_A)
    strength = (comp_num / total_strand_length).round()
    # add new and delete old
    _bulk_add(new_id_arr, start_new, id_other, start_other, length, strength)
    _active[b_in_A] = False

    # id_b in field B
    id_other = _A_id[b_in_B]
    start_other = _A_start[b_in_B]
    start_new = _B_start[b_in_B] + a_len
    length = _length[b_in_B]
    new_id_arr = np.full(len(length), new_id, dtype=np.uint32)
    # calc strength
    other_strand_length = strand_lens[id_other]
    total_strand_length = other_strand_length + b_len
    comp_num = _get_num_of_bound_nucs_in_bind(b_in_B)
    strength = (comp_num / total_strand_length).round()
    # add new and delete old
    _bulk_add(new_id_arr, start_new, id_other, start_other, length, strength)
    _active[b_in_B] = False

    # TODO: I can't even imagine how to begin debugging this, this is a candidate for fuck ups


def bulk_anneal() -> None:
    """Anneal all neighboring strands that are bound to the same third strand with a small probability of failure."""
    # get ids, start position (relative to host) and bind lengths each strand bound to host
    bound_ids, bind_start = np.array([], dtype=np.uint32), np.array([], dtype=np.uint16)
    length, hosts = np.array([], dtype=np.uint16), np.array([], dtype=np.uint32)
    for host_id in strand.get_living_ids():
        # bound_ids, bind_start, length = get_bound_strands(host_id)
        func_output = get_bound_strands(host_id)
        if func_output[0].size == 0:  # if host has no bindings
            continue            # skip
        bound_ids = np.append(bound_ids, func_output[0])
        bind_start = np.append(bind_start, func_output[1])
        length = np.append(length, func_output[2])
        hosts = np.append(hosts, np.full(bound_ids.size, host_id))
    # Candidates are those whose end position is immediately before the another's start position
    candidate_mask = (bind_start[:-1] + length[:-1]) == bind_start[1:]
    candidates = np.column_stack((bound_ids[:-1][candidate_mask], bound_ids[1:][candidate_mask]))
    # Filter the candidates list so that no value in any pair appears in an earlier pair.
    # This is to avoid having to concatenate anneals. that is, new strands produced in this run of bulk_anneal
    # will not merge with any other strand during this run.
    candidates, host = _keep_only_unique(candidates, hosts)
    # TODO: the following section could probably be parallelized
    [_merge_strands(pair[0], pair[1], host) for pair, host in zip(candidates, host)]