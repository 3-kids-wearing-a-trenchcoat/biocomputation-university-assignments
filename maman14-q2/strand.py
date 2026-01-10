from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import sequences
from TwoBitArray import TwoBitArray
from bitarray import bitarray
import threading
from typing import List, Tuple

# ID itself is the index value, so the offset and length of strand i are at index i of the offset and length arrays respectively
_offset = np.empty(0,dtype=np.uint32)   # starting position of strand's sequence in sequences
_length = np.empty(0, dtype=np.uint8)   # length of strand's sequence in sequences, starting from offset
# _active = np.empty(0, dtype=bool)       # whether strand is "alive"
_active = bitarray()                          # whether strand is alive
_lock: threading.RLock = threading.RLock()   # Avoid race-conditions as a result of parallel writing to the above arrays

def get_seq(strand_id:int, decoded:bool = False) -> TwoBitArray|List:
    """
    Get the genetic sequence of a DNA strand
    :param strand_id: (int) strand ID
    :param decoded: (bool) optional -- Whether to decode the sequence into nucleotide letter representation
    :return: If not decoded, return the TwoBitArray representing the genetic sequence.
             Otherwise, return a list of characters representing the genetic sequence (TGCA)
    """
    offset, length = _offset[strand_id], _length[strand_id]
    if decoded:
        return sequences.decode(offset, length)
    return sequences.get(offset, length)

def new_strand(seq_str: str) -> int:
    global _offset, _length, _active
    _lock.acquire()
    offset, length = sequences.seq_append(seq_str)
    _offset, _length = np.append(_offset, offset), np.append(_length, length)
    # _active = np.append(_active, True)
    _active.append(1)
    output_id = len(_offset) - 1
    _lock.release()
    return output_id

def _delete_strand(strand_id: int) -> None:
    """soft delete -- SHOULD ONLY BE USED WHEN ANNEALING OR SPLITTING STRANDS"""
    _lock.acquire()
    _active[strand_id] = 0
    sequences.make_inactive(_offset[strand_id], _length[strand_id])
    _lock.release()

def reindex() -> NDArray[np.int32]:
    """
    Compactify data by removing all inactive strands.
    This function works under the assumption that all inactive strands are not bound to any other strand
    WITHOUT CHECKING FOR IT, be careful.
    :return: numpy array of the same size as the old _offset and _length, where the value of the i-th cell
             contains the new index of the strand of id i, or -1 if this strand has been pruned.
    """
    global _offset, _length, _active
    # acquire locks
    _lock.acquire()
    sequences.acquire_lock()

    old_to_new = np.full(len(_offset), -1, dtype=np.int32)
    alive = np.fromiter(_active, dtype=bool)    # translate _active to a boolean numpy array for masking
    old_to_new[alive] = np.arange(np.count_nonzero(alive)) # generate old-to-new ID array
    sequences.prune_inactive()  # remove all nucleotides belonging to dead strands
    _offset, _length = _offset[alive], _length[alive]
    _active = bitarray(len(_offset))
    _active.setall(1)

    # release locks
    sequences.release_lock()
    _lock.release()

    return old_to_new

def get_dead_fraction() -> float:
    """Get fraction of strand entries which are dead"""
    return _active.count(0) / len(_active)

def is_complement(strand_id_A:int, strand_id_B:int) -> bool:
    """
    Check if the two strands have element-wise complementary sequences.
    A sequence is element-wise complementary iff for every i-th nucleotide, nucleotide i of strand A
    is complementary (A <-> T or G <-> C) to nucleotide i of strand B.
    Strands of different lengths are by definition never element-wise complementary.
    :param strand_id_A: id of first strand
    :param strand_id_B: id of second strand
    :return: True if the two strands are complementary, otherwise False.
    """
    length_A, length_B = _length[strand_id_A], _length[strand_id_B]
    if length_A != length_B:
        return False
    return sequences.is_complement(strand_id_A, strand_id_B, length_A)

def _find_seed_matches(id_a:int, id_b:int, seed_len:int) -> List[Tuple[int, int]]:
    """
    Find all starting points at which the two given strands can duplex-bind and return them.
    A valid starting point is any 2-tuple (i,j) in which the nucleotide sub-sequences A[i:seed_len] and B[j:seed_len]
    are complementary. This can be thought of as the "starting point" of a duplex bind between strands.
    This function considers only complementarity and does not check whether a nucleotide is not already bound,
    This check should be performed on the output of this function.
    :param id_a: id of strand A
    :param id_b: id of strand B
    :param seed_len: Length of sequence to check at each point
    :return: List of Tuples made up of two ints each.
             The first int representing the start index of the matching sequence in strand A, and the second
             int representing the start index of the matching sequence in strand B.
    """
    seq_a = sequences.get(_offset[id_a], _length[id_a])
    seq_b = ~sequences.get(_offset[id_b], _length[id_b]) # seq_b is inverted as we're looking for complementarity
    output = []
    for j in range(len(seq_b) - seed_len + 1):           # for each possible start position for a seed_len-long sub-sequence of B
        seed = seq_b[j:j+seed_len]
        output.extend([i,j] for i in seq_a.search(seed)) # add all pairs of start indexes for A and B respectively where there's a match
    return output

# TODO: _extend_seed() -- extend seed we got from _find_seed_matches -- return start_A and start_B -- should probably calculate strength here too
# TODO: (maybe) _extend_seeds() -- do _extend_seed() for every seed found