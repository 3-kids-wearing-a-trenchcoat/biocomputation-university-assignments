from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import sequences
from TwoBitArray import TwoBitArray
from bitarray import bitarray
import threading
from typing import List, Tuple, Iterable

# ID itself is the index value, so the offset and length of strand i are at index i of the offset and length arrays respectively
_offset = np.empty(0,dtype=np.uint32)   # starting position of strand's sequence in sequences
_length = np.empty(0, dtype=np.uint16)   # length of strand's sequence in sequences, starting from offset
_active = bitarray()                          # whether strand is alive
_magnetic = bitarray()                        # whether strand is magnetic (for selection)
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

def new_strand(input_seq: TwoBitArray|Iterable, magnetic:bool=False) -> int|None:
    global _offset, _length, _active
    if input_seq is None:
        return None
    _lock.acquire()
    offset, length = sequences.seq_append(input_seq)
    _offset, _length = np.append(_offset, offset), np.append(_length, length)
    _active.append(1)
    _magnetic.append(magnetic)
    output_id = len(_offset) - 1
    _lock.release()
    return output_id

def _delete_strand(strand_id: int) -> None:
    """soft delete -- SHOULD ONLY BE USED WHEN ANNEALING OR SPLITTING STRANDS"""
    _lock.acquire()
    _active[strand_id] = 0
    sequences.make_inactive(_offset[strand_id], _length[strand_id])
    _lock.release()

def bulk_delete(ids: NDArray[np.uint32]) -> None:
    """Delete all strands in the input array.
    Strands that are already deleted will not be affected."""
    global _active
    _lock.acquire()
    mask = np.ones(len(_active), dtype=np.bool)
    mask[ids] = False
    mask = bitarray(mask.tolist())
    _active &= mask
    sequences.bulk_make_inactive(_offset[ids].tolist(), _length[ids].tolist())
    _lock.release()

def reindex() -> NDArray[np.int64]:
    """
    Compactify data by removing all inactive strands.
    This function works under the assumption that all inactive strands are not bound to any other strand
    WITHOUT CHECKING FOR IT, be careful.
    :return: numpy array of the same size as the old _offset and _length, where the value of the i-th cell
             contains the new index of the strand of id i, or -1 if this strand has been pruned.
    """
    global _offset, _length, _active, _magnetic
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
    new_magnetic_np = np.fromiter(_magnetic, dtype=np.bool)[alive]
    _magnetic = bitarray(new_magnetic_np.tolist())
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

def _extend_seed(id_a:int, nucleation_index_a:int, id_b:int, nucleation_index_b:int) -> Tuple[int, int, int, float]:
    """
    Extend a nucleation index on A and on B (as found in _find_seed_matches) into a duplex-binding range between
    strands A and B and get the start indexes for A and B, the length of the bind and its strength.
    :param id_a: id of strand A
    :param nucleation_index_a: index of nucleation point on strand A
    :param id_b: id of strand B
    :param nucleation_index_b: index of nucleation point on strand B
    :return: Tuple - (start index on strand A for the bind,
                      start index on strand B for the bind,
                      length,
                      fraction of overall nucleotides bound (strength))
    """
    len_a, len_b = _length[id_a], _length[id_b]
    # Bind starts from the start of the strand whose starting position is to the right of the other's starting position
    start_shift = min(nucleation_index_a, nucleation_index_b)
    start_a, start_b = nucleation_index_a - start_shift, nucleation_index_b - start_shift
    # Bind length is equal to the distance between bind start and the nearer of A's end and B's end
    length_a, length_b = len_a - start_a, len_b - start_b
    length = min(length_a, length_b)
    # bind strength is equal to fraction of all nucleotide pairs that are bound to a complementary nucleotide
    bind_seq_a = sequences.get(_offset[id_a] + start_a, length)
    bind_seq_b = sequences.get(_offset[id_b] + start_b, length)
    comp = (bind_seq_a ^ bind_seq_b).count()  # number of complementary pairs, obviously every pair has 2 nucleotides
    # strength = (comp * 2) / (len_a + len_b)   # fraction of all nucleotides in both strands that are bound to a complementary
    strength = comp / length # fraction of nucleotides in binding sequence that are complementary
    return start_a, start_b, length, strength


def get_possible_binds(id_a:int, id_b:int, seed_len:int) -> List[Tuple[int, int, int, float]]:
    """
    Find all possible bindings between strand A and strand B
    :param id_a: id of strand A
    :param id_b: id of strand B
    :param seed_len: length of sequence to check at each point (discard possible binds with fewer matches)
    :return: List of possible bindings, each binding represented by a Tuple containing the following:
                1. Bind start position for strand A (int)
                2. Bind start position for strand B (int)
                3. Bind length (int)
                4. Bind strength - fraction of total nucleotides in both strands bound to a complementary nucleotide (float)
    """
    seeds = _find_seed_matches(id_a, id_b, seed_len)
    return [_extend_seed(id_a, seed[0], id_b, seed[1]) for seed in seeds]

def merge_strands(id_a:int, id_b:int) -> int:
    """
    Merge two existing strands into one by appending the sequence of strand B to the end of strand A to create a new
    strand C and (soft) deleting strands A and B.
    :param id_a: id of strand A
    :param id_b: id of strand B
    :return: id of new merged strand
    """
    _lock.acquire()
    seq_a, seq_b = get_seq(id_a), get_seq(id_b)
    new_seq = seq_a.merge(seq_b)
    _delete_strand(id_a), _delete_strand(id_b)
    new_id = new_strand(new_seq)
    _lock.release()
    return new_id

def split_strand(strand_id, split_index) -> Tuple[int, int]:
    """
    split a strand according to split_index and update data accordingly
    :param strand_id: id of strand to split
    :param split_index: Index according to which the strand is split.
                        If split_index == i, will produce the sub-strands strand[0:i] and strand[i:len(strand)].
    :return: IDs of the two new strands
    """
    _lock.acquire()
    old_seq = get_seq(strand_id)
    seq_a, seq_b = old_seq[0:split_index], old_seq[split_index:len(old_seq)]
    _delete_strand(strand_id)
    id_a, id_b = new_strand(seq_a), new_strand(seq_b)
    _lock.release()
    return id_a, id_b

def get_length_mask(length:int) -> NDArray[np.bool]:
    """
    Get a mask showing the indexes of all active strands of the given length
    :param length: requested length
    :return: boolean NDArray where cell i is 'True' iff strand i is active and if of length 'length'
    """
    is_active = np.fromiter(_active, dtype=np.bool)
    return is_active & (_length == length)

# def filter_by_mask(mask:NDArray[np.bool]) -> None:
#     """
#     (soft) Delete all strands in the data according to the input mask.
#     If cell `i` in the input mask is 'False', strand `i` will be deleted.
#     :param mask: boolean numpy array
#     """
#     global _active, _offset, _length
#     _lock.acquire()
#     np_active = np.fromiter(_active, dtype=np.bool)
#     change_mask = (not mask) & np_active    # active strands to make inactive
#     # make sequences inactive in the sequences module
#     for offset, length in zip(_offset[change_mask], _length[change_mask]):
#         sequences.make_inactive(offset, length)
#     # make strands inactive in this module
#     _active &= bitarray(mask.tolist())
#     _lock.release()
#
# def get_pattern_mask(pattern:TwoBitArray) -> NDArray[np.bool]:
#     """
#     Get a mask for all strands which contain the input pattern
#     :param pattern: TwoBitArray object representing a sequence in a strand
#     :return: boolean numpy array where cell `i` is 'True' iff strand `i` contains the input pattern
#     """
#     output = [(pattern in sequences.get(offset, length))
#               if active else False
#               for offset, length, active in zip(_offset, _length, _active)]
#     return np.asarray(output, dtype=np.bool)

def get_living_ids() -> NDArray[np.uint32]:
    """Get a numpy array of living strands' id values"""
    # equivalent to np.nonzero for a numpy boolean array
    return np.fromiter(_active.search(1), dtype=np.uint32)

def get_active_mask() ->NDArray[np.bool]:
    return np.fromiter(_active, dtype=np.bool)

def get_length() -> NDArray:
    return _length

def is_magnetic(strand_id:int) -> bool:
    return _magnetic[strand_id] == 1

MAG_SELECTION_FAILURE_PROB = 1e-6

def get_magnetic_id(failure_prob:float=MAG_SELECTION_FAILURE_PROB) -> NDArray[np.uint32]:
    """get IDs of all living magnetic strands
    :param failure_prob: probability of magnetic strand to not be selected"""
    magnetic_mask = get_magnetic_mask()
    failure = np.random.default_rng().random(len(magnetic_mask)) <= failure_prob
    magnetic_mask[failure] = False
    return np.nonzero(magnetic_mask)[0].astype(np.uint32)
    # return np.nonzero(get_magnetic_mask())[0].astype(np.uint32)

def get_magnetic_mask() -> NDArray[np.bool]:
    return np.fromiter(_magnetic & _active, dtype=np.bool)

def get_entry_num() -> int:
    return len(_active)

def get_active_num() -> int:
    return _active.count()

def empty_sample() -> bool:
    """returns 'True' if no strands are in the sample, otherwise 'False'"""
    return _active.all() == 0