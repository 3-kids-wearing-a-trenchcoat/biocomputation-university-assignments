from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from math import comb, log2, ceil
from Droplet import Droplet

def append_to_size(seq: str) -> str:
    """Words in the language are represented by 5-bits, therefor we need the overall sequence length to be a
    multiple of 5. To do that we will append 0s to the seq to complete it to a multiple of 5, if that is necessary.
    The true length of the sequence should be stored somehow to make sure we can tell apart junk appended data
    and real data"""
    rem = len(seq) % 5
    if rem == 0:    # if it is divisible by 5, no appending is needed
        return seq
    output = seq + ("0" * rem)
    return output

def split_into_unique_segments(seq: str, div_by: int = 0) -> List[str]:
    """
    Split the input sequence into segments such that each segment is of the same length and is unique.
    This function will attempt to find the smallest segment size that satisfies this requirement.
    If no such division exists, return a list containing a single element, the input sequence, as it trivially
    satisfies these requirements.
    :param seq: string representing sequence to divide
    :param div_by: demand that the segment length will be divisible by this number.
                   Defaults to 0, which is equivalent to not having this requirement
    :return: List of strings, each representing unique, equally sized segment.
             Segments appear in order, meaning concatenating all the output strings in order would yield the input string.
    """
    seq_len = len(seq)
    # length options are all values which the input's length is divisible by AND are divisible by div_by
    len_options = [i for i in range(1, seq_len, 1) if (seq_len % i == 0) and (i % div_by == 0)]
    # search through length candidates in descending order, for each length check if it forms unique segments
    for l in len_options:
        segments = [seq[i: i + l] for i in range(0, seq_len, l)]    # generate segments
        # check uniqueness, if the segments are unique, than the set formed from these segments will have the same
        # number of elements, since no element would be pruned due to uniqueness
        if len(segments) == len(set(segments)):
            return segments # if the segments really are unique, return them
        # print("failed for length " + str(l) + ". size of set is " + str(len(set(segments))) + " whereas num of segments is " + str(len(segments)) +".")
    return [seq]    # no satisfactory segmentation found, return list containing the input

def convert_str_to_bool_ndarray(s: str) -> NDArray[np.bool]:
    bool_list = [bool(int(c)) for c in s]
    return np.fromiter(bool_list, dtype=np.bool)

def convert_segments_to_bool_ndarrays(segments: List[str]) -> List[NDArray[np.bool]]:
    return [convert_str_to_bool_ndarray(s) for s in segments]

def calc_number_of_seeds(segment_num: int) -> Tuple[int, int]:
    """
    Calculate the number of possible seeds a droplet can choose as a function of the number of segments
    :param segment_num: number of segments the input sequence was divided to
    :return: Tuple containing these two ints in order:
             1. Number of possible seeds that a droplet should choose
             2. Number of bits needed to represent these values
    """
    seg_permutations = sum([comb(segment_num, rank) for rank in Droplet.POSSIBLE_RANKS])
    return seg_permutations, ceil(log2(seg_permutations))


# ===== TEST =====
# if __name__ == "__main__":
#     from main import EXAMPLE_SEQUENCE
#     assert len(EXAMPLE_SEQUENCE) == len(append_to_size(EXAMPLE_SEQUENCE))
#     split = split_into_unique_segments(EXAMPLE_SEQUENCE, 5)
#     print("length of overall sequence is " + str(len(EXAMPLE_SEQUENCE)))
#     print("len of each sequence is " + str(len(split[0])))
#     print("number of sequences is " + str(len(split)))