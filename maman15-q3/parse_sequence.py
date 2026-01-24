from __future__ import annotations
from typing import List, Dict
import numpy as np
from numpy.typing import NDArray


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

def uint_to_binary(x: int, width: int = 5) -> NDArray[np.bool]:
    """
    Convert the given unsigned integer into a boolean NDArray representing its binary value
    :param x: uint to convert
    :param width: desired minimum width of binary representation
    :return: NDArray[np.bool]
    """
    if x < 0 or x != int(x):
        raise ValueError("value must be a non-negative integer")
    x = int(x)
    bits = bin(x)[2:].zfill(width)
    return convert_str_to_bool_ndarray(bits)

def binary_to_uint(bits: NDArray[np.bool]) -> int:
    return int("".join("1" if b else "0" for b in bits), 2)
    # output_str = "".join("1" if b else "0" for b in bits)
    # return int(output_str, 2)

def bucket_strings_by_prefix(strings: List[str], prefix_len: int = 10,
                             discard_prefix: bool = False) -> Dict[str, List[str]]:
    """
    Divide the input string array into buckets such that each bucket contains strings
    with the same prefix.
    :param strings: List of strings
    :param prefix_len: length of prefix
    :param discard_prefix: whether to discard the prefix in the value field
    :return: Dictionary, key is the prefix and the value is a list of all strings that have that prefix.
    """
    buckets: Dict[str, List[str]] = dict()
    for s in strings:
        if len(s) < prefix_len:
            raise ValueError("At least one of the input strings is shorter than the prefix length")
        key = s[:prefix_len]
        if buckets.get(key) is None:
            buckets[key] = []
        buckets[key].append(s if not discard_prefix else s[prefix_len:])
    return buckets

def np_binary_to_str(arr: NDArray[np.bool]) -> str:
    """Convert boolean numpy array to a string of 0s and 1s"""
    return "".join(["1" if arr[i] else "0" for i in range(len(arr))])