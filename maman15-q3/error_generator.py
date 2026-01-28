from __future__ import annotations
from typing import List
import numpy as np
DEFAULT_PROB = 1e-6 # default error probability for any single event
SUB_OPTIONS = {"T": ["G", "C", "A"], "G": ["T", "C", "A"],
               "C": ["T", "G", "A"], "A": ["T", "G", "C"]}
RNG = np.random.default_rng()

def inject_substitution_error(input_list: List[str], prob: float = DEFAULT_PROB) -> int:
    """
    Inject substitution errors in-place into the input list.
    Each character in every string in the list has a chance of being changed for a different character
    :param input_list: List of strings that may be subjected to errors
    :param prob: Probability in [0,1] for each character in each string to be substituted
    :return: Total number of errors injected into the list
    """
    if prob == 0:
        return 0

    total_errors = 0
    # TODO: very inefficient, but may not be noticeable
    for i in range(len(input_list)):
        for j in range(len(input_list[i])):
            if RNG.random() <= prob:
                total_errors += 1
                new_c = RNG.choice(SUB_OPTIONS[input_list[i][j]])
                input_list[i] = input_list[i][:j] + new_c + input_list[i][j+1:]
    return total_errors

def inject_deletion_error(input_list: List[str], prob: float = DEFAULT_PROB) -> int:
    """
    Inject deletion errors in-place into the input list.
    Each character in every string in the list has a chance of being deleted.
    :param input_list: List of strings that ay be subjected to errors
    :param prob: Probability in [0,1] for each character in each string to be deleted
    :return: Total number of errors injected into the list
    """
    if prob == 0:
        return 0

    total_errors = 0
    for i in range(len(input_list)):
        entry = np.array(list(input_list[i]))
        mask = (RNG.random(size=entry.size) > prob)
        input_list[i] = "".join(entry[mask].tolist())
        total_errors += np.count_nonzero(~mask)
    return total_errors