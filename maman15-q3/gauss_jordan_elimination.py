"""This module implements Gauss-Jordan elimination over the field GF(2).
Specifically, it implements functions used to solve the augmented linear system Ax=Y over GF(2)
by performing Gaussian elimination on [A|Y] until A becomes the identity matrix, at which point
Y would equal x.
For the purposes of decoding DNA storage, the matrices will be:
    A - The segment mask matrix, A[i][j] == True iff the payload of droplet `i` encodes segment `j`
    Y - the vector Y[i] is a boolean vector representing the payload of droplet `i`
    x - vector x[i] is the exact value of the `i`th segment of the encoded data. (what we're looking for, naturally)"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List
from DropletGenerator import IDX_DTYPE

def form_into_mat(input_list: List[NDArray[np.bool]] | List[NDArray[IDX_DTYPE]]) -> NDArray[np.bool]:
    """
    Form a python list of numpy 1D arrays into a 2D array that can be used for row elimination.
    All vectors in the list must be of the same length and the same dtype.
    :param input_list: A python list made up of 1D numpy arrays, the dtype of the arrays can be one of two:
                        1. **np.bool**, in which case the arrays are just stacked in order
                        2. **np.uint**, in which case each vector is treated as the indexes of 'True'
                            values for that same row. The length of each boolean row will be equal to the max
                            index found in all vectors plus 1.
    :return: boolean numpy 2D matrix of shape (len(input_list), <length of vector>)
    """
    if input_list[0].dtype == np.bool:  # if input_list is a binary string
        output = np.stack(input_list, dtype=np.bool)
        return output
    # else, dtype is uint
    vec_len = max([np.max(vec) for vec in input_list]) + 1
    output = np.zeros((len(input_list), vec_len), dtype=np.bool)
    for i in range(len(input_list)):
        output[i][input_list[i]] = True
    return output

def elimination_step(A: NDArray[np.bool], Y: NDArray[np.bool], pivot_row: int, pivot_col: int) -> bool:
    """
    Perform a single gauss-jordan elimination step.
    Every action taken by the following algorithm on A is also performed on Y.
    1. Find a row `r` such that `r` >= `pivot_row` and `A[r, pivot_col] == True`.
       (If no such row exists, stop and return 'False')
    2. Swap the found row `r` and `pivot_row` so that the vector in `r` is now found in index `pivot_row`
    3. For every row in A that is not `pivot_row`, XOR it with `A[pivot_row]`
    4. return 'True'
    :param A: Boolean 2D numpy matrix we are trying to reduce.
    :param Y: Boolean 2D numpy matrix on which we apply the same operations as we do on A.
    :param pivot_row: (int) row index to use as a pivot
    :param pivot_col: (int) column index to use as a pivot
    :return: 'True' if changes were made, which should tell the caller that pivot_row should be incremented.
             Otherwise 'False', which should tell the caller that pivot_row should NOT be incremented
    """
    true_in_col = np.flatnonzero(A[:, pivot_col])   # row indexes where the cell in pivot_col is true
    if true_in_col.size == 0 or true_in_col[-1] < pivot_row:
        # There are no rows equal-to-or-greater-than pivot_row in which pivot_col is True, move on
        return False
    sel_idx = np.argmax(true_in_col >= pivot_row)
    sel = true_in_col[sel_idx]
    # switch rows sel and pivot_row, if necessary
    if sel != pivot_row:
        A[[sel, pivot_row]] = A[[pivot_row, sel]]
        Y[[sel, pivot_row]] = Y[[pivot_row, sel]]
    # on every row in true_in_col other than pivot_row, apply XOR with the vector in row pivot_row (formerly in sel)
    for r in range(A.shape[0]):
        if r != pivot_row and A[r, pivot_col]:
            A[r] ^= A[pivot_row]
            Y[r] ^= Y[pivot_row]
    return True

def gauss_jordan_elimination(A_in: NDArray[np.bool], Y_in: NDArray[np.bool]) -> NDArray[np.bool]:
    """
    Solve the linear equation Ax=Y via row reduction on the augmented matrix [A|Y]
    :return: x if Ax=Y has a unique solution, otherwise raise assertion error.
    """
    # apply reduction
    A, Y = A_in.copy(), Y_in.copy()
    pivot_row = 0
    for pivot_col in range(A.shape[1]):
        if pivot_row >= A.shape[0]:
            break
        if elimination_step(A, Y, pivot_row, pivot_col):
            pivot_row += 1

    # Y should now equal x, validate result
    for row in range(A.shape[0]):
        assert np.all(A[row] == 0) and np.any(Y[row] != 0)

    # Remove all zero rows from Y before returning it
    non_zero_rows_mask = np.any(A, axis=1)
    return Y[non_zero_rows_mask]