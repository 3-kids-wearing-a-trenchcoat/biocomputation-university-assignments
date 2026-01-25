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
    if input_list[0].dtype == np.bool:
        output = np.stack(input_list, dtype=np.bool)
        return output

    # else, dtype is uint
    vec_len = max([np.max(vec) for vec in input_list]) + 1
    output = np.zeros((len(input_list), vec_len), dtype=np.bool)
    for i in range(len(input_list)):
        output[i][input_list[i]] = True
        # indexes = input_list[i]
        # output[i][indexes] = True
    return output

def reduce_by_column(A: NDArray[np.bool], Y: NDArray[np.bool], col: int) -> None:
    """Choose the first row where the value in `col` is True in A, we'll say it's the `i`th row.
    Let j be the index of all other rows in A in which the column `col` is also True.
    Set every row j in A to be A[j] XOR A[i] and set every row j in Y to be Y[j] XOR Y[i]."""
    rows_where_col_is_true = np.flatnonzero(A[:, col])
    pivot_idx = rows_where_col_is_true[0]
    reduce_idx = rows_where_col_is_true[1:]
    pivot_row_A, pivot_row_Y = A[pivot_idx], Y[pivot_idx]
    A[reduce_idx] ^= pivot_row_A
    Y[reduce_idx] ^= pivot_row_Y

def diagonalize_identity(A: NDArray[np.bool], Y: NDArray[np.bool]) -> None:
    """Assuming A is a "disordered" identity matrix, rearrange it into an "ordered" identity
    matrix and apply the same row-swaps to Y."""
    cols = np.argmax(A, axis=1)
    perm = np.argsort(cols)
    A_orig, Y_orig = A.copy(), Y.copy()
    A[:] = A_orig[perm]
    Y[:] = Y_orig[perm]

def gauss_jordan_elimination(A_in: NDArray[np.bool], Y_in: NDArray[np.bool]) -> NDArray[np.bool] | None:
    """
    Solve the linear equation Ax=Y via row reduction on the augmented matrix [A|Y]
    :return: x if Ax=Y has a unique solution, otherwise None.
    """
    A, Y = A_in.copy(), Y_in.copy()
    # reduce by column
    [reduce_by_column(A, Y, col_idx) for col_idx in range(A.shape[1])]
    # Make sure A is now a permutation index (one True in every row of A, each True position unique)
    # That is the same as making sure each row has exactly one 'True' value and so does each column
    if not np.all(A.sum(axis=1) == 1) or not np.all(A.sum(axis=0) == 1):
        # If it doesn't pass that check, no solution
        return None
    # if it does pass the checks, diagonalize A and apply the same row swaps to Y
    diagonalize_identity(A, Y)
    # since A is the identity matrix, Ax=Y <=> x=Y, therefor we return Y
    return Y

