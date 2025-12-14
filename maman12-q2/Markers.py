# from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
# from Individual import FTYPE

FTYPE = np.float64

def gene_variance_and_mean(H_input: NDArray[FTYPE], log_transform:bool = True) -> tuple[NDArray[FTYPE], NDArray[FTYPE]]:
    """return two numpy vectors.
    The first representing variance of each gene (row), and the second
    representing the mean of each gene."""
    if log_transform:
        H = np.log1p(H_input) # log(1+H)
    else:
        H = H_input
    return np.var(H, axis=1), np.mean(H, axis=1)

def marker_genes_by_variance(H_input:NDArray[FTYPE], top_n:int = 200, min_mean:float = 1e-6,
                             log_transform:bool = True) -> NDArray[bool]:
    """
    Generate a boolean vector indicating which genes are marker genes.
    Marker genes are the genes whose variance across cell-types is the highest.
    :param H_input: numpy matrix representing genes x cell-type
    :param top_n: number of marker genes to select
    :param min_mean: genes with a cell-type mean lower than this value will be disregarded as noise
    :param log_transform: whether to apply log transform on H before choosing variances
                            (reduces the dominance of high absolute values)
    :return: boolean column vector where element i is True iff gene i (row i in H) is selected as a marker gene
    """
    H_var, H_mean = gene_variance_and_mean(H_input, log_transform)
    H_var = np.where(H_mean < min_mean, 0, H_var)  # remove from consideration genes whose min is below minimum
    if top_n > np.count_nonzero(H_var):  # if less than top_n values are left
        return np.where(H_var > 0, True, False).astype(np.bool)  # all non-zero variables will be marked as marker genes
    sorted_indexes = np.argsort(H_var)  # sort indexes by var in ascending order
    top_indexes = sorted_indexes[-top_n:]  # select indexes of top_n highest var values
    output = np.zeros(H_var.shape, dtype=np.bool)
    output[top_indexes] = True  # value i is true iff the var of gene i is among the top_n highest
    return output

def redact_to_marker_genes(H_input:NDArray[FTYPE], M_input:NDArray[FTYPE],
                           top_n:int = 200, min_mean:float = 1e-6,
                           log_transform:bool = True) -> tuple[NDArray[FTYPE], NDArray[FTYPE]]:
    """
    generate new H and M matrices made up of only marker genes
    :param H_input: numpy matrix representing H (genes x cell-type)
    :param M_input: numpy matrix representing M (genes x samples)
    :param top_n: number of marker genes
    :param min_mean: discard genes whose mean across cell-types is lower than this. (considered noise)
    :param log_transform: whether to apply log transformation on H. (reduces dominance of high values)
    :return: a tuple of new H and new M matrices with non-marker genes redacted
    """
    H, M = H_input.copy(), M_input.copy()
    gene_marker_mask = marker_genes_by_variance(H, top_n, min_mean, log_transform)
    H, M = H[gene_marker_mask], M[gene_marker_mask]
    return H, M