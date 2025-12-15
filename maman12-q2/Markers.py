# from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
# from Individual import FTYPE

FTYPE = np.float64

def specificity_matrix(H_input: NDArray[FTYPE], min_mean:float, log_transform: bool = True) -> NDArray[FTYPE]:
    """
    Calculate a specificity score for every gene x celltype combination.
    The specificity score of gene g for cell-type k is H's value at (g,k) divided by the sum of values
    for gene g in all cell-types."""
    if log_transform:
        H = np.log1p(H_input)  # log(1+H)
    else:
        H = H_input

    gene_sum = H.sum(axis=1, keepdims=True) + np.finfo(FTYPE).tiny
    output = H / gene_sum # specificity matrix

    # find genes whose mean is below the minimum and set their specificity to be very small,
    # as these genes are probably noise and should only be used as a last resort
    means = np.mean(H, axis=1, keepdims=True)
    means = np.tile(means, output.shape[1])
    eps = 1e-12
    output = np.where(means < min_mean, eps, output)
    return output

def marker_genes_by_specificity(H_input:NDArray[FTYPE], k_per_cell:int = 30, min_mean:float = 1e-6,
                                log_transform:bool = True) -> NDArray[bool]:
    specif = specificity_matrix(H_input, min_mean, log_transform)
    output = np.zeros(H_input.shape[0], dtype=bool)
    for k in range(H_input.shape[1]):
        scores = specif[:, k]  # (G,)
        top_idx = np.argsort(scores)[-k_per_cell:]  # top k genes for this cell
        output[top_idx] = True
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
    gene_marker_mask = marker_genes_by_specificity(H, top_n, min_mean, log_transform)
    H, M = H[gene_marker_mask], M[gene_marker_mask]
    return H, M