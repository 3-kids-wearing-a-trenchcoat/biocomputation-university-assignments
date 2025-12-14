from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from numba import njit
from Markers import redact_to_marker_genes

# constants
FTYPE = np.float64

@njit
def ext_calc_fitness_score(H, self_phenotype, L, M):
    mu = (H @ self_phenotype) * L
    return np.sum(np.square(M - mu))

@njit
def ext_calc_phenotype(self_genotype, discard_unclassified: bool = True) -> NDArray[FTYPE]:
    """Compute the column-wise softmax of the genotype matrix, which is the phenotype."""
    # tiny epsilon added to denominator to avoid rare divisions by 0
    eps = np.finfo(FTYPE).tiny  # tiny epsilon added to denominator to avoid rare divisions by 0
    # subtract column max from every element, reduces the likelihood of overflow without changing softmax
    # genotype_shift = self_genotype - np.max(self_genotype, axis=0, keepdims=True)
    exp_g = np.exp(self_genotype)  # apply exp on every element (numerator)
    sum_exp = np.sum(exp_g, axis=0)  # sum columns (denominator)
    if discard_unclassified:
        return exp_g[:-1] / (sum_exp + eps)  # return without 'unclassified' row
    return exp_g / (sum_exp + eps)  # return with 'unclassified' category

@njit
def ext_apply_mutation(genotype, rng, mut_prob, mut_std):
    rows, cols = genotype.shape
    out = genotype.copy()

    row_mask = rng.random(rows) < mut_prob
    mutated_element = rng.integers(0, cols, rows)
    for i in range(rows):
        if row_mask[i]:
            out[i, mutated_element[i]] += rng.normal(0.0, mut_std)

    return out


class Individual:
    """An individual in the population representing a candidate solution.
    Each Individual is made up of a genotype, phenotype and fitness score.
    A pair of Individuals can breed, producing """
    # static variables
    rng: np.random.Generator = None # random number generator
    _mut_prob: float = None # mutation probability, between 0 and 1
    _mut_standard_deviation: float = None # standard deviation of random values we add to mimic mutation (sigma)
    _crossover_prob: float = None # probability of children being produced by crossover rather than cloning
    M: NDArray[FTYPE] = None # number of sequences for gene i in sample j
    H: NDArray[FTYPE] = None # number of sequences for gene i in cell-type j
    L: NDArray[FTYPE] = None
    TRUE: NDArray[FTYPE] = None

    @staticmethod
    def set_static_vars(rng:np.random.Generator, mut_prob:float, mut_standard_deviation:float, crossover_prob:float,
                        M:NDArray[FTYPE], H:NDArray[FTYPE], TRUE:NDArray[FTYPE]|None = None,
                        use_marker_genes:bool = True, top_n:int = 200, min_mean:float = 1e-6,
                        log_transform:bool = True) -> None:
        """
        Initialize static variables for the Individual class
        :param rng: numpy random number generator
        :param mut_prob: mutation probability for every gene. (float between 0 and 1)
        :param mut_standard_deviation: standard deviation for random (normal distribution) mutation
        :param crossover_prob: probability of offspring Individual to be the result of crossover rather than cloning
        :param M: Matrix where the cell at index (i,j) represents the number of sequences for gene i in sample j
        :param H: Matrix where the cell at index (i,j) represents the number of sequences for gene i in cell type j
        :param use_marker_genes: whether to limit H to marker genes
        :param top_n: number of marker genes (highest variance for each cell-type among genes)
        :param min_mean: treat any genes with a mean (across cell-types) lower than this as noise
        :param log_transform: whether to apply log_transform on H before choosing marker genes, reduces impact of high values.
        """
        Individual.rng = rng
        Individual._mut_prob = mut_prob
        Individual._mut_standard_deviation = mut_standard_deviation
        Individual._crossover_prob = crossover_prob
        if use_marker_genes:
            # Choose marker genes and redact H and M to just those genes
            Individual.H, Individual.M = redact_to_marker_genes(H, M, top_n, min_mean, log_transform)
        else:
            Individual.M, Individual.H = M, H
        # normalize each column of H so it becomes a per-celltype relative profile
        h_col_sum = Individual.H.sum(axis=0, keepdims=True)
         # add some miniscule value to prevent division by zero
        h_col_sum = np.where(h_col_sum == 0, np.finfo(FTYPE).tiny, h_col_sum)
        Individual.H = Individual.H / h_col_sum
        # L is the per-sample TPM "library size" (the column sums of M)
        Individual.L = Individual.M.sum(axis=0, keepdims=True)
        Individual.L = np.where(Individual.L == 0, np.finfo(FTYPE).tiny, Individual.L)

        Individual.TRUE = TRUE

    @staticmethod
    def apply_mutation(genotype: NDArray[FTYPE]) -> NDArray[FTYPE]: # per-row mutation where only one element mutates
        """Apply mutation to the given genotype by adding to it a matrix of random values.
        Each value in genotype has a probability of _mut_prob to have a value added to it (mutation probability).
        Each value to be mutated has a random, normal (gaussian) distribution value added to it.
        The normal distribution has a mean of 0 and a standard deviation of _mut_standard_deviation."""
        # rows, cols = genotype.shape
        # row_mask = (Individual.rng.random((rows,)) < Individual._mut_prob).astype('bool')
        # mutated_element = Individual.rng.integers(0, cols, rows) # pick random element for each row
        # noise = np.zeros_like(genotype)
        # noise[row_mask, mutated_element[row_mask]] = Individual.rng.normal(0, Individual._mut_standard_deviation, size=row_mask.sum())
        # return genotype + noise
        return ext_apply_mutation(genotype, Individual.rng, Individual._mut_prob, Individual._mut_standard_deviation)

    def calc_phenotype(self, discard_unclassified:bool = True) -> NDArray[FTYPE]:
        """Compute the column-wise softmax of the genotype matrix, which is the phenotype."""
        # # tiny epsilon added to denominator to avoid rare divisions by 0
        # eps = np.finfo(FTYPE).tiny # tiny epsilon added to denominator to avoid rare divisions by 0
        # # subtract column max from every element, reduces the likelihood of overflow without changing softmax
        # genotype_shift = self.genotype - np.max(self.genotype, axis=0, keepdims=True)
        # exp_g = np.exp(genotype_shift) # apply exp on every element (numerator)
        # sum_exp = np.sum(exp_g, axis=0, keepdims=True) # sum columns (denominator)
        # if discard_unclassified:
        #     return exp_g[:-1] / (sum_exp + eps) # return without 'unclassified' row
        # return exp_g / (sum_exp + eps) # return with 'unclassified' category
        return ext_calc_phenotype(self.genotype, discard_unclassified)

    def calc_fitness_score(self) -> FTYPE:
        # mu = (Individual.H @ self.phenotype) * Individual.L
        # return np.sum(np.square(Individual.M - mu))
        return ext_calc_fitness_score(Individual.H, self.phenotype,
                                      Individual.L, Individual.M)

    def __init__(self, genotype: NDArray[FTYPE], mutate:bool = True):
        """
        Initialize an Individual by giving it a genotype and automatically calculating its phenotype and fitness score.
        :param genotype: 2D float matrix
        :param mutate: Whether to apply mutation on this Individual. defaults to True.
                        Should only be set to False when Individual is not created as a child, for example
                        when creating an initial population.
        """
        if mutate:
            self.genotype = self.apply_mutation(genotype)
        else:
            self.genotype = genotype
        self.phenotype_with_unclassified = self.calc_phenotype(False)
        # self.phenotype = self.calc_phenotype()
        # self.phenotype = self.phenotype_with_unclassified[:-1]
        self.phenotype = self.phenotype_with_unclassified
        self.fitness_score = self.calc_fitness_score()

    def get_genotype(self) -> NDArray[FTYPE]:
        """get genotype matrix (given as input)"""
        return self.genotype

    def get_phenotype(self, with_unclassified:bool=False) -> NDArray[FTYPE]:
        """get phenotype matrix (calculated upon initialization)"""
        # return self.phenotype
        return self.phenotype_with_unclassified if with_unclassified else self.phenotype

    def get_fitness_score(self) -> FTYPE:
        """get fitness score (calculated upon initialization)"""
        return self.fitness_score

    def crossover(self, partner:Individual) -> tuple[NDArray[FTYPE], NDArray[FTYPE]]:
        """
        Produce two new, arithmetically symmetrical genotypes via crossover of this Individual's genotype and
        its partner's genotype.
        :param partner: Individual with which this Individual produces offsprings
        :return: two new genotypes (2D float matrices)
        """
        # generate two arithmetically symmetrical blending rate matrices (alpha) randomly in uniform distribution
        # alpha_1 = Individual.rng.uniform(0, 1, self.genotype.shape) # element-wise
        # row-wise blending
        row_alpha = Individual.rng.uniform(0,1, (self.genotype.shape[0], 1)) # choose alpha for each row
        # row_alpha = Individual.rng.uniform(0.4,0.61, (self.genotype.shape[0], 1)) # choose alpha for each row
        alpha_1 = np.tile(row_alpha, self.genotype.shape[1])
        alpha_2 = np.ones_like(alpha_1) - alpha_1
        # generate child genotypes via intermediate recombination
        child_1 = self.genotype + alpha_1 * (partner.genotype - self.genotype)
        child_2 = self.genotype + alpha_2 * (partner.genotype - self.genotype)
        # return both genotypes
        return child_1, child_2

    def breed(self, partner:Individual) -> tuple[Individual, Individual]:
        """
        Have this Individual and its partner spawn two offsprings.
        These offsprings can be either crossovers at random points of their parents or clones of their parents.
        The odds for crossover are set by _crossover_prob.
        If crossover occurs, the two offsprings will be arithmetically symmetrical, meaning that for every cell at
        index (i,j), if offspring 1's cell is influenced by parent 1's value at that cell with a weight of *a*
        and *1-a* for parent 2's value, offspring 2's weights for parent 1 and 2 would be *1-a* and *a* respectively.
        :param partner: The other Individual with which this Individual produces the offsprings
        :return: tuple of two new Individuals
        """
        # offsprings will be produced by crossover with a probability of _crossover_prob
        crossover:bool = (Individual.rng.uniform() <= Individual._crossover_prob)
        if not crossover:
            # Each offspring's genome is a mutated clone of a parents' genome
            return Individual(self.genotype.copy()), Individual(partner.genotype.copy())
        # offspring genomes are a crossover of their parents' genome
        genotype_1, genotype_2 = self.crossover(partner)
        return Individual(genotype_1), Individual(genotype_2)

    def __lt__(self, other:Individual|None) -> bool:
        if other is None:
            return True
        if not isinstance(other, Individual):
            raise NotImplemented
        return self.fitness_score < other.fitness_score

    def __gt__(self, other:Individual|None) -> bool:
        if other is None:
            return False
        if not isinstance(other, Individual):
            raise NotImplemented
        return self.fitness_score > other.fitness_score

    def __eq__(self, other:Individual|None) -> bool:
        if other is None:
            return False
        if not isinstance(other, Individual):
            return NotImplemented
        return self.fitness_score == other.fitness_score
