from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import copy

# constants
FTYPE = np.float32

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

    @staticmethod
    def set_static_vars(rng:np.random.Generator, mut_prob:float, mut_standard_deviation:float, crossover_prob:float,
                        M:NDArray[FTYPE], H:NDArray[FTYPE]) -> None:
        """
        Initialize static variables for the Individual class
        :param rng: numpy random number generator
        :param mut_prob: mutation probability for every gene. (float between 0 and 1)
        :param mut_standard_deviation: standard deviation for random (normal distribution) mutation
        :param crossover_prob: probability of offspring Individual to be the result of crossover rather than cloning
        :param M: Matrix where the cell at index (i,j) represents the number of sequences for gene i in sample j
        :param H: Matrix where the cell at index (i,j) represents the number of sequences for gene i in cell type j
        """
        Individual.rng = rng
        Individual._mut_prob = mut_prob
        Individual._mut_standard_deviation = mut_standard_deviation
        Individual._crossover_prob = crossover_prob
        Individual.M = M
        Individual.H = H

    # @staticmethod
    # def apply_mutation(genotype: NDArray[FTYPE]) -> NDArray[FTYPE]: # per-element mutation
    #     """Apply mutation to the given genotype by adding to it a matrix of random values.
    #     Each value in genotype has a probability of _mut_prob to have a value added to it (mutation probability).
    #     Each value to be mutated has a random, normal (gaussian) distribution value added to it.
    #     The normal distribution has a mean of 0 and a standard deviation of _mut_standard_deviation."""
    #     # "roll the dice" for every value in genotype, any value above _mut_prob will mutate
    #     # The choice of which cells to mutate is uniformly distributed, i.e. every cell has the same chance of mutating
    #     # that is independent of whether other cells mutated
    #     mut_dice_roll = Individual.rng.uniform(0, 1, genotype.shape)
    #     mut_mask = np.where(mut_dice_roll < Individual._mut_prob, True, False).astype(np.bool)
    #     # apply mutation in cells that have been chosen for mutation
    #     # each cell's mutation is random and normally distributed
    #     mutation = np.zeros_like(genotype)
    #     mutation[mut_mask] = Individual.rng.normal(0, Individual._mut_standard_deviation, mutation[mut_mask].shape)
    #     # add mutation to genotype and return
    #     return genotype + mutation

    # @staticmethod
    # def apply_mutation(genotype: NDArray[FTYPE]) -> NDArray[FTYPE]: # per-row mutation
    #     """Apply mutation to the given genotype by adding to it a matrix of random values.
    #     Each value in genotype has a probability of _mut_prob to have a value added to it (mutation probability).
    #     Each value to be mutated has a random, normal (gaussian) distribution value added to it.
    #     The normal distribution has a mean of 0 and a standard deviation of _mut_standard_deviation."""
    #     rows, cols = genotype.shape
    #     row_mask = Individual.rng.random((rows,)) < Individual._mut_prob
    #     noise = Individual.rng.normal(0, Individual._mut_standard_deviation, size=genotype.shape)
    #     noise[~row_mask.astype('bool')] = 0
    #     return genotype + noise

    @staticmethod
    def apply_mutation(genotype: NDArray[FTYPE]) -> NDArray[FTYPE]: # per-row mutation where only one element mutates
        """Apply mutation to the given genotype by adding to it a matrix of random values.
        Each value in genotype has a probability of _mut_prob to have a value added to it (mutation probability).
        Each value to be mutated has a random, normal (gaussian) distribution value added to it.
        The normal distribution has a mean of 0 and a standard deviation of _mut_standard_deviation."""
        rows, cols = genotype.shape
        row_mask = (Individual.rng.random((rows,)) < Individual._mut_prob).astype('bool')
        mutated_element = Individual.rng.integers(0, cols, rows) # pick random element for each row
        noise = np.zeros_like(genotype)
        noise[row_mask, mutated_element[row_mask]] = Individual.rng.normal(0, Individual._mut_standard_deviation)
        return genotype + noise

    def calc_phenotype(self, discard_unclassified:bool = True) -> NDArray[FTYPE]:
        """Compute the column-wise softmax of the genotype matrix, which is the phenotype."""
        # tiny epsilon added to denominator to avoid rare divisions by 0
        eps = np.finfo(FTYPE).tiny # tiny epsilon added to denominator to avoid rare divisions by 0
        # subtract column max from every element, reduces the likelihood of overflow without changing softmax
        genotype_shift = self.genotype - np.max(self.genotype, axis=0, keepdims=True)
        exp_g = np.exp(genotype_shift) # apply exp on every element (numerator)
        sum_exp = np.sum(exp_g, axis=0, keepdims=True) # sum columns (denominator)
        if discard_unclassified:
            return exp_g[:-1] / (sum_exp + eps) # return without 'unclassified' row
        return exp_g / (sum_exp + eps) # return with 'unclassified' category

    def calc_fitness_score(self) -> FTYPE:
        """Return the Residual Sum of Squares (RSS) of the phenotype, specifically RSS(X)=||M-HX||^2
        **M** - number of sequences for gene i in sample j
        **H** - number of sequences for gene i in cell type j
        **X** - Individual phenotype, the candidate solution"""
        # return np.square(np.linalg.norm(Individual.M - Individual.H.dot(self.phenotype)))
        return np.linalg.norm(Individual.M - Individual.H.dot(self.phenotype))

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
        self.phenotype = self.phenotype_with_unclassified[:-1]
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
