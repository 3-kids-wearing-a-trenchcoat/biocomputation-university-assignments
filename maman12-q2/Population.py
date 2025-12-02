from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import atexit
import numpy as np
from numpy.typing import NDArray
import copy
import heapq
from tqdm import tqdm, trange
from Individual import Individual, FTYPE

# constants
DEFAULT_POP_SIZE = 1000 # default population size
DEFAULT_WIN_PROB = 0.7 # default probability of top candidate to become a parent
DEFAULT_INIT_POP_SIGMA = 0.7 # default standard deviation for random values of initial pop (mean 0)
DEFAULT_CANDIDATE_NUM = 3 # default number of candidates in each tournament for the purpose of procreation
DEFAULT_CARRY_OVER_NUM = 2 # default number of the best candidates which will be carried over to the next generation
THREADS = 4 # number of workers in ThreadPoolExecutor

class Population:
    """This class represents the entire population of candidate solution.
    The class contains a list of Individual objects and handles generation of the next generation."""

    def __init__(self, pop_size:int = DEFAULT_POP_SIZE, win_prob:float = DEFAULT_WIN_PROB,
                 init_sigma:float = DEFAULT_INIT_POP_SIGMA, tournament_participants:int = DEFAULT_CANDIDATE_NUM,
                 carry_over:int = DEFAULT_CARRY_OVER_NUM):
        """
        Initialize a Population object with Individuals whose values are random. (normally distributed)
        Population is assumed to be initialized AFTER Individual.Individual.set_static_vars has been called.
        :param pop_size: total size of population (default: DEFAULT_POP_SIZE)
        :param win_prob: probability of the best current candidate in a tournament to win it. (default: DEFAULT_WIN_PROB)
        :param init_sigma: standard deviation of random values of initial population. (default: DEFAULT_INIT_POP_SIGMA)
        :param tournament_participants: number of participants in each tournament. (default: DEFAULT_CANDIDATE_NUM)
        :param carry_over: This number of the Individuals with the best fitness score will carry over to the next generation
                            (default: DEFAULT_CARRY_OVER_NUM)
        """
        self.rng = Individual.rng # shared random number generator
        self.pop_size = pop_size # population size
        self.win_prob = win_prob # probability to win a tournament
        self.participants = tournament_participants # participants in each tournament
        self.num_carry_over = carry_over # number of the best candidates to carry over to next generation

        # X must be of shape (t x s), where M is of size (g x s) and H is of size (g x t)
        ind_size = (Individual.H.shape[1], Individual.M.shape[1]) # shape of candidate solution
        # the "bucket" of Individual objects that make up the population, generated randomly via normal distribution
        self.pop = [Individual(self.rng.normal(0, init_sigma, ind_size).astype(FTYPE),False)
                    for _ in trange(pop_size, desc='Generating initial population', dynamic_ncols=True)]

        # initialize ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=THREADS)
        atexit.register(self.executor.shutdown, wait=True)

        # stored carry-overs - We look for the n best Individuals twice an iteration, when finding carry-overs
        # and when finding the best current solution.
        # when either one is called we will simply find the carry-overs and store the results for when the other one
        # needs them, instead of running that search twice
        self.carry_over = None

    def tournament_selection(self, candidates:NDArray[Individual], exclude:Individual=None) -> Individual:
        """
        Select a winner from the given list of candidates.
        We do this by starting from the first element (candidates is presumed to be sorted by fitness), choosing that
        Individual with a probability of win_prob, if it is chosen we return it and if not we move on to the next
        element and do the same.
        If we reach the last element on the list it is chosen automatically.
        If we reach the last element and that last element happens to be the excluded individual, the second-to-last
        element is automatically chosen instead
        :param candidates: NDArray of candidates from which a winner is selected.
        :param exclude: An Individual that automatically loses, this optional parameter should be used when selecting
        a partner for a chosen parent, and we want to avoid matching an individual with itself.
        :return: Individual object
        """
        candidates.sort() # sort by fitness
        for candidate in candidates: # check each candidate in descending order of fitness (ascending fitness score)
            if candidate is exclude:
                continue # excluded Individual automatically loses
            if candidate is candidates[-1]: # if this is the last candidate and it's not excluded
                return candidate # auto-win
            if self.rng.uniform() <= self.win_prob: # roll the dice on whether this candidate won
                return candidate # if so, return it
        # Leaving the for loop can happen only if we reached the last candidate and it's excluded
        # in this case we return the second-to-last Individual
        # At most one individual is excluded, so we can guarantee the second-to-last member isn't excluded
        return candidates[-2]


    def tournament(self) -> tuple[Individual, Individual]:
        """
        Get a pair of parents selected via the tournament method
        :return: tuple of two Individual objects
        """
        # TODO: reconsider efficiency if I have the time
        candidates = self.rng.choice(self.pop, self.participants) # first group of candidates
        parent1 = self.tournament_selection(candidates) # pick first winner
        candidates = self.rng.choice(self.pop, self.participants) # second group of candidates
        parent2 = self.tournament_selection(candidates, parent1) # pick second winner that is not parent1
        return parent1, parent2

    def gen_children(self, children_num) -> list[Individual]:
        """
        Generate a number of children via tournament selection.
        The idea is for several calls to this function to run in parallel to generate a new population
        :param children_num: number of child Individual objects to spawn
        :return: list of Individual objects
        """
        output: list[Individual] = []
        while len(output) < children_num:
            parents = self.tournament()
            output += parents[0].breed(parents[1])
        return output

    def get_carry_overs(self) -> list[Individual]:
        """
        Get a list of the highest fitness individuals in this population, which are going to carry over
        to the next generation.
        :return: list of Individual objects of length self.num_carry_over
        """
        if self.carry_over is None:
            self.carry_over = list(heapq.nsmallest(self.num_carry_over, self.pop))
        return self.carry_over

    def get_best(self) -> Individual:
        """Get the Individual with the best fitness score in the population"""
        if self.carry_over is None:
            self.carry_over = list(heapq.nsmallest(self.num_carry_over, self.pop))
        return self.carry_over[0]

    def __iter__(self):
        """Population is its own iterator, returning the next generation of the population at each step"""
        return self

    def __next__(self) -> Population:
        """Generate the next generation of the population"""
        # TODO: implement stop conditions for maximum iterations, stagnation and sufficient fitness
        # TODO: play around with parallelization to make sure it's beneficial and if so, find a conservative value for it
        output = copy.copy(self) # other than pop, all other attributes are identical and can be shallow copies
        output.pop = [] # start with an empty pop
        output.carry_over = [] # carry_over also needs to be reset as the old values aren't relevant

        children_per_worker_base, children_per_worker_rem = divmod(self.pop_size - self.num_carry_over, THREADS)
        # number of children each worker will generate, divided as equally possible
        child_quant = [children_per_worker_base + 1 if i < children_per_worker_rem
                       else children_per_worker_base
                       for i in range(THREADS)]
        # start parallel work
        futures = {self.executor.submit(self.gen_children, child_quant[i]) for i in range(THREADS)}
        # update output from each worker when they're done
        for fut in futures:
            output.pop += fut.result()
        # add carry-overs to next pop
        output.pop += self.get_carry_overs()

        return output