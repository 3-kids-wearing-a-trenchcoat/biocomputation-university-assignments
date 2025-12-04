from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import atexit
from heapq import heapify

import numpy as np
from numpy.typing import NDArray
import copy
from HeapqIndividual import HeapqIndividual as hqi
from tqdm import trange
from Individual import Individual, FTYPE

# constants
DEFAULT_POP_SIZE = 1000 # default population size
DEFAULT_WIN_PROB = 0.7 # default probability of top candidate to become a parent
DEFAULT_INIT_POP_SIGMA = 0.7 # default standard deviation for random values of initial pop (mean 0)
DEFAULT_CANDIDATE_NUM = 3 # default number of candidates in each tournament for the purpose of procreation
DEFAULT_CARRY_OVER_NUM = 2 # default number of the best candidates which will be carried over to the next generation
THREADS = 4 # number of workers in ThreadPoolExecutor
DEFAULT_MAX_ITER = 10000 # default number of iterations at which we stop regardless of other stop conditions
DEFAULT_SATISFACTORY = 1e-3 # default satisfactory fitness value, if an individual is less-than-or-equal to it, stop.
DEFAULT_STAGNATION_LIMIT = 200 # default number of iterations at which, if no significant improvement was found, stop.
DEFAULT_STAGNATION_DIFF = 1e-3 # default value of difference between iteration best-fitness-score at which we consider
                               # this generation to be "stagnant" (negligible improvement)

class Population:
    """This class represents the entire population of candidate solution.
    The class contains a list of Individual objects and handles generation of the next generation."""

    def __init__(self, max_iter:int = DEFAULT_MAX_ITER, satisfactory:float = DEFAULT_SATISFACTORY,
                 stagnation_limit:float = DEFAULT_STAGNATION_LIMIT, stagnation_diff:float = DEFAULT_STAGNATION_DIFF,
                 pop_size:int = DEFAULT_POP_SIZE, win_prob:float = DEFAULT_WIN_PROB,
                 init_sigma:float = DEFAULT_INIT_POP_SIGMA, tournament_participants:int = DEFAULT_CANDIDATE_NUM,
                 carry_over:int = DEFAULT_CARRY_OVER_NUM):
        """
        Initialize a Population object with Individuals whose values are random. (normally distributed)
        Population is assumed to be initialized AFTER Individual.Individual.set_static_vars has been called.
        :param max_iter: maximum number of iterations to run (default DEFAULT_MAX_ITER)
        :param satisfactory: if the best individual has a score less-than-or-equal to this, stop and return it.
                             (default: DEFAULT_SATISFACTORY)
        :param stagnation_limit: number of consecutive "stagnant" iterations at which we stop the run.
                                 (default: DEFAULT_STAGNATION_LIMIT)
        :param stagnation_diff: a generation whose best fitness improvement is below this value is considered "stagnant"
                                (default: DEFAULT_STAGNATION_DIFF)
        :param pop_size: total size of population (default: DEFAULT_POP_SIZE)
        :param win_prob: probability of the best current candidate in a tournament to win it. (default: DEFAULT_WIN_PROB)
        :param init_sigma: standard deviation of random values of initial population. (default: DEFAULT_INIT_POP_SIGMA)
        :param tournament_participants: number of participants in each tournament. (default: DEFAULT_CANDIDATE_NUM)
        :param carry_over: This number of the Individuals with the best fitness score will carry over to the next generation
                            (default: DEFAULT_CARRY_OVER_NUM)
        """
        self.max_iter = max_iter
        self.current_iter = 0 # iteration counter
        self.satisfactory = satisfactory # satisfactory distance from ideal at which we stop
        self.stagnation_limit = stagnation_limit # this many stagnant iterations would stop the run
        self.stagnation_diff = stagnation_diff # improvement below this value will be considered a stagnant iteration
        self.current_stagnant_iter = 0 # number of consecutive, stagnant iterations
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
        self.carry_over: list[Individual]|None = None
        # pocket Individual - the best Individual encountered so far in all iterations that have come so far
        # it is replaced by the current iteration's best whenever its fitness value is better than the current pocket
        # (I already examine the best current Individual every iteration, so this is not noticeably more expensive)
        self.pocket: Individual|None = None
        # worst fitness score in this population
        self.worst_fitness_score = 0

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

    def gen_children(self, children_num) -> tuple[list[Individual], hqi, float]:
        """
        Generate a number of children via tournament selection and record a list of the children with the best fitness
        as well as the worst fitness score encountered
        :param children_num: number of child Individual objects to spawn
        :return: a tuple of the following elements in order:
                 * list of generated children
                 * HeapqIndividual of size self.num_carry_over containing the best elements encountered thus far
                 * float equal to the worst (highest) fitness score encountered
        """
        output: list[Individual] = [] # list of all generated children
        k_best: hqi = hqi(self.num_carry_over) # sorted list of the best fitness children, from worst to best
        worst_score: float = 0 # worst fitness score encountered

        while len(output) < children_num:
            parents = self.tournament() # choose parents
            children = parents[0].breed(parents[1]) # spawn children of chosen parents
            output += children # add children to next generation
            # if either child is better than the worst in k_best, remove the worst scorer and add the child
            k_best.push(children[0])
            k_best.push(children[1])
            # check if either child has a fitness score worse than the worst recorded
            if worst_score < children[0].fitness_score:
                worst_score = children[0].fitness_score
            if worst_score < children[1].fitness_score:
                worst_score = children[1].fitness_score

        return output, k_best, worst_score

    def best_score(self) -> float:
        return self.carry_over[-1].fitness_score

    def get_pocket(self) -> Individual:
        return self.pocket

    def __iter__(self):
        """Population is its own iterator, returning the next generation of the population at each step"""
        return self

    def __next__(self) -> Population:
        """Generate the next generation of the population"""
        # stop if any of the stop conditions are met
        if (self.current_iter >= self.max_iter or  # max iterations reached
            (self.pocket is not None and self.pocket.fitness_score <= self.satisfactory) or # satisfactory score
            self.current_stagnant_iter >= self.stagnation_limit): # reached stagnant iteration limit
            raise StopIteration
        # TODO: play around with parallelization to make sure it's beneficial and if so, find a conservative value for it
        # initialize next generation
        output = copy.copy(self) # other than pop, all other attributes are identical and can be shallow copies
        output.pop = [] # start with an empty pop
        output.carry_over = [] # carry_over also needs to be reset as the old values aren't relevant
        output.worst_fitness_score = 0 # this variable refers only to this population's scores

        # generate individuals in the next generation
        children_per_worker_base, children_per_worker_rem = divmod(self.pop_size - self.num_carry_over, THREADS)
        # number of children each worker will generate, divided as equally possible
        child_quant = [children_per_worker_base + 1 if i < children_per_worker_rem
                       else children_per_worker_base
                       for i in range(THREADS)]
        # start parallel work
        futures = {self.executor.submit(self.gen_children, child_quant[i]) for i in range(THREADS)}

        # update output from each worker when they're done
        total_k_best = hqi(self.num_carry_over)
        for fut in futures:
            fut_results = fut.results()
            output.pop += fut_results[0] # generated children added to pop
            total_k_best.merge(fut_results[1]) # merge k_best heaps
            if fut_results[2] > output.worst_fitness_score: # update worst score found
                output.worst_fitness_score = fut_results[2]
        # transform total_k_best into a sorted list and store it as output's carry-overs
        output.carry_over = total_k_best.list()
        # add carry-overs (the best individuals this iteration) to next generation's pop
        output.pop += self.carry_over

        # update stop-condition-related variables in output
        output.current_iter += 1 # increase iteration counter
        pocket_candidate = output.carry_over[-1] # get the best fitness Individual in the new generation
        # check if the next generation is stagnant by checking if the diff of the best fitness scores is less
        # than or equal to the defined stagnation_diff
        if np.absolute(self.carry_over[-1].fitness_score - pocket_candidate.fitness_score) <= self.stagnation_diff:
            output.current_stagnant_iter += 1 # if so, increase stagnation counter by 1
        else:
            output.current_stagnant_iter = 0 # otherwise, this generation is not stagnant and the counter is zeroed
        # pocket
        if pocket_candidate < output.pocket: # if new best Individual is better than the pocket individual
            output.pocket = pocket_candidate # make it the new pocket Individual

        return output