from __future__ import annotations
import numpy as np
import copy
from Individual import Individual
from Population import Population
from sys import maxsize as MAXSIZE
from HeapqIndividual import HeapqIndividual

# constants
DEFAULT_NICHE_NUM = 4
DEFAULT_MIGRATION_INTERVAL = 50
DEFAULT_MIGRATOR_NUM = 1
NICHE_STAGNATION_LIM = MAXSIZE

class Niches:
    """A collection of Population objects acting as niches.
    Each population develops independently and once in a while niches exchange some Individuals via migration.
    Niches has the same function signatures as Population which should make them easily interchangeable."""

    def __init__(self, max_iter: int, satisfactory: float, stagnation_limit: float, stagnation_diff: float,
                 pop_size: int, win_prob: float, init_sigma: float, tournament_participants: int,
                 carry_over: int, calculate_mean: bool = False,
                 niche_num:int = DEFAULT_NICHE_NUM, migration_interval:int = DEFAULT_MIGRATION_INTERVAL,
                 migrator_num:int=DEFAULT_MIGRATOR_NUM):
        # set initial values
        self.current_iter = 0 # iteration counter
        self.rng = Individual.rng # shared random number generator
        self.max_iter = max_iter # maximum iterations
        self.migration_interval = migration_interval # every this many iterations, migrate
        self.satisfactory = satisfactory # satisfactory score
        self.calculate_mean = calculate_mean
        if calculate_mean:
            self.mean_param_std = 0
        self.stagnation_limit = stagnation_limit
        self.stagnation_diff = stagnation_diff
        # calculate per-niche size
        div, rem = divmod(pop_size, niche_num)
        niche_sizes = [div + 1 if i < rem else div for i in range(niche_num)]
        # Initialize niches
        self.niches:list[Population] = [Population(max_iter, satisfactory, NICHE_STAGNATION_LIM, stagnation_diff,
                                                   niche_size, win_prob, init_sigma, tournament_participants,
                                                   carry_over, calculate_mean)
                                        for niche_size in niche_sizes]
        self.best_individual:Individual|None = min([niche.get_best() for niche in self.niches]) # best single Individual out of all niches
        self.current_stagnant_iter = 0 # stagnant iterations (min of all niches)
        self.migrator_num = migrator_num
        self.worst_fitness_score = 0

    def get_best(self) -> Individual:
        return self.best_individual

    def best_score(self) -> float:
        return self.get_best().fitness_score

    def get_diversity(self):
        return self.mean_param_std

    def _migrate(self):
        migrators:list[list[Individual]] = []
        for niche in self.niches: # for every niche
            migrators.append(np.random.choice(niche.pop, self.migrator_num, False)) # choose migrators
            [niche.pop.remove(migrator) for migrator in migrators[-1]] # remove migrators from pop of origin
        np.roll(migrators, 1) # rotate migrators clockwise
        # each migration group settles in the pop to the right, the last group settles in the first pop
        for i in range(len(migrators)):
            self.niches[i].pop += list(migrators[i])
            # recalculate carry-overs
            self.niches[i].carry_over = HeapqIndividual(self.niches[i].num_carry_over)
            [self.niches[i].carry_over.push(ind) for ind in self.niches[i].pop]
            return

    def gen_children(self, output:Niches) -> None:
        output.niches = []
        output.best_individual = None
        output.worst_fitness_score = 0
        self.current_stagnant_iter = 0
        fitness_mean_sum = 0

        for niche in self.niches:
            output.niches.append(next(niche))
            # best individual among all niches
            best_in_new_niche = output.niches[-1].get_best()
            if best_in_new_niche < output.best_individual:
                output.best_individual = best_in_new_niche
        output.mean_param_std = np.mean([niche.mean_param_std for niche in output.niches])

    def __iter__(self):
        return NichesIterator(self)

    def __next__(self) -> Niches:
        # stop if any of the stop conditions are met
        if (self.current_iter >= self.max_iter or # max iterations reached
            self.best_score() <= self.satisfactory or # satisfactory score
            self.current_stagnant_iter >= self.stagnation_limit):
            raise StopIteration
        # if migration interval reached, migrate before generating children
        if self.current_iter % self.migration_interval == 0:
            self._migrate()
        # generate next generation
        output = copy.copy(self)
        self.gen_children(output)
        # update stagnation
        if np.abs(self.best_score() - output.best_score()) < self.stagnation_diff:
            output.current_stagnant_iter += 1
        else:
            output.current_stagnant_iter = 0
        output.current_iter += 1
        return output


class NichesIterator:
    def __init__(self, niches:Niches):
        self.niches = niches

    def __next__(self) -> Niches:
        try:
            self.niches = next(self.niches)
        except StopIteration:
            raise StopIteration
        return self.niches