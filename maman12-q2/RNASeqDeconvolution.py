from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from Individual import Individual, FTYPE
from Population import Population

class RNASeqDeconvolution:
    """This class runs the RNA-seq deconvolution genetic algorithm and produces and stores the associated results.
    In addition to the actual matrix representing the percent of cell types for a sample, these results include:
    * History of the best fitness score per iteration
    * History of the worst fitness score per iteration"""
    def __init__(self, pop:Population):
        """
        Initialize an RNA-sequence deconvolution run
        :param pop: Initialized Population object
        """
        self.best_history = [] # history of best fitness value per iteration
        self.worst_history = [] # history of worst fitness value per iteration
        self.pop = pop # Population object that implements the actual algorithm using Individual
        self.result: NDArray[FTYPE]|None = None # result at the end of the run (phenotype of pocket Individual)
        self.result_fitness_score: float|None = None
        self.confusion_matrix: NDArray[FTYPE]|None = None # confusion matrix of the result compared to true result
        self.stop_reason: str = ""

    def detect_stop_reason(self, generation:Population = None) -> str:
        if self.stop_reason != "":
            return self.stop_reason
        if generation is None:
            generation = self.pop
        if generation.current_iter >= generation.max_iter:
            self.stop_reason = "iteration limit"
        elif generation.current_stagnant_iter >= generation.stagnation_limit:
            self.stop_reason = str(generation.stagnation_limit) + "consecutive stagnant iterations"
        else:
            self.stop_reason = "satisfactory fitness score"
        return self.stop_reason

    def run(self, tqdm_leave:bool=True, tqdm_pos:int=0, print_stop_reason:bool=False) -> None:
        """Run the RNA-seq deconvolution algorithm on the given population.
        This function doesn't return anything and its results need to be extracted with the getter methods"""
        pbar = tqdm(self.pop, total=self.pop.max_iter, desc="running RNA-seq deconvolution",
                    dynamic_ncols=True, leave=tqdm_leave, position=tqdm_pos)
        for generation in pbar:
            self.best_history.append(generation.best_score())
            self.worst_history.append(generation.worst_fitness_score)
            pocket_score = generation.pocket.fitness_score
            pbar.set_postfix_str(f"best score: {self.best_history[-1]:.4f}, "
                                 f"worst score: {self.worst_history[-1]:.4f}, pocket score: {pocket_score:.4f}, "
                                 f"stagnant iterations: {generation.current_stagnant_iter}")
        self.pop = generation
        if print_stop_reason:
            print(self.detect_stop_reason(generation))
        self.result = generation.get_pocket().get_phenotype()
        self.result_fitness_score = generation.get_pocket().get_fitness_score()

