from __future__ import annotations
import heapq
from Individual import Individual
import copy

class HeapqIndividual:
    """a sorted heap limited to a certain number of Individuals, sorted by their fitness score"""
    def __init__(self, max_size:int):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.data = []
        self.max_size = max_size
        self.length = 0

    def push(self, ind:Individual) -> None:
        """Push an Individual into the heap.
        If the push makes this heap exceed its max_size, it will instead REPLACE the lowest-fitness element"""
        new_element = (ind.fitness_score * -1, ind)
        if self.length == self.max_size - 1: # if this heap is full
            if new_element < self.data[0]: # if more fit than the least fit Individual in data
                heapq.heapreplace(self.data, new_element)
        heapq.heappush(new_element)

    def merge(self, other: HeapqIndividual) -> HeapqIndividual:
        """merge two HeapqIndividual objects into one.
        The output will also be limited in size, specifically it will inherit the size limitation from self.
        Meaning that merging a heapqIndividual object limited to k objects with another limited to n objects
        would output a heapqIndividual object containing the k most fit Individuals in both lists"""
        output = copy.copy(self)
        merged_data = list(heapq.nlargest(self.max_size, heapq.merge(self.data, other.data)))
        output.data = merged_data
        return output

    def list(self) -> list[Individual]:
        return [self.data[i][1] for i in range(self.max_size)]