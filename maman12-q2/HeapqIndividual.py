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
        if self.length == self.max_size: # if this heap is full
            if new_element > self.data[0]: # if more fit than the least fit Individual in data
                # we aspire for the closest score to 0, and we negated the fitness scores, so in this case higher is better
                heapq.heapreplace(self.data, new_element)
            # if new_element's score isn't better than that of the worst scorer in the list, discard the new value
        else:
            heapq.heappush(self.data, new_element)
            self.length += 1

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

    def get(self, index:int) -> Individual:
        if index >= self.max_size or index < (self.max_size * -1):
            raise IndexError("index" + str(index) + " out of bounds for HeapqIndividual of size " + str(self.max_size))
        return self.data[index][1]

    def __iter__(self) -> HeapqIndividualIterator:
        """Iterate over Individuals stores in this heap"""
        return HeapqIndividualIterator(self)


class HeapqIndividualIterator:
    """HeapqIndividual Iterator class.
    Iterates over the Individuals stored in order of fitness"""
    def __init__(self, hqi:HeapqIndividual):
        self.hqi:HeapqIndividual = hqi
        self.i = 0

    def __next__(self) -> Individual:
        if self.i == self.hqi.max_size:
            raise StopIteration
        self.i += 1
        return self.hqi.data[self.i - 1][1]
