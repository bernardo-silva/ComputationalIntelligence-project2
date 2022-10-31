import random
from deap import tools, base, creator
from deap.benchmarks.tools import hypervolume
import numpy as np


def PMX(ind1, ind2):
    """

    Parameters
    ----------
    ind1 : TYPE
        DESCRIPTION.
    ind2 : TYPE
        DESCRIPTION.

    Returns
    -------
    ind1 : TYPE
        DESCRIPTION.
    ind2 : TYPE
        DESCRIPTION.

    """
    ind1 -= 1
    ind2 -= 1
    tools.cxPartialyMatched(ind1, ind2)
    ind1 += 1
    ind2 += 1

    return (ind1, ind2)


def inversion(ind):
    """

    Parameters
    ----------
    ind : TYPE
        DESCRIPTION.

    Returns
    -------
    ind : TYPE
        DESCRIPTION.

    """
    r1 = random.randrange(len(ind)+1)
    r2 = random.randrange(len(ind)+1)

    invpoint1, invpoint2 = min(r1, r2), max(r1, r2)

    ind[invpoint1:invpoint2] = ind[invpoint1:invpoint2][::-1]
    return ind


class MultipleObjectiveTSP:
    def __init__(self, ind_size, distances, orders, coords, pop_size, CXPB, INVPB):
        """

        Parameters
        ----------
        ind_size : TYPE
            DESCRIPTION.
        distances : TYPE
            DESCRIPTION.
        orders : TYPE
            DESCRIPTION.
        coords : TYPE
            DESCRIPTION.
        pop_size : TYPE
            DESCRIPTION.
        CXPB : TYPE
            DESCRIPTION.
        INVPB : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.ind_size = ind_size
        self.distances = distances
        self.orders = orders
        self.coords = coords
        self.pop_size = pop_size
        self.toolbox = None
        self.CXPB = CXPB
        self.INVPB = INVPB

    def _evaluate(self, individual, max_capacity=1000):
        """

        Parameters
        ----------
        individual : TYPE
            DESCRIPTION.
        max_capacity : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        dist : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """

        dist = self.distances[0][individual[0]]
        cost = max_capacity*dist
        capacity = max_capacity - self.orders[individual[0]]

        for i, f in zip(individual[:-1], individual[1:]):
            if capacity >= self.orders[f]:
                cost += capacity*self.distances[i][f]
                dist += self.distances[i][f]
            else:
                dist += self.distances[i][0] + self.distances[0][f]
                cost += capacity * self.distances[i][0]
                capacity = max_capacity
                cost += capacity * self.distances[0][f]

            capacity -= self.orders[f]

        cost += capacity * self.distances[0][individual[-1]]
        dist += self.distances[0][individual[-1]]

        return (dist, cost/1000)

    def _init_MO_EA(self):
        """

        Returns
        -------
        toolbox : TYPE
            DESCRIPTION.

        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, range(1, self.ind_size+1),
                         self.ind_size)
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         toolbox.indices)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._evaluate)

        toolbox.register("mate",   PMX)
        toolbox.register("select", tools.selNSGA2)
        # toolbox.register("mutate", my_mute, distances=self.distances)
        # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        toolbox.register("invert", inversion)

        return toolbox

    def run_algorithm(self):
        """

        Returns
        -------
        hypervolumes : TYPE
            DESCRIPTION.
        population : TYPE
            DESCRIPTION.

        """

        if self.toolbox is None:
            self.toolbox = self._init_MO_EA()

        population = self.toolbox.population(n=self.pop_size)

        population = self.toolbox.select(population, self.pop_size)

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Generations
        g = 0

        # mins = []
        listfits = []
        hypervolumes = []

        while g < 10_000 // self.pop_size:
            g += 1
            # print(f"----- Generation {g} -----")

            offspring = tools.selTournamentDCD(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.INVPB:
                    self.toolbox.invert(mutant)
                    del mutant.fitness.values

            # Evaluate the individualrun_algorithm with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population = self.toolbox.select(population + offspring,
                                             len(population))
            # Gather all the fitnesses in one list and print the stats

            fits = [ind.fitness.values[0] for ind in population]
            listfits += [fits]
            hypervolumes += [hypervolume(population, [2000, 2000])]

        return hypervolumes, population

    def many_runs(self, n_runs):
        """

        Parameters
        ----------
        n_runs : TYPE
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        if self.toolbox is None:
            self.toolbox = self._init_MO_EA()

        result = {}
        result["best_hv"] = -1
        result["final_solutions"] = []
        result["final_fitnesses"] = []

        for _ in range(n_runs):
            hypervolumes, population = self.run_algorithm()

            if hypervolumes[-1] > result["best_hv"]:
                result["best_solution"] = population
                result["best_hv"] = hypervolumes[-1]
                result["best_evolution"] = hypervolumes

            result["final_solutions"] += [population]
            result["final_fitnesses"] += [hypervolumes[-1]]

        return result
