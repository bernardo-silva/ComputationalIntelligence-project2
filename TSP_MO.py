import random
from deap import tools, base, creator
from deap.benchmarks.tools import hypervolume
from TSP import PMX, inversion
import numpy as np


class MultipleObjectiveTSP:
    def __init__(self, ind_size, distances, orders, coords, pop_size, CXPB, INVPB):
        """Creates Multi Objective TSP instance

        Parameters
        ----------
        ind_size : int
            Size of individuals. Corresponds the the number of costumers
        distances : list of lists or 2D array
            Distances between each pair of costumers.
        orders : list
            amount of orders for each costumer.
        coords : list of lists or 2D array.
            coordinates of each costumer
        pop_size : int
            population size to consider in the evolutionary algorithm.
        CXPB : float
            probability of crossover happening to a pair of individuals.
        INVPB : float
            probability of inversion happening to an individual.

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

    def _evaluate(self, individual, max_capacostumer=1000):
        """Computes the distance traveled and the cost in a route

        Parameters
        ----------
        individual : Individual
        max_capacostumer : int, optional
            Maximum truck capacostumer. The default is 1000.

        Returns
        -------
        dist : float
            total distance.
        cost : float
            total cost.

        """

        dist = self.distances[0][individual[0]]
        cost = max_capacostumer*dist
        capacostumer = max_capacostumer - self.orders[individual[0]]

        for i, f in zip(individual[:-1], individual[1:]):
            if capacostumer >= self.orders[f]:
                cost += capacostumer*self.distances[i][f]
                dist += self.distances[i][f]
            else:
                dist += self.distances[i][0] + self.distances[0][f]
                cost += capacostumer * self.distances[i][0]
                capacostumer = max_capacostumer
                cost += capacostumer * self.distances[0][f]

            capacostumer -= self.orders[f]

        cost += capacostumer * self.distances[0][individual[-1]]
        dist += self.distances[0][individual[-1]]

        return (dist, cost/1000)

    def _init_MO_EA(self):
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
        """Runs evolutionary algorithm

        Returns
        -------
        hypervolumes : list of float
            Hypervolumes at each generation.
        population : list of Individual
            Final population.

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
        n_runs : int
            number of times to run the algorithm.

        Returns
        -------
        result : dict
            Dictionary with useful information about the run algorithms.

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
