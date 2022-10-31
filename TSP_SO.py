import random
from deap import tools, base, creator
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


class SingleObjectiveTSP:
    def __init__(self, ind_size, distances, orders, coords, pop_size,
                 use_heuristic, elitist_size, CXPB,
                 INVPB):
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
        use_heuristic : TYPE
            DESCRIPTION.
        elitist_size : TYPE
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
        self.use_heuristic = use_heuristic
        self.elitist_size = elitist_size
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

        """

        dist = self.distances[0][individual[0]]
        capacity = max_capacity - self.orders[individual[0]]

        for i, f in zip(individual[:-1], individual[1:]):
            if capacity >= self.orders[f]:
                dist += self.distances[i][f]
            else:
                dist += self.distances[i][0] + self.distances[0][f]
                capacity = max_capacity

            capacity -= self.orders[f]
        dist += self.distances[0][individual[-1]]
        return (dist,)

    def _heuristic_route(self, split=50):
        """

        Parameters
        ----------
        split : TYPE, optional
            DESCRIPTION. The default is 50.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        split = 50
        order = (self.coords[:, 0] > split) * - self.coords[:, 1] + \
            (self.coords[:, 0] <= split) * (self.coords[:, 1] - 1000)

        return np.argsort(order) + 1

    def _init_SO_EA(self):
        """

        Returns
        -------
        toolbox : TYPE
            DESCRIPTION.

        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, range(1, self.ind_size+1),
                         self.ind_size)
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         toolbox.indices)
        if self.use_heuristic:
            init_funcs = [lambda: creator.Individual(self._heuristic_route())] + \
                [toolbox.individual]*(self.pop_size-1)
            toolbox.register("population", tools.initCycle,
                             list, init_funcs, n=1)
        else:
            toolbox.register("population", tools.initRepeat,
                             list, toolbox.individual)

        toolbox.register("evaluate", self._evaluate)

        toolbox.register("mate",   PMX)
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("invert", inversion)

        return toolbox

    def run_algorithm(self):
        """

        Returns
        -------
        mins : TYPE
            DESCRIPTION.
        population : TYPE
            DESCRIPTION.

        """

        if self.toolbox is None:
            self.toolbox = self._init_SO_EA()

        if not self.use_heuristic:
            population = self.toolbox.population(n=self.pop_size)
        else:
            population = self.toolbox.population()

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Generations
        g = 0
        mins = []

        while g < 10_000 // self.pop_size:
            g += 1
            population.sort(key=lambda x: x.fitness.values[0])

            offspring = self.toolbox.select(population,
                                            len(population)-self.elitist_size)
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

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[self.elitist_size:] = offspring
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in population]

            mins.append(min(fits))
        return mins, population

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
        self.toolbox = self._init_SO_EA()

        result = {}
        result["best_fitness"] = 1e6
        result["final_solutions"] = []
        result["final_fitnesses"] = []

        for _ in range(n_runs):
            mins, population = self.run_algorithm()

            best = min(population, key=lambda x: x.fitness.values[0])

            if best.fitness.values[0] < result["best_fitness"]:
                result["best_solution"] = best
                result["best_fitness"] = best.fitness.values[0]
                result["best_evolution"] = mins

            result["final_solutions"] += [best]
            result["final_fitnesses"] += [best.fitness.values[0]]

        return result
