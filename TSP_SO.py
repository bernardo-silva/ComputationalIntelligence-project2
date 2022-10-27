import random
from deap import tools, base, creator
import numpy as np
from time import perf_counter


def PMX(ind1, ind2):
    ind1 -= 1
    ind2 -= 1
    tools.cxPartialyMatched(ind1, ind2)
    ind1 += 1
    ind2 += 1

    return (ind1, ind2)


def inversion(ind):
    r1 = random.randrange(len(ind)+1)
    r2 = random.randrange(len(ind)+1)

    invpoint1, invpoint2 = min(r1, r2), max(r1, r2)

    ind[invpoint1:invpoint2] = ind[invpoint1:invpoint2][::-1]
    return ind


class SingleObjectiveTSP:
    def __init__(self, ind_size, distances, orders, coords, pop_size,
                 use_heuristic, elitist_cross, elitist_size, CXPB, MUTPB,
                 INVPB):

        self.ind_size = ind_size
        self.distances = distances
        self.orders = orders
        self.coords = coords
        self.pop_size = pop_size
        self.use_heuristic = use_heuristic
        self.elitist_cross = elitist_cross
        self.elitist_size = elitist_size
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.INVPB = INVPB

    def _evaluate(self, individual, distances, orders, max_capacity=1000):

        dist = distances[0, individual[0]]
        capacity = max_capacity - orders[individual[0]]

        for i, f in zip(individual[:-1], individual[1:]):
            if capacity < orders[f]:
                dist += distances[i][0]
                capacity = max_capacity
                dist += distances[0][f]
            else:
                dist += distances[i][f]

            capacity -= orders[f]
        dist += distances[0, individual[-1]]
        return (dist,)

    def _heuristic_route(self, split=50):
        split = 50
        order = (self.coords[:, 0] > split) * - self.coords[:, 1] + \
            (self.coords[:, 0] <= split) * (self.coords[:, 1] - 1000)

        return np.argsort(order) + 1

    def _init_SO_EA(self):
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

        toolbox.register("evaluate", self._evaluate, distances=self.distances,
                         orders=self.orders)

        toolbox.register("mate",   PMX)
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        toolbox.register("invert", inversion)

        return toolbox

    def run_SO_EA(self):

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
        # means = []
        mins = []
        # maxs = []

        while g < 10_000 // self.pop_size:
            g += 1
            # print(f"----- Generation {g} -----")
            population.sort(key=lambda x: x.fitness.values[0])

            offspring = self.toolbox.select(
                population[:self.elitist_cross*self.elitist_size],
                len(population)-self.elitist_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

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

            # mean = np.mean(fits)
            # std = np.std(fits)
            # means.append(mean)
            mins.append(min(fits))
            # maxs.append(max(fits))
        return mins, population

    def many_runs(self, n_runs):
        self.toolbox = self._init_SO_EA()

        result = {}
        result["best_fitness"] = 1e6
        result["final_solutions"] = []
        result["final_fitnesses"] = []

        for _ in range(n_runs):
            mins, population = self.run_SO_EA()
            
            best = min(population, key=lambda x: x.fitness.values[0])

            if best.fitness.values[0] < result["best_fitness"]:
                result["best_solution"] = best
                result["best_fitness"] = best.fitness.values[0]
                result["best_evolution"] = mins

            result["final_solutions"] += [best]
            result["final_fitnesses"] += [best.fitness.values[0]]

        return result