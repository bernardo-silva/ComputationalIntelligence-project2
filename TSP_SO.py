import random
from deap import tools, base, creator
import numpy as np

def PMX(ind1, ind2):
    ind1 -= 1
    ind2 -= 1
    tools.cxPartialyMatched(ind1, ind2)
    ind1 += 1
    ind2 += 1

    return (ind1, ind2)


def inversion(ind):
    #invpoint1 = random.randint(0, len(ind) - 2)
    #invpoint2 = random.randint(invpoint1+1, len(ind)-1)
    r1 = random.randint(0, len(ind) - 1)
    r2 = random.randint(0, len(ind) - 1)
    
    invpoint1 , invpoint2 = min(r1,r2), max(r1,r2)
    
    ind[invpoint1:invpoint2+1]=list(reversed(ind[invpoint1:invpoint2+1]))
    
    
def evaluate(individual, distances, orders, max_capacity=1000):
    
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

def init_SO_EA(ind_size, distances, orders):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(1, ind_size+1), ind_size)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate, distances=distances, orders=orders)
    
    toolbox.register("mate",   PMX)
    #toolbox.register("select", tools.selRoulette) 
    toolbox.register("select", tools.selTournament, tournsize=4) 
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("invert", inversion)
    
    return toolbox

def run_SO_EA(distances, orders, 
              pop_size, 
              elitist_cross, elitist_size, 
              CXPB, MUTPB, INVPB,
              toolbox=None):
    
    if toolbox is None:
        toolbox = init_SO_EA(ind_size, distances, orders)
        
    population = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, population))

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Generations     
    g = 0
    # means = []
    mins = []
    # maxs = []
    
    while g < 10_000 // pop_size:
        g += 1
        # print(f"----- Generation {g} -----")
        population.sort(key=lambda x: x.fitness.values[0])
        offspring = toolbox.select(population[:elitist_cross*elitist_size], len(population)-elitist_size)
        offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for mutant in offspring:
            if random.random() < INVPB:
                toolbox.invert(mutant)
                del mutant.fitness.values
                
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        population[elitist_size:] = offspring
            # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]

        mean = np.mean(fits)
        std  = np.std(fits)
        # means.append(mean)
        mins.append(min(fits))
        # maxs.append(max(fits))
    return mins, population
        
    

def many_runs(ind_size, distances, orders, pop_size, elitist_cross, elitist_size, CXPB, MUTPB, INVPB, n_runs): 
    toolbox = init_SO_EA(ind_size, distances, orders)

    final_solutions = []
    final_fitnesses = []
    best_fitness = 1e6
    best_evolution = []
    
    for _ in range(n_runs):
        mins, population = run_SO_EA(distances, orders, 
                                      pop_size, 
                                      elitist_cross, elitist_size, 
                                      CXPB, MUTPB, INVPB,
                                      toolbox=toolbox)
        
        best = min(population, key=lambda x: x.fitness.values[0])
        
        if best.fitness.values[0] < best_fitness:
            best_solution  = best
            best_fitness   = best.fitness.values[0]
            best_evolution = mins
        
        
        final_solutions += [best]
        final_fitnesses += [best.fitness.values[0]]
        
    return (best_solution, best_fitness, best_evolution), (final_solutions, final_fitnesses)