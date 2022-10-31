# Applied Computational Intelligence Project 2

Group 11:
- Bernardo Silva - 93365
- Ana Sofia Guerreiro - 92620

## Code usage
The used algorithms are implemented in their own classes, `SingleObjectiveTSP`
and `MultipleObjectiveTSP`, which can be found in the `TSP_SO.py` and
`TSP_MO.py`, respectively.

When creating a class each class instance, the parameters to use for the
algorithm can be set, namely:

- `ind_size`: the size of the individuals, corresponding to the number of
  cities
- `distances`: a 2D list of the distances between each city
- `orders`: a list of size `ind_size + 1` with the orders for each city 
- `coords`: a 2D list of tuples with each city's coordinates
- `pop_size`: the number of individuals to use for the evolutionary
  algorithm
- `CXPB`: the probability of applying crossover to the offsprings
- `INVPB`: the probability of applying the inversion mutation to each
  offspring
      
Additionally, for the Single Objective problem only:
- `use_heuristic`: whether to include the route generated with the heuristic
  in the initial population
- `elitist_size`: the number of individuals with the best fitness to keep
  intact at each generation

Both classes have two main methods:
- `run_algorithm` which runs the algorithm and returns a list with the best
  fitness at each iteration, for the single objective case, or a list with
  the hypervolume at each generation, for the multiple objective case, and
  the final population for both cases.
- `many_runs` which receives as a parameter `n_runs` the number of times to
  execute the previous function. It returns a dictionary with useful
  information of the runs.

The file `TSP.py` also implements two methods that allow the visualization of
the routes.

The usage of the code is exemplified in the notebooks, which were used to obtain
the results on the report.
