import timeit
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(7)
pts = 5

class Differential_Evolution:
    def __init__(self, particles, generations, fitness_function):
        self.particles = particles.copy()
        self.n = particles.shape[0]   # number of particles
        self.m = particles.shape[1]   # number of variables
        self.g = generations
        self.f = fitness_function
        self.eps = 0.2              # mutation constant
        
    def compute_diff_vecs(self):
        """
        Computes the difference vectors between all pairs of particles in the population
        Parameters
        ----------
        population : numpy.ndarray
            Population of particles
        Returns
        -------
        numpy.ndarray   
            Difference vectors between all pairs of particles
        """
        difference_vectors = np.zeros((nCr(self.n, 2),self.m))
        itr = 0
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                difference_vectors[itr] = self.particles[i,:] - self.particles[j,:]
                itr += 1
        return difference_vectors
    
    def generate_mutant(self):
        """
        Generates a mutant vector
        Parameters
        ----------
        population : numpy.ndarray
            Population of particles
        Returns
        -------
        numpy.ndarray
            Mutant vector
        """
        diff_vecs = self.compute_diff_vecs()
        mutant_vectors = np.zeros_like(self.particles)
        # F = np.random.rand()*np.random.choice([-1,1])
        # F = F*Lc(self.d_volume, 40, 0.00000001, 1)
        for i, vec in enumerate(self.particles):
            F = 15*np.random.rand()*np.random.choice([-1,1])
            chosen_vec = diff_vecs[np.random.choice(np.arange(0, self.n))]
            mutant_vectors[i] = vec + F*chosen_vec + self.eps*np.random.rand(self.m)
        return mutant_vectors
    
    def crossover(self, mutant_vectors):
        """
        Performs crossover between the population and mutant vectors
        Parameters
        ----------
        population : numpy.ndarray
            Population of particles
        mutant_vectors : numpy.ndarray
            Mutant vectors
        Returns
        -------
        numpy.ndarray
            Trial vectors
        """
        trial_vectors = np.zeros_like(self.particles)
        for i, vec in enumerate(self.particles):
            trial_vectors[i,0] = vec[0] + np.random.rand()*(mutant_vectors[i,0] - vec[0])
            trial_vectors[i,1] = vec[1] + np.random.rand()*(mutant_vectors[i,1] - vec[1])
        return trial_vectors
    
    def select(self, trial_vectors):
        """
        Selects the next generation of particles
        Parameters
        ----------
        population : numpy.ndarray
            Population of particles
        trial_vectors : numpy.ndarray
            Trial vectors
        Returns
        -------
        numpy.ndarray
            Next generation of particles
        """
        selection_pool = np.concatenate((self.particles, trial_vectors))
        pool_fitness = self.f(selection_pool)
        tournament_size = 15
        selection = np.zeros_like(self.particles)

        # Tournament selection
        for i in range (int(len(self.particles))):
            # print(f'shape of selection pool and fitness: {selection_pool.shape}, {pool_fitness.shape}')
            match = np.random.choice(np.arange(0,int(selection_pool.shape[0])), tournament_size, replace=False)
            match_fitness = [pool_fitness[j] for j in match]
            winner = match[np.argmax(match_fitness)]
            selection[i] = selection_pool[winner]
            selection_pool = np.delete(selection_pool, winner,axis=0)
            pool_fitness = np.delete(pool_fitness, winner)
        return selection
    
    def converged(self, i):
        """
        Checks if the algorithm has converged or not
        Parameters
        ----------
        i : int
            Current generation
        Returns
        -------
        bool
            True if the algorithm should terminate, False otherwise
        """
        if i > 10 and np.std(self.particles, axis=0).max() < 0.5:
            return True
        else:
            return False
        
    def run(self, verbose=False):
        tic = timeit.default_timer()

        # make list to keep track of volume of convex hull
        volume = 1
        self.volumes = np.zeros((self.g,2))
        
        # EWMA stuff
        rho = 0.95 # Rho value for smoothing
        volume_prev = 1 # Initial value ewma value

        print(f'Beginning DE with {self.g} generations\n')
        for i in range(self.g):
            toc = timeit.default_timer()

            if i !=0:
                try:
                    hull = ConvexHull(self.particles)
                    # use ewma to store convex hull volume
                    volume = hull.volume
                except:
                    volume = 0
                self.d_volume = volume_cur_bc - self.volumes[i-1,1]
            else:
                self.d_volume = 0

            # use ewma to store volume
            volume_cur_bc, volume_prev = ewma(volume_prev, volume, rho, i)
            self.volumes[i] = np.array([volume, volume_cur_bc])
            

            # if i == 0:
            # else:

            mutant_vectors = self.generate_mutant()
            trial_vectors = self.crossover(mutant_vectors)
            self.particles = self.select(trial_vectors)

            self.fitnesses = self.f(self.particles)
            self.best = max(self.fitnesses)
            self.best_individual = self.particles[np.argmax(self.fitnesses)]

            converged = self.converged(i)

            if i %5 == 0 and verbose:
                print(f'\n\n\n\n{round(toc-tic,3)}s: Generation {i}, number of particles: {self.particles.shape[0]} (all std: {np.std(self.particles, axis=0)})')
                print(f'      Convex hull volume: {np.array([volume])}')
                print(f'      change in volume: {self.d_volume}')
                print(f'      best fitness: {self.best}')
                print(f'      worst fitness: {self.fitnesses.min()}')
                print(f'      all fitnesses: {self.fitnesses}')
                print(f'      best particle: {self.best_individual}')
            if converged:
                print(f'\n\nEvolution terminated, termanation criteria met at generation {i}')
                itr_taken = i
                break
                

        if not converged:
            itr_taken = self.g
            print(f'\n\nEvolution terminated, maximum number of generations reached')

        print(f'Best fitness: {self.best}')
        if verbose:
            print(f'\n\n\n\n*****************************************************\n\n\n\n')
            print(f'Total time taken: {round(toc-tic,3)}s')
            print(f'Number of iterations taken: {itr_taken}')
            print(f'Best fitness: {self.fitnesses.max()}')

        return converged, itr_taken, self.particles



class Individual:
    def __init__(self, genotype, strategy_parameters):
        self.genotype = genotype
        self.fitness = None
        self.strategy_parameters = strategy_parameters

    @staticmethod
    def initializeWithGenotype(genotype: np.array):
        individual = Individual(len(genotype))
        individual.genotype = genotype.copy()
        return individual
    
    @staticmethod
    def initializeUniformAtRandom(genotype_length):
        individual = Individual(genotype_length)
        individual.genotype = np.random.choice((0,1), p=(0.5, 0.5), size=genotype_length)
        return individual
    

class ES:
    def __init__(self, fitness_function=lambda x: 0, num_dimensions=3, 
                 num_generations=200, num_individuals=80, 
                 num_offspring_per_individual=5, verbose=False):
        self.fitness_function = fitness_function
        self.num_dimensions = num_dimensions
        self.num_generations = num_generations
        self.num_individuals = num_individuals
        self.num_offspring_per_individual = num_offspring_per_individual
        self.verbose = verbose

        assert num_individuals % 2 == 0, "Population size needs to be divisible by 2 for cross-over"
    
        self.terminate_confidence = 0

    def terminate(self):
        if self.itr_best_fit <= self.group_best_fit:
            self.terminate_confidence += 1
        else:
            self.terminate_confidence = 0

        if self.terminate_confidence > 12:
            return True
        else:
            return False
        
    def run(self):
        population = [self.generate_random_individual() for _ in range(self.num_individuals)]
        best = sorted(population, key=lambda individual: self.fitness_function(individual.genotype))[0]

        self.group_best_fit = self.fitness_function(best.genotype)
        self.itr_best_fit = self.group_best_fit
        tic = timeit.default_timer()
        for generation in range(self.num_generations):
            toc = timeit.default_timer()
            # --- Perform mutation and selection here ---
            # - Each parent individual should produce `num_offspring_per_individual` children by mutation
            #   (recombination is ignored for this exercise)
            # - Implement P+O (parent+offspring) with truncation selection (picking the best n individuals)
            # - Update the `best` variable to hold the best individual of this generation (to then be printed below)
            

            offsprings = []
            for parent in population:
                for _ in range(self.num_offspring_per_individual):
                    parent_genotype = parent.genotype
                    parent_strategy_parameter = parent.strategy_parameters[0]
                    new_genotype = np.array([parent_genotype[i]+np.random.normal(0,parent_strategy_parameter) for i in range(self.num_dimensions)])
                    new_strategy_parameters = np.array([max(parent_strategy_parameter*np.exp(np.random.normal(0,1/self.num_dimensions)),10**(-6))])
                    offsprings.append(Individual(new_genotype, new_strategy_parameters))
            population += offsprings
            # population = sorted(population, key=lambda individual: self.fitness_function(individual.genotype))[:self.num_individuals]

            selection = []
            selection_pool = population.copy()  
            pool_fitness = np.array([self.fitness_function(individual.genotype) for individual in population])
            tournament_size = 17
            for i in range(self.num_individuals):
                # print(f'shape of selection pool and fitness: {selection_pool.shape}, {pool_fitness.shape}')
                match = np.random.choice(np.arange(0,int(len(selection_pool))), tournament_size, replace=False)
                match_fitness = [pool_fitness[j] for j in match]
                winner = match[np.argmin(match_fitness)]
                selection.append(selection_pool[winner])
                selection_pool = np.delete(selection_pool, winner,axis=0)
                pool_fitness = np.delete(pool_fitness, winner)

            population = selection

            all_fitnesses = np.array([self.fitness_function(individual.genotype) for individual in population])

            best = population[np.argmin(all_fitnesses)]
            self.itr_best_fit = self.fitness_function(best.genotype)
            self.group_best_fit = self.itr_best_fit if self.itr_best_fit < self.group_best_fit else self.group_best_fit

            if self.terminate():
                break
            if self.verbose:
                print(f"[gen {generation:3}] Best fitness: {self.fitness_function(best.genotype)}")
            # if generation%10==0:
            #     print(f'{round(toc-tic,6)} s tour size{tournament_size}\nself.itr_best_fit: {self.itr_best_fit}\nself.group_best_fit: {self.group_best_fit}\n\n')

        return best.genotype
    
    def generate_random_individual(self):
        # --- Initialize the population here ---
        # - For the genotype, sample a standard random normal distribution for each variable separately
        # - For the strategy parameter, sample a standard random normal distribution and then take the maximum of that sample and 0.1 
        #   (to ensure it is not negative and not too small for exploration)
        return Individual(np.array([np.random.normal(0,1) for _ in range(self.num_dimensions)]), np.array([max(np.random.normal(0,3), 0.1) for _ in range(self.num_dimensions)]))


def ewma(s_prev, x, rho, i):
    """
    Computes exponentially weighted moving average
    Parameters
    ----------
    s_prev : float
        Previous smoothed value
    x : float
        Current value
    rho : float
        Smoothing factor
    i : int
        Current iteration
    Returns
    -------
    s_cur_bc: float
        Smoothed value bias corrected
    s_cur: float
        Smoothed value
    """
    s_cur = rho*s_prev + (1-rho)*x
    s_cur_bc = s_cur/(1-(rho**(i+1)))
    return s_cur_bc, s_cur


def nCr(n,r):
    """
    Computes the number of combinations of n things taken r at a time
    Parameters
    ----------
    n : int
        Number of things
    r : int
        Number of things taken at a time
    Returns
    -------
    int
        Number of combinations
        """
    return int(np.math.factorial(n)/(np.math.factorial(r)*np.math.factorial(n-r)))


def Lc(x, L, m, c):
    """
    Computes the value of the logistic curve at x
    Parameters
    ----------
    x : float
        x-coordinate
    L : float
        Maximum value of the curve
    m : float
        Slope of the curve
    c : float
        Minimum value of the curve
    Returns
    -------
    float
        Value of the logistic curve at x
    """
    return L/(1 + np.exp((-m*x))) + c

