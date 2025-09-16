import numpy as np
import random

class AntColony:
    def __init__(self, coords, n_ants, n_iterations, alpha=1.0, beta=5.0, rho=0.5, initial_pheromone=1.0):
        self.coords = coords
        self.n_cities = len(coords)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.initial_pheromone = initial_pheromone
        
        self.dist_matrix = self._calculate_distances()
        self.pheromone = np.ones((self.n_cities, self.n_cities)) * self.initial_pheromone
        
        with np.errstate(divide='ignore'):
            self.heuristic = 1 / (self.dist_matrix + np.eye(self.n_cities))
        self.heuristic[self.heuristic == np.inf] = 0
        
        self.best_cost = float('inf')
        self.best_solution = None

    def _calculate_distances(self):
        dist_matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i+1, self.n_cities):
                dist = np.linalg.norm(np.array(self.coords[i]) - np.array(self.coords[j]))
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        return dist_matrix

    def _choose_next_city(self, current_city, visited):
        pheromone = self.pheromone[current_city]
        heuristic = self.heuristic[current_city]

        prob_num = np.zeros(self.n_cities)
        for city in range(self.n_cities):
            if city not in visited:
                prob_num[city] = (pheromone[city] ** self.alpha) * (heuristic[city] ** self.beta)

        prob_den = np.sum(prob_num)
        if prob_den == 0:
            candidates = [city for city in range(self.n_cities) if city not in visited]
            return random.choice(candidates)

        probabilities = prob_num / prob_den
        next_city = np.random.choice(range(self.n_cities), p=probabilities)
        return next_city

    def _construct_solution(self):
        solution = []
        start_city = random.randint(0, self.n_cities - 1)
        solution.append(start_city)

        while len(solution) < self.n_cities:
            current_city = solution[-1]
            next_city = self._choose_next_city(current_city, solution)
            solution.append(next_city)

        return solution

    def _calculate_cost(self, solution):
        cost = 0.0
        for i in range(len(solution) - 1):
            cost += self.dist_matrix[solution[i]][solution[i+1]]
        cost += self.dist_matrix[solution[-1]][solution[0]]
        return cost

    def _update_pheromone(self, solutions, costs):
        self.pheromone = (1 - self.rho) * self.pheromone

        for solution, cost in zip(solutions, costs):
            deposit_amount = 1.0 / cost
            for i in range(len(solution) - 1):
                self.pheromone[solution[i]][solution[i+1]] += deposit_amount
                self.pheromone[solution[i+1]][solution[i]] += deposit_amount
            self.pheromone[solution[-1]][solution[0]] += deposit_amount
            self.pheromone[solution[0]][solution[-1]] += deposit_amount

    def run(self):
        for iteration in range(self.n_iterations):
            all_solutions = []
            all_costs = []
            for _ in range(self.n_ants):
                sol = self._construct_solution()
                cost = self._calculate_cost(sol)
                all_solutions.append(sol)
                all_costs.append(cost)

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = sol

            self._update_pheromone(all_solutions, all_costs)
            print(f"Iteration {iteration + 1}/{self.n_iterations} - Best cost so far: {self.best_cost:.4f}")

        return self.best_solution, self.best_cost

cities = [
    (0, 0),
    (1, 5),
    (5, 2),
    (6, 6),
    (8, 3),
    (7, 7),
    (2, 8),
    (3, 3)
]

aco = AntColony(coords=cities, n_ants=10, n_iterations=5, alpha=1.0, beta=5.0, rho=0.5, initial_pheromone=1.0)
best_solution, best_cost = aco.run()

print("\nBest route found:")
print(best_solution)
print(f"Cost of the best route: {best_cost:.4f}")
