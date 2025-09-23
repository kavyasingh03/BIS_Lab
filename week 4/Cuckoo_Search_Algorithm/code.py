import numpy as np
import math
import random

def tour_length(tour, coords):
    tour_coords = coords[tour]
    d = np.sqrt(((tour_coords - np.roll(tour_coords, -1, axis=0))**2).sum(axis=1)).sum()
    return d

def levy_flight(Lambda, size):
    sigma_u = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
               (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma_v = 1.0
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, sigma_v, size)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

def apply_levy_move(tour, coords, Lambda=1.5):
    n = len(tour)
    step = levy_flight(Lambda, n)
    magnitude = max(1, int(np.ceil(np.mean(np.abs(step)) * n / 4.0)))
    new_tour = tour.copy()
    for _ in range(magnitude):
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        if i > j:
            i, j = j, i
        if i != j:
            new_tour[i:j+1] = new_tour[i:j+1][::-1]
    return new_tour

def cuckoo_search_tsp(coords, n_nests=20, pa=0.25, iterations=10):
    num_cities = len(coords)
    nests = np.array([np.random.permutation(num_cities) for _ in range(n_nests)])
    fitness = np.array([tour_length(t, coords) for t in nests])
    best_idx = np.argmin(fitness)
    best = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    for it in range(iterations):
        for i in range(n_nests):
            new_tour = apply_levy_move(nests[i], coords)
            new_fit = tour_length(new_tour, coords)
            if new_fit < fitness[i]:
                nests[i] = new_tour
                fitness[i] = new_fit
        abandon_mask = np.random.rand(n_nests) < pa
        for i in np.where(abandon_mask)[0]:
            nests[i] = np.random.permutation(num_cities)
            fitness[i] = tour_length(nests[i], coords)
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best = nests[best_idx].copy()
        print(f"Iteration {it+1}: Best Tour Length = {best_fitness:.6f}")
    return best, best_fitness

if __name__ == "__main__":
    coords = np.array([
        [0,0],[1,5],[5,2],[6,6],[8,3],
        [2,9],[7,9],[3,4],[9,0],[4,7]
    ], dtype=float)
    best_tour, best_len = cuckoo_search_tsp(coords, n_nests=30, pa=0.25, iterations=10)
    print("\nBest tour (city indices):", best_tour)
    print("Best tour length:", best_len)
