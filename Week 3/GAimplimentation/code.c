import cv2
import numpy as np
import random

POP_SIZE = 20
GENS = 15
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
X_MIN, X_MAX = 0, 255

img = cv2.imread('D:/1BM23CS145/BIS/download.jpg', 0)
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()

def fitness_function(threshold):
    threshold = int(threshold)
    w0 = np.sum(hist[:threshold])
    w1 = np.sum(hist[threshold:])
    if w0 == 0 or w1 == 0:
        return 0
    mu0 = np.sum(np.arange(0, threshold) * hist[:threshold]) / w0
    mu1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / w1
    return w0 * w1 * ((mu0 - mu1) ** 2)

def create_individual():
    return random.randint(X_MIN, X_MAX)

def select_parents(population):
    selected = random.sample(population, 3)
    selected.sort(key=fitness_function, reverse=True)
    return selected[0], selected[1]

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        return random.randint(min(parent1, parent2), max(parent1, parent2))
    else:
        return parent1

def mutate(individual):
    if random.random() < MUTATION_RATE:
        mutation_amount = random.randint(-5, 5)
        individual += mutation_amount
        individual = max(min(individual, X_MAX), X_MIN)
    return individual

def genetic_algorithm():
    population = [create_individual() for _ in range(POP_SIZE)]
    best_solution = max(population, key=fitness_function)
    for generation in range(GENS):
        new_population = []
        for _ in range(POP_SIZE):
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
        current_best = max(population, key=fitness_function)
        if fitness_function(current_best) > fitness_function(best_solution):
            best_solution = current_best
        print(f"Generation {generation + 1}: Best Threshold = {best_solution}")
    print(f"\nBest threshold found: {best_solution}")
    _, segmented_img = cv2.threshold(img, best_solution, 255, cv2.THRESH_BINARY)
    cv2.imwrite("segmented.jpg", segmented_img)
    print("Segmented image saved as 'segmented.jpg'.")

if __name__ == "__main__":
    genetic_algorithm()
