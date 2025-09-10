import cv2
import numpy as np
import random

POPULATION_SIZE = 20
GENERATIONS = 15
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.1
GENE_LENGTH = 8

image = cv2.imread('D:/1BM23CS146/download.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Verify the image path.")

histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

def calculate_fitness(thresh_val):
    thresh_val = int(thresh_val)
    weight_bg = np.sum(histogram[:thresh_val])
    weight_fg = np.sum(histogram[thresh_val:])
    if weight_bg == 0 or weight_fg == 0:
        return 0
    mean_bg = np.sum(np.linspace(0, thresh_val - 1, thresh_val) * histogram[:thresh_val]) / weight_bg
    mean_fg = np.sum(np.linspace(thresh_val, 255, 256 - thresh_val) * histogram[thresh_val:]) / weight_fg
    return weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)

def generate_individual():
    return ''.join(random.choices(['0', '1'], k=GENE_LENGTH))

def decode(chromosome):
    return int(chromosome, 2)

def select_two_parents(pop):
    contenders = random.sample(pop, 3)
    contenders.sort(key=lambda chrom: calculate_fitness(decode(chrom)), reverse=True)
    return contenders[0], contenders[1]

def apply_crossover(p1, p2):
    if random.random() < CROSSOVER_PROB:
        point = random.randint(1, GENE_LENGTH - 1)
        return p1[:point] + p2[point:]
    return p1

def apply_mutation(chrom):
    if random.random() < MUTATION_PROB:
        index = random.randint(0, GENE_LENGTH - 1)
        chrom_list = list(chrom)
        chrom_list[index] = '1' if chrom_list[index] == '0' else '0'
        return ''.join(chrom_list)
    return chrom

def run_genetic_algorithm():
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    best = max(population, key=lambda ind: calculate_fitness(decode(ind)))
    for gen in range(GENERATIONS):
        next_gen = []
        for _ in range(POPULATION_SIZE):
            parent_a, parent_b = select_two_parents(population)
            offspring = apply_crossover(parent_a, parent_b)
            offspring = apply_mutation(offspring)
            next_gen.append(offspring)
        population = next_gen
        generation_best = max(population, key=lambda ind: calculate_fitness(decode(ind)))
        if calculate_fitness(decode(generation_best)) > calculate_fitness(decode(best)):
            best = generation_best
        print(f"Generation {gen + 1}: Current Best Threshold = {decode(best)}")
    final_threshold = decode(best)
    print(f"\nOptimal threshold found: {final_threshold}")
    _, result = cv2.threshold(image, final_threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite("segmented_gea.jpg", result)
    print("Segmented image saved as 'segmented_gea.jpg'.")

if __name__ == "__main__":
    run_genetic_algorithm()
