import random

POP_SIZE = 20
GENES = 5
GENERATIONS = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

def generate_individual():
    return ''.join(random.choice(['0', '1']) for _ in range(GENES))

def decode(individual):
    return int(individual, 2)

def fitness(x):
    return x ** 2

def select_parents(population):
    selected = random.sample(population, 3)
    selected.sort(key=lambda ind: fitness(decode(ind)), reverse=True)
    return selected[0], selected[1]

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENES - 1)
        return p1[:point] + p2[point:]
    return p1

def mutate(individual):
    if random.random() < MUTATION_RATE:
        point = random.randint(0, GENES - 1)
        genes = list(individual)
        genes[point] = '1' if genes[point] == '0' else '0'
        return ''.join(genes)
    return individual

def genetic_algorithm():
    population = [generate_individual() for _ in range(POP_SIZE)]
    best = max(population, key=lambda ind: fitness(decode(ind)))

    for gen in range(GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE):
            p1, p2 = select_parents(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
        current_best = max(population, key=lambda ind: fitness(decode(ind)))
        if fitness(decode(current_best)) > fitness(decode(best)):
            best = current_best
        print(f"Generation {gen+1}: Best x = {decode(best)}, f(x) = {fitness(decode(best))}")

    print(f"\nOptimal solution: x = {decode(best)}, f(x) = {fitness(decode(best))}")

if __name__ == "__main__":
    genetic_algorithm()
