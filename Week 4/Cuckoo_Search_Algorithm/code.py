import numpy as np
import math

def calculate_makespan(schedule, jobs, num_machines):
    machine_loads = [0] * num_machines
    for job, machine in enumerate(schedule):
        machine_loads[machine] += jobs[job]
    return max(machine_loads)

def levy_flight(Lambda, size):
    sigma_u = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
              (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma_v = 1
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, sigma_v, size)
    step = u / (abs(v) ** (1 / Lambda))
    return step

def cuckoo_search_scheduling(jobs, num_machines=2, n=15, pa=0.25, iterations=50):
    num_jobs = len(jobs)
    nests = np.random.randint(0, num_machines, (n, num_jobs))
    fitness = np.array([-calculate_makespan(nest, jobs, num_machines) for nest in nests])
    best_index = np.argmax(fitness)
    best = nests[best_index].copy()
    best_fitness = fitness[best_index]

    for t in range(iterations):
        for i in range(n):
            step_size = levy_flight(1.5, num_jobs)
            new_solution = nests[i] + step_size
            new_solution = np.where(np.random.rand(num_jobs) < 1/(1+np.exp(-new_solution)),
                                    np.random.randint(0, num_machines, num_jobs),
                                    nests[i])
            new_fitness = -calculate_makespan(new_solution, jobs, num_machines)
            if new_fitness > fitness[i]:
                nests[i] = new_solution
                fitness[i] = new_fitness

        abandon = np.random.rand(n, num_jobs) < pa
        new_nests = np.random.randint(0, num_machines, (n, num_jobs))
        nests = np.where(abandon, new_nests, nests)
        fitness = np.array([-calculate_makespan(nest, jobs, num_machines) for nest in nests])

        best_index = np.argmax(fitness)
        if fitness[best_index] > best_fitness:
            best_fitness = fitness[best_index]
            best = nests[best_index].copy()

        print(f"Iteration {t+1}: Best Makespan = {-best_fitness}")

    return best, -best_fitness

jobs = [2, 14, 4, 16, 6, 5, 3]
num_machines = 2

best_schedule, best_makespan = cuckoo_search_scheduling(jobs, num_machines)

print("\nBest Schedule (job assignment to machines):", best_schedule)
print("Best Makespan (time to finish all jobs):", best_makespan)
