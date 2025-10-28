
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from scipy.ndimage import uniform_filter

def parallel_cellular_denoise_color_fast(noisy_img, iterations=20, alpha=0.4):
    """
    PCA for color image denoising using vectorized convolution.
    Computes average pixel change per iteration.
    """
    img = noisy_img.astype(float) / 255.0
    denoised = img.copy()

    print("\n🔹 Running Parallel Cellular Algorithm (Color Image)...")
    for it in range(iterations):

        avg_neighbors = uniform_filter(denoised, size=(3,3,1), mode='reflect')
        new_denoised = (1 - alpha) * denoised + alpha * avg_neighbors

        avg_change = np.mean(np.abs(new_denoised - denoised)) * 255
        denoised = np.clip(new_denoised, 0, 1)

        if (it + 1) % 5 == 0 or it == 0:
            print(f"   → Iteration {it + 1}/{iterations} | Avg Pixel Change: {avg_change:.2f}")

    return (denoised * 255).astype(np.uint8)


def genetic_denoise_color_fast(noisy_img, population_size=5, generations=20, mutation_rate=0.05):
    noisy = noisy_img.astype(float) / 255.0
    rows, cols, channels = noisy.shape

    population = [np.clip(noisy + np.random.normal(0, 0.05, size=noisy.shape), 0, 1)
                  for _ in range(population_size)]

    def fitness(candidate):
        diff_term = np.mean((candidate - noisy) ** 2)
        gx = np.gradient(candidate, axis=0)
        gy = np.gradient(candidate, axis=1)
        smoothness = sum(np.mean(np.abs(g)) for g in gx) + sum(np.mean(np.abs(g)) for g in gy)
        return diff_term + 0.5 * smoothness

    print("\n🔹 Running Genetic Algorithm (Color Image)...")
    for g in range(generations):
        fitness_values = [fitness(p) for p in population]
        ranked = np.argsort(fitness_values)
        best_fitness = fitness_values[ranked[0]]
        best = population[ranked[0]]

        if (g + 1) % 5 == 0 or g == 0:
            print(f"   → Generation {g + 1}/{generations} | Best Fitness: {best_fitness:.6f}")

        parents = [population[ranked[0]], population[ranked[1]]]
        new_population = []
        for _ in range(population_size):
            p1, p2 = random.choices(parents, k=2)
            mask = np.random.rand(rows, cols, channels) < 0.5
            child = np.where(mask, p1, p2)

            mutation = np.random.rand(rows, cols, channels) < mutation_rate
            child[mutation] += np.random.normal(0, 0.05, size=np.sum(mutation))
            child = np.clip(child, 0, 1)
            new_population.append(child)

        population = new_population

    return (best * 255).astype(np.uint8)


img_color = cv2.imread("image.jpg")
if img_color is None:
    raise FileNotFoundError("Image file not found. Please check the filename and path.")

img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

scale_factor = 0.5
img_color_small = cv2.resize(img_color, (0,0), fx=scale_factor, fy=scale_factor)

noisy_color = img_color_small + np.random.normal(0, 25, img_color_small.shape)
noisy_color = np.clip(noisy_color, 0, 255).astype(np.uint8)

pca_result = parallel_cellular_denoise_color_fast(noisy_color, iterations=20, alpha=0.4)
ga_result = genetic_denoise_color_fast(noisy_color, population_size=5, generations=20, mutation_rate=0.05)

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(noisy_color)
plt.title("Noisy Color Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(pca_result)
plt.title("PCA Denoised Color Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(ga_result)
plt.title("GA Denoised Color Image")
plt.axis('off')

plt.tight_layout()
plt.show()
