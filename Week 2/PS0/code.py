import numpy as np
import cv2
from skimage import data, img_as_float
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt


def apply_gaussian_blur(image, sigmas):
    blurred = np.zeros_like(image)
    for i in range(3):  # For R, G, B
        sigma = sigmas[i]
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred_channel = cv2.GaussianBlur(image[:, :, i], (ksize, ksize), sigma)
        blurred[:, :, i] = blurred_channel
    return blurred

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for (low, high) in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_score = -np.inf

def fitness(sigmas, noisy_img, original_img):
    denoised = apply_gaussian_blur(noisy_img, sigmas)
    mse = mean_squared_error(original_img, denoised)
    return -mse  

def pso(noisy_img, original_img, bounds, num_particles=20, max_iter=30, w=0.7, c1=1.5, c2=1.5):
    swarm = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = None
    global_best_score = -np.inf

    for iter in range(max_iter):
        for particle in swarm:
            score = fitness(particle.position, noisy_img, original_img)

            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()

            if score > global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()

        for particle in swarm:
            r1, r2 = np.random.rand(3), np.random.rand(3)
            cognitive = c1 * r1 * (particle.best_position - particle.position)
            social = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive + social
            particle.position += particle.velocity

            # Clip to bounds
            for i in range(len(bounds)):
                particle.position[i] = np.clip(particle.position[i], bounds[i][0], bounds[i][1])

        if (iter + 1) % 5 == 0:
            print(f"Iteration {iter + 1}, Best MSE: {-global_best_score:.6f}")

    return global_best_position, -global_best_score


original_img = img_as_float(data.astronaut())
original_img = cv2.resize(original_img, (128, 128))

noise = 0.05 * np.random.randn(*original_img.shape)
noisy_img = np.clip(original_img + noise, 0, 1)

noisy_img = noisy_img.astype(np.float32)
original_img = original_img.astype(np.float32)

bounds = [(0.1, 3.0), (0.1, 3.0), (0.1, 3.0)]

best_sigmas, best_mse = pso(noisy_img, original_img, bounds)
print("\nBest σ values (R, G, B):", best_sigmas)
print("Best MSE:", best_mse)

denoised_img = apply_gaussian_blur(noisy_img, best_sigmas)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_img)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Denoised (PSO Optimized)")
plt.imshow(denoised_img)
plt.axis('off')
plt.tight_layout()
plt.show()
