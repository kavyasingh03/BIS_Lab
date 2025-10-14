import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload image in Colab
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Read the uploaded image
color_img = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), cv2.IMREAD_COLOR)
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

# Generate reference grayscale image using standard weights
ref_gray_img = 0.2989 * color_img[:, :, 0] + 0.5870 * color_img[:, :, 1] + 0.1140 * color_img[:, :, 2]

def fitness(weights, color_img, ref_gray_img):
    weights = np.abs(weights)
    weights /= np.sum(weights)
    w_r, w_g, w_b = weights
    gray_img = w_r * color_img[:, :, 0] + w_g * color_img[:, :, 1] + w_b * color_img[:, :, 2]
    mse = np.mean((gray_img - ref_gray_img) ** 2)
    return mse

num_wolves = 30
max_iter = 100

wolves = np.random.rand(num_wolves, 3)
fitness_values = np.array([fitness(w, color_img, ref_gray_img) for w in wolves])
a_linear_component = 2

def update_position(wolf, alpha, beta, delta, a):
    def get_new_pos(pos, leader_pos):
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * leader_pos - pos)
        return leader_pos - A * D

    new_pos = (
        get_new_pos(wolf, alpha) +
        get_new_pos(wolf, beta) +
        get_new_pos(wolf, delta)
    ) / 3
    return np.clip(new_pos, 0, 1)

for iter in range(max_iter):
    sorted_indices = np.argsort(fitness_values)
    alpha, beta, delta = wolves[sorted_indices[0]], wolves[sorted_indices[1]], wolves[sorted_indices[2]]

    a = a_linear_component - iter * (a_linear_component / max_iter)

    for i in range(num_wolves):
        wolves[i] = update_position(wolves[i], alpha, beta, delta, a)

    fitness_values = np.array([fitness(w, color_img, ref_gray_img) for w in wolves])
    print(f"Iteration {iter+1}/{max_iter}, Best Fitness (MSE): {fitness_values.min():.6f}")

best_idx = np.argmin(fitness_values)
best_weights = wolves[best_idx]
best_weights /= np.sum(best_weights)
print(f"Optimized weights: R={best_weights[0]:.4f}, G={best_weights[1]:.4f}, B={best_weights[2]:.4f}")

optimized_gray_img = best_weights[0] * color_img[:, :, 0] + best_weights[1] * color_img[:, :, 1] + best_weights[2] * color_img[:, :, 2]

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Original Color Image')
plt.imshow(color_img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Reference Grayscale')
plt.imshow(ref_gray_img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Optimized Grayscale')
plt.imshow(optimized_gray_img, cmap='gray')
plt.axis('off')

plt.show()
