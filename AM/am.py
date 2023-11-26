import numpy as np
import matplotlib.pyplot as plt

class AdaptiveMetropolis:
    def __init__(self, target_distribution, initial_params, sd, epsilon, t0, iterations):
        self.target_distribution = target_distribution
        self.current_params = np.array(initial_params)
        self.sd = sd
        self.epsilon = epsilon
        self.t0 = t0
        self.iterations = iterations
        self.acceptance_count = 0
        self.samples = []

    def propose_new_params(self):
        if len(self.samples) < self.t0:
            # Before t0, use the initial covariance matrix C0
            covariance_matrix = np.eye(len(self.current_params))
        else:
            # After t0, use the time-dependent covariance structure
            epsilon_matrix = self.epsilon * np.eye(len(self.current_params))
            cov_matrix_t = self.sd * np.cov(np.vstack(self.samples[-self.t0:]), rowvar=False)
            covariance_matrix = cov_matrix_t + epsilon_matrix

        return np.random.multivariate_normal(self.current_params, covariance_matrix)

    def acceptance_probability(self, proposed_params):
        target_prob_current = self.target_distribution(self.current_params)
        target_prob_proposed = self.target_distribution(proposed_params)
        return min(1, target_prob_proposed / target_prob_current)

    def draw_sample(self):
        return self.current_params

    def run(self):
        for _ in range(self.iterations):
            proposed_params = self.propose_new_params()
            acceptance_prob = self.acceptance_probability(proposed_params)

            if np.random.rand() < acceptance_prob:
                self.current_params = proposed_params
                self.acceptance_count += 1

            self.samples.append(self.current_params)

        return np.array(self.samples)

def target_distribution(params):
    mean = np.array([2, 3])
    covariance = np.array([[1, 0.5], [0.5, 2]])
    inv_covariance = np.linalg.inv(covariance)
    exponent = -0.5 * np.dot(np.dot((params - mean).T, inv_covariance), (params - mean))
    return np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))

# Create an instance of AdaptiveMetropolis
initial_params = [0, 0]
sd = 0.1  # Scaling parameter
epsilon = 1.0  # Parameter for the time-dependent covariance structure
t0 = 100  # Time index for transition in covariance structure
iterations = 500
am_sampler = AdaptiveMetropolis(target_distribution, initial_params, sd, epsilon, t0, iterations)

# Run the algorithm
samples = am_sampler.run()

# Draw a single sample and print it
single_sample = am_sampler.draw_sample()
print("Single Sample:", single_sample)

# Visualize the target distribution and the obtained samples
x, y = np.meshgrid(np.linspace(-5, 8, 100), np.linspace(-5, 8, 100))
params = np.vstack([x.flatten(), y.flatten()]).T
target_probs = np.array([target_distribution(p) for p in params]).reshape(100, 100)

plt.figure(figsize=(12, 6))

# Plot the contour of the target distribution
plt.subplot(1, 2, 1)
plt.contour(x, y, target_probs, levels=20, cmap='viridis')
plt.title('Target Distribution')

# Plot the scatter plot of obtained samples
plt.subplot(1, 2, 2)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='Samples')
plt.title('Samples from Adaptive Metropolis')
plt.legend()

plt.show()
