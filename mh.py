import numpy as np

class MetropolisHastings:
    def __init__(self, target_distribution, initial_params, iterations):
        self.target_distribution = target_distribution
        self.current_params = np.array(initial_params)
        self.iterations = iterations
        self.samples = []
        self.covariance_matrix = np.eye(len(initial_params))  # Proposal covariance matrix

    def propose(self, current):
        return np.random.multivariate_normal(current, self.covariance_matrix)

    def acceptance_probability(self, current, proposed):
        pi_current = self.target_distribution(current)
        pi_proposed = self.target_distribution(proposed)
        return min(1, pi_proposed / pi_current)

    def draw_sample(self):
        return self.current_params

    def run(self):
        for iteration in range(self.iterations):
            proposed_params = self.propose(self.current_params)
            alpha = self.acceptance_probability(self.current_params, proposed_params)

            if np.random.rand() < alpha:
                self.current_params = proposed_params

            self.samples.append(self.current_params)

        return np.array(self.samples)

# Example target distribution: a 2D Gaussian
def target_distribution(params):
    mean = np.array([2, 3])
    covariance = np.array([[5, 1], [0, 1]])
    inv_covariance = np.linalg.inv(covariance)
    exponent = -0.5 * np.dot((params - mean), np.dot(inv_covariance, (params - mean)))
    return np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))

# Create an instance of MetroPolis Hastings
initial_params = [0, 0]
iterations = 500
mh_sampler = MetropolisHastings(target_distribution, initial_params, iterations)

# Run the algorithm
samples = mh_sampler.run()

# The 'samples' variable now contains the samples from the Delayed Rejection MCMC run
# Draw a single sample and print it
single_sample = mh_sampler.draw_sample()
print("Single Sample:", single_sample)

# # Visualize the target distribution and the obtained samples
x, y = np.meshgrid(np.linspace(-5, 8, 100), np.linspace(-5, 8, 100))
params_grid = np.vstack([x.ravel(), y.ravel()]).T
target_probs = np.array([target_distribution(p) for p in params_grid]).reshape(x.shape)

# Create the plot
plt.figure(figsize=(12, 6))

# Combine the contour and scatter plot
plt.contour(x, y, target_probs, levels=20, cmap='viridis')  # Contour plot for the target distribution
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='Samples', color='orange')  # Scatter plot for the samples
plt.title('Target Distribution and Samples')
plt.legend()
plt.grid(True)

plt.show()
