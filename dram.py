import numpy as np
import matplotlib.pyplot as plt


class DRAM:
    def __init__(self, target_distribution, initial_params, iterations, adaptation_interval=200, sd=0.1, epsilon=1e-6):
        self.target_distribution = target_distribution
        self.current_params = np.array(initial_params)
        self.iterations = iterations
        self.samples = []
        self.adaptation_interval = adaptation_interval
        self.scaling_factors = [1, 0.5]  # Pre-defined scaling factors for two-stage DR
        self.epsilon = epsilon
        self.sd = sd
        self.covariance_matrix = np.eye(len(initial_params))  # Initial covariance matrix

    def propose(self, current, covariance_matrix):
        return np.random.multivariate_normal(current, covariance_matrix)

    def acceptance_probability(self, current, proposed, last_rejected=None):
        pi_current = self.target_distribution(current)
        pi_proposed = self.target_distribution(proposed)

        if last_rejected is None:
            return min(1, pi_proposed / pi_current)
        else:
            pi_last_rejected = self.target_distribution(last_rejected)
            return min(1, max(0, (pi_proposed - pi_last_rejected) / (pi_current - pi_last_rejected)))

    def draw_sample(self):
        return self.current_params

    def adapt_proposal_distribution(self):
        # Adapt the proposal distribution based on the samples so far
        if len(self.samples) > self.adaptation_interval:
            past_samples = np.array(self.samples[-self.adaptation_interval:])
            self.covariance_matrix = self.sd * np.cov(past_samples, rowvar=False) + self.epsilon * np.eye(len(self.current_params))

    def run(self):
        for iteration in range(self.iterations):
            self.adapt_proposal_distribution()  # Adapt the covariance matrix of the proposal distribution

            # First proposal
            proposed_params = self.propose(self.current_params, self.covariance_matrix)
            alpha = self.acceptance_probability(self.current_params, proposed_params)

            if np.random.rand() < alpha:
                self.current_params = proposed_params
            else:
                # First proposal was rejected, try delayed rejection
                scaled_covariance_matrix = self.scaling_factors[1] * self.covariance_matrix
                proposed_params_dr = self.propose(proposed_params, scaled_covariance_matrix)
                alpha_dr = self.acceptance_probability(self.current_params, proposed_params_dr, proposed_params)

                if np.random.rand() < alpha_dr:
                    self.current_params = proposed_params_dr

            self.samples.append(self.current_params)

        return np.array(self.samples)

if __name__ == "__main__":

    # Example target distribution: a 2D Gaussian
    def target_distribution(params):
        mean = np.array([2, 3])
        covariance = np.array([[1, 0.5], [0.5, 2]])
        inv_covariance = np.linalg.inv(covariance)
        exponent = -0.5 * np.dot((params - mean), np.dot(inv_covariance, (params - mean)))
        return np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))

    # Create an instance of DelayedRejectionMCMC
    initial_params = [0, 0]
    iterations = 500
    t0 = 100
    dram_sampler = DRAM(target_distribution, initial_params, iterations, t0)

    # Run the algorithm
    samples = dram_sampler.run()

    # The 'samples' variable now contains the samples from the Delayed Rejection MCMC run
    # Draw a single sample and print it
    single_sample = dram_sampler.draw_sample()
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
