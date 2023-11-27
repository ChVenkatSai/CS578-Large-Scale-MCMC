import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Standard normal distribution as target distribution function
def target_distribution(x):
    return stats.norm.pdf(x, loc=0, scale=1)

#def target_distribution(x, theta):
    #return stats.norm.pdf(x, loc=theta, scale=1)

# Normal distribution centered at the current sample as proposal distribution function
def proposal_distribution(theta, sigma=0.5):
    return np.random.normal(theta, sigma)


# Function to compute correction factor
def compute_correction_factor(num_points, sigma):
    def objective(C_sigma):
        convolved = np.convolve(stats.norm.cdf(np.linspace(-3*sigma, 3*sigma, num_points)/sigma), C_sigma, mode='same')
        S = 1 / (1 + np.exp(-np.linspace(-3*sigma, 3*sigma, num_points)))
        return np.sum(np.abs(convolved - S))

    initial_guess = np.ones(num_points) / num_points
    result = minimize(objective, initial_guess, method='SLSQP')
    return np.mean(result.x) if result.success else 1.0

# Function to create correction distribution
def create_correction_distribution_function(average_correction_factor, scale=1):
    def correction_distribution():
        return np.random.normal(average_correction_factor, scale)
    return correction_distribution

# Function to compute statistics (Δ*, sample variance, error)
def compute_statistics(current, proposed, minibatch, target_distribution):
    N = len(minibatch)
    lambda_values = np.array([np.log(target_distribution(proposed) / target_distribution(current)) for _ in minibatch])
    delta_star = np.mean(lambda_values)
    sample_variance = np.var(lambda_values)
    
    # First and third absolute moments of |Lambda_i - Lambda|
    first_moment = np.mean(np.abs(lambda_values - delta_star))
    third_moment = np.mean(np.abs(lambda_values - delta_star)**3)
    
    # Error estimate from Corollary 1
    error = (6.4 * third_moment / np.sqrt(N)) + (2 * first_moment / N)
    return delta_star, sample_variance, error


# Function to compute error estimate based on Corollary 1
def compute_error_estimate(first_moment, third_moment, minibatch_size):
    error_estimate = (6.4 * third_moment / np.sqrt(minibatch_size)) + (2 * first_moment / minibatch_size)
    return error_estimate

# Minibatch acceptance test function
def minibatch_acceptance_test(current, proposed, minibatch, target_distribution, correction_distribution, delta_threshold, error_tolerance):
    global final_minibatch_size  # Use a global variable to track the final minibatch size
    delta_star, sample_variance, error = compute_statistics(current, proposed, minibatch, target_distribution)
    while sample_variance >= 1 or error > error_tolerance:
        additional_samples = np.random.choice(data, m, replace=False)
        minibatch = np.concatenate((minibatch, additional_samples))
        delta_star, sample_variance, error = compute_statistics(current, proposed, minibatch, target_distribution)
        
    final_minibatch_size = len(minibatch)  # Update the final minibatch size
    X_nc = np.random.normal(0, np.sqrt(1 - sample_variance))
    X_corr = correction_distribution()
    return delta_star + X_nc + X_corr > 0


# MCMC sampling with minibatches
def mcmc_with_minibatch(T, m, delta, correction_distribution, initial_sample, proposal_distribution, target_distribution, data):
    samples = [initial_sample]
    theta = initial_sample

    for _ in range(T):
        theta_prime = proposal_distribution(theta)
        minibatch = np.random.choice(data, m, replace=False)
        if minibatch_acceptance_test(theta, theta_prime, minibatch, target_distribution, correction_distribution, 1, delta):
            theta = theta_prime
        else:
            theta = theta  # theta_t+1 = theta_t

        samples.append(theta)

    return samples

# Initialize a global variable to track the final minibatch size
final_minibatch_size = 0

# Running the MCMC sampler
T = 5000  # Number of samples
m = 50    # Initial minibatch size
delta = 0.1 # Error bound
average_correction_factor = compute_correction_factor(num_points=100, sigma=0.5)
correction_distribution = create_correction_distribution_function(average_correction_factor)
initial_sample = 0  # Initial sample
data = np.random.normal(0, 1, 1000)  # Example data

samples = mcmc_with_minibatch(T, m, delta, correction_distribution, initial_sample, proposal_distribution, target_distribution, data)

# Plotting the results
plt.figure(figsize=(12, 6))

# Histogram of the samples from the MCMC minibatch algorithm
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='MCMC Minibatch Samples')

# Plot the true distribution for comparison
x = np.linspace(min(samples), max(samples), 1000)
plt.plot(x, stats.norm.pdf(x, loc=0, scale=1), 'r', lw=2, label='True Distribution')

plt.title('Histogram of MCMC Minibatch Samples vs. True Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()


# Calculate the mean and variance of the samples
mean_of_samples = np.mean(samples)
variance_of_samples = np.var(samples)

# Print the results
print(f"Mean of Samples: {mean_of_samples}")
print(f"Variance of Samples: {variance_of_samples}")