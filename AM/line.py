
from am import AdaptiveMetropolis
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
parent_directory = os.path.dirname(current_directory)

# Append the 'HMC' directory to the Python path
hmc_directory = os.path.join(parent_directory, 'HMC')
sys.path.append(hmc_directory)
sys.path.append(project_root)
from scipy.optimize import minimize
import scipy.stats as stats
#print(sys.path)
from dram import DRAM
from HMC.TrueDistributions import *
from HMC import TrueDistributions, ChainMakers
from HMC.ChainMakers import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Example of how to use the object and visualize means
if __name__ == "__main__":

    mean_univariate = 3.0
    std_dev_univariate = 1.0
    target_distribution = norm(loc=mean_univariate, scale=std_dev_univariate).pdf
    lower_bound = mean_univariate - std_dev_univariate
    upper_bound = mean_univariate + std_dev_univariate
    initial_params = [0]
    sd = 0.1  # Scaling parameter
    epsilon = 1.0  # Parameter for the time-dependent covariance structure
    t0 = 100  # Time index for transition in covariance structure
    iterations = 500
    percentages = []
    percentages_h = []
    percentages_d =[]
    num_runs = 50
    target_hamilton = MVN(tf.constant([mean_univariate],dtype=tf.float32),tf.constant([std_dev_univariate],dtype=tf.float32))
# Run the experiment multiple times
    for _ in range(num_runs):
        samples_adaptive = AdaptiveMetropolis(target_distribution, initial_params, sd, epsilon, t0, iterations).run()

        samples_hamilton = HamiltonianChainMaker(target_hamilton, n=iterations).getOutput()
        samples_dram = DRAM(target_distribution,initial_params,iterations, t0,sd,epsilon).run()
        

        within_68_percent = np.logical_and(samples_adaptive >= lower_bound, samples_adaptive <= upper_bound)
        samples_adaptive_in_68_percent = samples_adaptive[within_68_percent]
        percentage_in_68_percent = (len(samples_adaptive_in_68_percent) / len(samples_adaptive)) * 100
        percentages.append(percentage_in_68_percent)

      
        within_68_percent_h = np.logical_and(samples_hamilton >= lower_bound, samples_hamilton <= upper_bound)
        samples_hamilton_in_68_percent = samples_hamilton[within_68_percent_h]
        percentage_in_68_percent_h= (len(samples_hamilton_in_68_percent) / len(samples_hamilton)) * 100
        percentages_h.append(percentage_in_68_percent_h)
     
        within_68_percent_d = np.logical_and(samples_dram >= lower_bound, samples_dram <= upper_bound)
        samples_dram_in_68_percent = samples_dram[within_68_percent_d]
        percentage_in_68_percent_d= (len(samples_dram_in_68_percent) / len(samples_dram)) * 100
        percentages_d.append(percentage_in_68_percent_d)

    def mama():
        def target_distribution(x):
            return stats.norm.pdf(x, loc=mean_univariate, scale=std_dev_univariate)

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
        T = 500  # Number of samples
        m = 50    # Initial minibatch size
        delta = 0.1 # Error bound
        average_correction_factor = compute_correction_factor(num_points=100, sigma=0.5)
        correction_distribution = create_correction_distribution_function(average_correction_factor)
        initial_sample = 0  # Initial sample
        data = np.random.normal(mean_univariate, std_dev_univariate, 1000)  # Example data

        samples = mcmc_with_minibatch(T, m, delta, correction_distribution, initial_sample, proposal_distribution, target_distribution, data)
        return samples
    
    percentages_m=[]
    for _ in range(num_runs):
        samples_m = np.array(mama())
        within_68_percent_m = np.logical_and(samples_m >= lower_bound, samples_m<= upper_bound)
        samples_m_in_68_percent = samples_m[within_68_percent_m]
        percentage_in_68_percent_m= (len(samples_m_in_68_percent) / len(samples_m)) * 100
        percentages_m.append(percentage_in_68_percent_m)

    # Plotting the results
    plt.plot([1,1],[min(percentages), max(percentages)], color='black', linewidth=2)
    plt.plot([2,2],[min(percentages_h), max(percentages_h)], color='black', linewidth=2)
    plt.plot([3,3],[min(percentages_d), max(percentages_d)], color='black', linewidth=2)
    plt.plot([4,4],[min(percentages_m), max(percentages_m)], color='black', linewidth=2)
    plt.scatter(1, np.mean(percentages), color='black', marker='*', label='AM')
    plt.scatter(2, np.mean(percentages_h), color='black', marker='s', label='HMC')
    plt.scatter(3, np.mean(percentages_d), color='black', marker='2', label='DRAM')
    plt.scatter(4, np.mean(percentages_m), color='black', marker='+', label='MiniBatch')
    plt.ylim(0, 100)
    # Additional plot customization
    plt.axhline(68.3, color='black', linestyle='--', linewidth=1)  # Horizontal line at y=0
    plt.title('Frequency of Hits')
    plt.xlabel('MCMC methods')
    plt.xticks([])  
    plt.legend()
    plt.savefig('LinePlot.png')
    plt.show()

    
