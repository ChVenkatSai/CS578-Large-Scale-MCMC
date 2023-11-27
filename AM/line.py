
from am import AdaptiveMetropolis
from ..HMC.ChainMakers import HamiltonianChainMaker
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
    num_runs = 10
# Run the experiment multiple times
    for _ in range(num_runs):
        samples = AdaptiveMetropolis(target_distribution, initial_params, sd, epsilon, t0, iterations).run()
        within_68_percent = np.logical_and(samples >= lower_bound, samples <= upper_bound)
        samples_in_68_percent = samples[within_68_percent]
        percentage_in_68_percent = (len(samples_in_68_percent) / len(samples)) * 100
        percentages.append(percentage_in_68_percent)

    # Plotting the results
    plt.plot([1,1],[min(percentages), max(percentages)], color='black', linewidth=2, label='Min to Max')
    plt.scatter(1, np.mean(percentages), color='black', marker='*', label='Mean')

    # Additional plot customization
    plt.axhline(68.3, color='black', linestyle='--', linewidth=1)  # Horizontal line at y=0
    plt.title('Frequency of Hits')
    plt.xlabel('MCMC methods')
    plt.xticks([])  
    plt.legend()
    plt.show()

    
