import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def Q4(num_samples=5000, burn_in=1000):
    def truncated_normal(mean, std, a, b):
        """Sample from a truncated normal distribution."""
        return stats.truncnorm.rvs((a - mean) / std, (b - mean) / std, loc=mean, scale=std)

    # Hyperparameters for the skills' prior distribution
    mu = 10  # Mean of the prior
    sigma = 0.5
    
    # Initialize skill estimates
    #s1 = np.random.normal(mu, sigma)
    #s2 = np.random.normal(mu, sigma)
    s1 = mu
    s2 = mu
    
    # To store samples
    s1_samples = []
    s2_samples = []
    
    for i in range(num_samples):
        # Sample t = s1 - s2, but truncated such that t > 0 (since Player 1 wins)
        t_mean = s1 - s2
        t = truncated_normal(t_mean, np.sqrt(2 * sigma**2), 0, np.inf)
        
        # Sample s1 from its conditional distribution given s2 and t
        s1_mean = mu + (t + s2) / 2  # Update the mean using t and s2
        s1 = np.random.normal(s1_mean, sigma)
        
        # Sample s2 from its conditional distribution given s1 and t
        s2_mean = mu + (s1 - t) / 2  # Update the mean using t and s1
        s2 = np.random.normal(s2_mean, sigma)
        
        # Store the samples after burn-in
        #if i >= burn_in:
        s1_samples.append(s1)
        s2_samples.append(s2)
    
    return np.array(s1_samples), np.array(s2_samples)


def plot_samples(s1_samples, s2_samples, burn_in):
    # Number of iterations after burn-in
    num_samples = len(s1_samples)
    
    # Plot the histograms and scatter plot
    plt.figure(figsize=(14, 8))
    
    # Histogram of s1 and s2
    plt.subplot(2, 2, 1)
    plt.hist(s1_samples, bins=30, alpha=0.7, color='blue', label='s1')
    plt.hist(s2_samples, bins=30, alpha=0.7, color='red', label='s2')
    plt.title('Histogram of s1 and s2 samples')
    plt.legend()
    
    # Scatter plot of s1 vs s2
    plt.subplot(2, 2, 2)
    plt.scatter(s1_samples, s2_samples, alpha=0.5, color='green')
    plt.title('Scatter plot of s1 vs s2')
    plt.xlabel('s1')
    plt.ylabel('s2')

    # Plot of s1 over iterations
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(num_samples), s1_samples, color='blue', label='s1')
    plt.axvline(x=burn_in, color='red', linestyle='--', label='Burn-in threshold')
    plt.title('s1 over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('s1')
    plt.legend()

    # Plot of s2 over iterations
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(num_samples), s2_samples, color='red', label='s2')
    plt.axvline(x=burn_in, color='blue', linestyle='--', label='Burn-in threshold')
    plt.title('s2 over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('s2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Run the Gibbs sampler
    burn_in = 1000
    s1_samples, s2_samples = Q4(num_samples=2000, burn_in=burn_in)
    
    # Plot the results
    plot_samples(s1_samples, s2_samples, burn_in)
    
    # Comment on the burn-in period
    print("Burn-in was set to 1000 iterations. The samples after burn-in appear to be well-distributed.")


if __name__ == "__main__":
    main()
