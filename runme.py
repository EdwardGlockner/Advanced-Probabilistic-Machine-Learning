import numpy as np
import time
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


def truncated_normal_sample(mean, stddev, lower, upper):
    """Sample from a truncated normal distribution."""
    a, b = (lower - mean) / stddev, (upper - mean) / stddev
    return stats.truncnorm.rvs(a, b, loc=mean, scale=stddev)


def Q4(num_samples=5000, burn_in=60, mu1=25, mu2=25, sigma1=25/3, sigma2=25/3, y=1, option=""):
    """Gibbs sampler for a single match."""
    s1 = mu1
    s2 = mu2

    s1_samples = []
    s2_samples = []

    Sigma_s = np.diag([sigma1**2, sigma2**2])  
    Sigma_t_s = (25/6)**2
    A = np.array([[1, -1]]) 

    for i in range(num_samples):
        t = s1 - s2

        if y == 1:  
            t_trunc = truncated_normal_sample(t, np.sqrt(Sigma_t_s), 0, np.inf)
        else:
            t_trunc = truncated_normal_sample(t, np.sqrt(Sigma_t_s), -np.inf, 0)

        Sigma_t = Sigma_t_s + A @ Sigma_s @ A.T  
        Sigma_t_inv = np.linalg.inv(Sigma_t)  
        Sigma_s_inv = np.linalg.inv(Sigma_s)  
        Sigma_t_s_inv = 1/Sigma_t_s

        Sigma_s_post = np.linalg.inv(Sigma_s_inv + Sigma_t_s_inv * A.T @ A)
        mu_s_post = Sigma_s_post @ (Sigma_s_inv @ np.array([mu1, mu2]) + Sigma_t_s_inv * A.T @ np.array([t_trunc]))

        s = np.random.multivariate_normal(mu_s_post, Sigma_s_post)
        s1, s2 = s

        if i > burn_in:
            s1_samples.append(s1)
            s2_samples.append(s2)

    #print(f"Mean s1: {np.mean(s1_samples)} \t Variance s1: {np.std(s1_samples)**2}")
    #print(f"Mean s2: {np.mean(s2_samples)} \t Variance s2: {np.std(s2_samples)**2}")

    if option == "mean":
        return np.mean(s1_samples), np.mean(s2_samples), np.std(s1_samples)**2, np.std(s2_samples)
    else:
        return s1_samples, s2_samples, np.std(s1_samples)**2, np.std(s2_samples)**2


def plot_histogram_and_gaussian(samples, mu, sigma, ax, title):
    """Plot histogram of samples and fitted Gaussian distribution."""
    ax.hist(samples, bins=30, density=True, alpha=0.6, color='b', label='Samples Histogram')
    
    # Fit a Gaussian distribution to the samples
    fitted_mu = np.mean(samples)
    fitted_sigma = np.std(samples)

    # Create a range of x values
    x = np.linspace(min(samples), max(samples), 100)
    # Calculate the fitted Gaussian PDF
    p = stats.norm.pdf(x, fitted_mu, fitted_sigma)
    
    ax.plot(x, p, 'r', label='Fitted Gaussian', linewidth=2)
    ax.set_title(title)
    ax.legend()


def run_experiments(sample_sizes, num_samples=1000, burn_in=60):
    """Run experiments for different sample sizes."""
    times = []
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, size in enumerate(sample_sizes):
        start_time = time.time()
        s1_samples, s2_samples, _, _ = Q4(num_samples=size, burn_in=burn_in)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        ax = axs[idx // 2, idx % 2]  # Get the subplot axis
        plot_histogram_and_gaussian(s1_samples, 25, 25/3, ax, f'Samples Size: {size}\nTime: {elapsed_time:.4f}s')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase spacing between plots
    plt.show()
    
    return times


def gaussian_approximation(s1_samples, s2_samples):
    """
    Calculate the Gaussian approximation of the posterior distribution of the skills.
    
    Parameters:
    s1_samples (list): Samples for player one's skill.
    s2_samples (list): Samples for player two's skill.
    
    Returns:
    mu (np.ndarray): Mean of the skills.
    cov (np.ndarray): Covariance matrix of the skills.
    """
    s1_samples = np.array(s1_samples)
    s2_samples = np.array(s2_samples)
    
    mu_s1 = np.mean(s1_samples)
    mu_s2 = np.mean(s2_samples)
    
    covariance_matrix = np.cov(s1_samples, s2_samples)

    mu = np.array([mu_s1, mu_s2])
    
    return mu, covariance_matrix


def plot_gaussian(mu, cov):
    """
    Plot the Gaussian approximation based on the mean and covariance.
    
    Parameters:
    mu (np.ndarray): Mean of the skills.
    cov (np.ndarray): Covariance matrix of the skills.
    """
    x = np.linspace(mu[0] - 3 * np.sqrt(cov[0, 0]), mu[0] + 3 * np.sqrt(cov[0, 0]), 100)
    y = np.linspace(mu[1] - 3 * np.sqrt(cov[1, 1]), mu[1] + 3 * np.sqrt(cov[1, 1]), 100)
    X, Y = np.meshgrid(x, y)
    
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mu, cov)
    Z = rv.pdf(pos)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.scatter(mu[0], mu[1], color='red', label='Mean Skill', s=100)
    plt.title('Gaussian Approximation of Skills')
    plt.xlabel('Player 1 Skill')
    plt.ylabel('Player 2 Skill')
    plt.legend()
    plt.show()


def plot_samples(s1_samples, s2_samples, burn_in=500):
    """

    """
    num_samples = len(s1_samples)
    
    plt.figure(figsize=(14, 8))
    
    plt.plot()
    plt.hist(s1_samples, bins=30, alpha=0.7, color='blue', label='s1')
    plt.hist(s2_samples, bins=30, alpha=0.7, color='red', label='s2')
    plt.title('Histogram of s1 and s2 samples')
    plt.legend()
    plt.show()
    
    plt.plot()
    plt.scatter(s1_samples, s2_samples, alpha=0.5, color='green')
    plt.title('Scatter plot of s1 vs s2')
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.show()

    plt.plot()
    plt.plot(np.arange(num_samples), s1_samples, color='blue', label='s1')
    plt.axvline(x=burn_in, color='red', linestyle='--', label='Burn-in threshold')
    plt.title('s1 over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('s1')
    plt.legend()
    plt.show()

    plt.plot()
    plt.plot(np.arange(num_samples), s2_samples, color='red', label='s2')
    plt.axvline(x=burn_in, color='blue', linestyle='--', label='Burn-in threshold')
    plt.title('s2 over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('s2')
    plt.legend(loc="lower right")
    plt.show()


def plot_prior_vs_posterior(s1_mean_prior, s1_std_prior, s2_mean_prior, s2_std_prior, s1_mean_post, s1_std_post, s2_mean_post, s2_std_post):
    x = np.linspace(10, 40, 1000)

    # Prior distributions
    s1_prior = stats.norm.pdf(x, s1_mean_prior, s1_std_prior)
    s2_prior = stats.norm.pdf(x, s2_mean_prior, s2_std_prior)

    # Posterior distributions
    s1_post = stats.norm.pdf(x, s1_mean_post, s1_std_post)
    s2_post = stats.norm.pdf(x, s2_mean_post, s2_std_post)

    # Plot for s1 (Player 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x, s1_prior, label=f"Prior $p(s_1)$", color='blue')
    plt.plot(x, s1_post, label=f"Posterior $p(s_1|y=1)$", color='red')
    plt.fill_between(x, s1_prior, alpha=0.3, color='blue')
    plt.fill_between(x, s1_post, alpha=0.3, color='red')
    plt.title('Player 1: Prior vs Posterior')
    plt.legend()

    # Plot for s2 (Player 2)
    plt.subplot(1, 2, 2)
    plt.plot(x, s2_prior, label=f"Prior $p(s_2)$", color='blue')
    plt.plot(x, s2_post, label=f"Posterior $p(s_2|y=1)$", color='red')
    plt.fill_between(x, s2_prior, alpha=0.3, color='blue')
    plt.fill_between(x, s2_post, alpha=0.3, color='red')
    plt.title('Player 2: Prior vs Posterior')
    plt.legend()

    plt.tight_layout()
    plt.show()


def Q5(matches, teams, num_samples=2000):
    """
    Update the skills and variances of teams based on match outcomes.
    """
    skills = {team: (20, 2.0) for team in teams}  # initialize
    correct_predictions = 0

    for match in matches:
        team1, team2, score1, score2 = match

        if score1 == score2:  # Skip draws
            continue

        s1_mean, s1_sigma = skills[team1]
        s2_mean, s2_sigma = skills[team2]

        prediction = predict_winner(s1_mean, s2_mean)
        actual_result = 1 if score1 > score2 else -1

        if prediction == actual_result:
            correct_predictions += 1

        if score1 > score2:
            s1, s2, s1_sigma_new, s2_sigma_new = Q4(num_samples=num_samples, mu1=s1_mean, mu2=s2_mean, sigma1=np.sqrt(s1_sigma), sigma2=np.sqrt(s2_sigma), y=1, option="mean")
        else:
            s1, s2, s1_sigma_new, s2_sigma_new = Q4(num_samples=num_samples, mu1=s1_mean, mu2=s2_mean, sigma1=np.sqrt(s1_sigma), sigma2=np.sqrt(s2_sigma), y=-1, option="mean")

        skills[team1] = (s1, s1_sigma_new)
        skills[team2] = (s2, s2_sigma_new)

    prediction_rate = correct_predictions / (len(matches) - correct_predictions)

    return skills, prediction_rate


def rank_teams(skills):
    """Rank teams by their final skill means."""
    ranked_teams = sorted(skills.items(), key=lambda x: x[1][0], reverse=True)
    for i, (team, skill) in enumerate(ranked_teams):
        print(f"{i + 1}. {team}: Skill {skill[0]:.2f}, Variance: {skill[1]:.2f}")


def load_dataset(filename):
    """Load the Serie A dataset from a CSV file."""
    df = pd.read_csv(filename)
    matches = []
    
    for _, row in df.iterrows():
        team1 = row['team1'].strip()
        team2 = row['team2'].strip()
        score1 = int(row['score1'])
        score2 = int(row['score2'])
        matches.append((team1, team2, score1, score2))

    return matches


def predict_winner(s1, s2):
    """
    Predicts the winner between two players based on their skills.
    Returns +1 if Player 1 is predicted to win, -1 if Player 2 is predicted to win.
    """
    return 1 if s1 > s2 else -1


def main():
    # Q4
    """
    burn_in = 10
    s1_samples, s2_samples, var_s1, var_s2 = Q4(num_samples=5000, burn_in=60)
    plot_samples(s1_samples, s2_samples, burn_in=60)

    mu, cov = gaussian_approximation(s1_samples, s2_samples)

    # Plotting the Gaussian approximation
    plot_gaussian(mu, cov)

    # Define different sample sizes
    sample_sizes = [250, 500, 1000, 2000]

    # Run experiments and get the execution times
    execution_times = run_experiments(sample_sizes)

    # Print execution times
    for size, time in zip(sample_sizes, execution_times):
        print(f"Time taken for {size} samples: {time:.4f} seconds")

    # Assuming the following values from your prior and posterior Gaussian approximations
    s1_mean_prior, s1_std_prior = 25, 25/3
    s2_mean_prior, s2_std_prior = 25, 25/3

    plot_prior_vs_posterior(s1_mean_prior, s1_std_prior, s2_mean_prior, s2_std_prior, np.mean(s1_samples), np.std(s1_samples), np.mean(s2_samples), np.std(s2_samples))
    """

    # Q5 and Q6
    filename = 'SerieA.csv'
    matches = load_dataset(filename)

    teams = set([match[0] for match in matches] + [match[1] for match in matches])
    final_skills, prediction_rate = Q5(matches, teams, num_samples=2000)
    rank_teams(final_skills)
    
    print(f"Prediction rate: {prediction_rate:.2f}")
    print("Is prediction better than random guessing? ", "Yes" if prediction_rate > 0.5 else "No")

    # shuffle matches
    random.shuffle(matches) 
    teams = set([match[0] for match in matches] + [match[1] for match in matches])
    final_skills, prediction_rate = Q5(matches, teams, num_samples=2000)
    rank_teams(final_skills)
    
    print(f"Shuffled Prediction rate: {prediction_rate:.2f}")
    print("Is shuffled prediction better than random guessing? ", "Yes" if prediction_rate > 0.5 else "No")

if __name__ == "__main__":
    main()


