import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


def truncated_normal_sample(mean, std, lower, upper):
    """Sample from a truncated normal distribution."""
    return stats.truncnorm.rvs((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)



def Q4(num_samples=5000, burn_in=1000, mu1=20, mu2=20, sigma1=0.5, sigma2=0.5, y=1):
    """
    Gibbs sampling to estimate s1 and s2 given the result of one game y.
    
    Parameters:
        num_samples: Number of samples to draw.
        burn_in: Number of samples to discard for burn-in.
        mu1: Mean for the prior distribution of s1.
        mu2: Mean for the prior distribution of s2.
        sigma1: Initial standard deviation for the prior distribution of s1.
        sigma2: Initial standard deviation for the prior distribution of s2.
        y: Result of the game (1 if Player 1 wins, -1 if Player 2 wins).
    """
    
    # Initial values
    s1 = mu1
    s2 = mu2
    
    # To store samples
    s1_samples = []
    s2_samples = []
    
    # Set up the covariance matrix
    Sigma_s = np.diag([sigma1**2, sigma2**2])  # Diagonal covariance matrix
    A = np.array([[1, -1]])  # Linear transformation matrix

    for i in range(num_samples):
        # Compute t based on current values of s1 and s2
        t = s1 - s2
        
        # Sample t based on the outcome
        if y == 1:  # Player 1 wins, t > 0
            t_trunc = truncated_normal_sample(t, np.sqrt(sigma1**2 + sigma2**2), 0, np.inf)
        else:  # Player 2 wins, t < 0
            t_trunc = truncated_normal_sample(t, np.sqrt(sigma1**2 + sigma2**2), -np.inf, 0)

        # Compute posterior covariance and mean for s
        Sigma_t = 0 + A @ Sigma_s @ A.T  # Covariance of t, assuming no extra noise
        Sigma_t_inv = np.linalg.inv(Sigma_t)  # Inverse of the covariance of t
        Sigma_s_inv = np.linalg.inv(Sigma_s)  # Inverse of the prior covariance
        
        # Posterior covariance
        Sigma_s_post = np.linalg.inv(Sigma_s_inv + A.T @ Sigma_t_inv @ A)

        # Posterior mean
        mu_s_post = Sigma_s_post @ (Sigma_s_inv @ np.array([mu1, mu2]) + A.T @ Sigma_t_inv @ np.array([t_trunc]))
        
        # Sample from the posterior
        s = np.random.multivariate_normal(mu_s_post, Sigma_s_post)
        s1, s2 = s

        # Store samples after burn-in
        s1_samples.append(s1)
        s2_samples.append(s2)

    print(np.mean(s1_samples))
    print(np.mean(s2_samples))
    return np.array(s1_samples), np.array(s2_samples)

def plot_samples(s1_samples, s2_samples, burn_in):
    """

    """
    num_samples = len(s1_samples)
    
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
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()


def Q4_match(num_samples=5000, burn_in=1000, mu1=20, mu2=20, sigma1=0.5, sigma2=0.5, y=1):
    """Gibbs sampler for a single match."""
    # Initial values
    s1 = mu1
    s2 = mu2
    
    # To store samples
    s1_samples = []
    s2_samples = []
    
    # Set up the covariance matrix
    Sigma_s = np.diag([sigma1**2, sigma2**2])  # Diagonal covariance matrix
    A = np.array([[1, -1]])  # Linear transformation matrix

    for i in range(num_samples):
        # Compute t based on current values of s1 and s2
        t = s1 - s2
        
        # Sample t based on the outcome
        if y == 1:  # Player 1 wins, t > 0
            t_trunc = truncated_normal_sample(t, np.sqrt(sigma1**2 + sigma2**2), 0, np.inf)
        else:  # Player 2 wins, t < 0
            t_trunc = truncated_normal_sample(t, np.sqrt(sigma1**2 + sigma2**2), -np.inf, 0)

        # Compute posterior covariance and mean for s
        Sigma_t = 0 + A @ Sigma_s @ A.T  # Covariance of t, assuming no extra noise
        Sigma_t_inv = np.linalg.inv(Sigma_t)  # Inverse of the covariance of t
        Sigma_s_inv = np.linalg.inv(Sigma_s)  # Inverse of the prior covariance
        
        # Posterior covariance
        Sigma_s_post = np.linalg.inv(Sigma_s_inv + A.T @ Sigma_t_inv @ A)

        # Posterior mean
        mu_s_post = Sigma_s_post @ (Sigma_s_inv @ np.array([mu1, mu2]) + A.T @ Sigma_t_inv @ np.array([t_trunc]))
        
        # Sample from the posterior
        s = np.random.multivariate_normal(mu_s_post, Sigma_s_post)
        s1, s2 = s

        # Store samples after burn-in
        s1_samples.append(s1)
        s2_samples.append(s2)

    return np.mean(s1_samples), np.mean(s2_samples)


def Q5(matches, teams, num_samples=1000):
    """
    Problems: Should Q4 be run with num_samples???
    """
    skills = {team: (50, 2) for team in teams}  # (mean, sigma)

    for match in matches:
        print(match)
        team1, team2, score1, score2 = match

        # Skip draws 
        if score1 == score2:
            continue
        
        s1_mean, s1_sigma = skills[team1]
        s2_mean, s2_sigma = skills[team2]
        if score1 > score2:
            s1, s2 = Q4_match(num_samples=num_samples, mu1=s1_mean, mu2=s2_mean)
        else:
            s2, s1 = Q4_match(num_samples=num_samples, mu1=s1_mean, mu2=s2_mean)
        
        skills[team1] = (s1, s1_sigma)  # Update skill for team 1
        skills[team2] = (s2, s2_sigma)  # Update skill for team 2

    return skills


def rank_teams(skills):
    """Rank teams by their final skill means."""
    ranked_teams = sorted(skills.items(), key=lambda x: x[1][0], reverse=True)
    for i, (team, skill) in enumerate(ranked_teams):
        print(f"{i + 1}. {team}: Skill {skill[0]:.2f}, Variance: {skill[1]:.2f}")
            s2, s1 = Q4_match(s2_mean, s1_mean, num_samples=num_samples)


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

def Q6():
    pass


def main():
    # Q4
    """
    burn_in = 1000
    s1_samples, s2_samples = Q4(burn_in=burn_in)
    plot_samples(s1_samples, s2_samples, burn_in)
    """
    
    # Q5
    filename = 'SerieA.csv'  # Change to the correct path of your dataset
    matches = load_dataset(filename)

    teams = set([match[0] for match in matches] + [match[1] for match in matches])
    final_skills = Q5(matches, teams, num_samples=1000)
    rank_teams(final_skills)


if __name__ == "__main__":
    main()

