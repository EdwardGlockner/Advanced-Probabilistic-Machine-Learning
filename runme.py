import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def Q4(num_samples=5000, burn_in=1000):
    """

    """
    def truncated_normal(mean, std, a, b):
        """Sample from a truncated normal distribution."""
        return stats.truncnorm.rvs((a - mean) / std, (b - mean) / std, loc=mean, scale=std)

    mu = 10  # Mean of the prior
    sigma = 0.5
    
    #s1 = np.random.normal(mu, sigma)
    #s2 = np.random.normal(mu, sigma)
    s1 = mu
    s2 = mu
    
    # To store samples
    s1_samples = []
    s2_samples = []
    
    for _ in range(num_samples):
        # Sample t = s1 - s2, but truncated such that t > 0 (since Player 1 wins)
        t_mean = s1 - s2
        t = truncated_normal(t_mean, np.sqrt(2 * sigma**2), 0, np.inf)
        
        s1_mean = mu + (t + s2) / 2  
        s1 = np.random.normal(s1_mean, sigma)
        
        s2_mean = mu + (s1 - t) / 2 
        s2 = np.random.normal(s2_mean, sigma)
        
        #if i >= burn_in:
        s1_samples.append(s1)
        s2_samples.append(s2)
    
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
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def Q4_match(s1, s2, mu=25, sigma=2, num_samples=1000):
    """Gibbs sampler for a single match."""
    def truncated_normal(mean, std, a, b):
        """Sample from a truncated normal distribution."""
        return stats.truncnorm.rvs((a - mean) / std, (b - mean) / std, loc=mean, scale=std)

    s1_samples, s2_samples = [], []

    for _ in range(num_samples):
        t_mean = s1 - s2
        t = truncated_normal(t_mean, np.sqrt(2 * sigma**2), 0, np.inf)
        
        s1_mean = mu + (t + s2) / 2
        s1 = np.random.normal(s1_mean, sigma)
        
        s2_mean = mu + (s1 - t) / 2
        s2 = np.random.normal(s2_mean, sigma)

        s1_samples.append(s1)
        s2_samples.append(s2)
    
    return np.mean(s1_samples), np.mean(s2_samples)

def Q5(matches, teams, num_samples=1000):
    """Assumed Density Filtering for a stream of matches."""
    skills = {team: (50, 2) for team in teams}  # (mean, sigma)

    for match in matches:
        team1, team2, score1, score2 = match

        # Skip draws 
        if score1 == score2:
            continue
        
        s1_mean, s1_sigma = skills[team1]
        s2_mean, s2_sigma = skills[team2]

        if score1 > score2:
            s1, s2 = Q4_match(s1_mean, s2_mean, num_samples=num_samples)
        else:
            s2, s1 = Q4_match(s2_mean, s1_mean, num_samples=num_samples)
        
        skills[team1] = (s1, s1_sigma)  # Update skill for team 1
        skills[team2] = (s2, s2_sigma)  # Update skill for team 2

    return skills


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


def main():
    # Q4
    """
    burn_in = 1000
    s1_samples, s2_samples = Q4(num_samples=2000, burn_in=burn_in)
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

