import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


def truncated_normal_sample(mean, stddev, lower, upper):
    """Sample from a truncated normal distribution."""
    if stddev <= 0:
        raise ValueError(f"Standard deviation must be positive. Got {stddev}.")
    a, b = (lower - mean) / stddev, (upper - mean) / stddev
    return stats.truncnorm.rvs(a, b, loc=mean, scale=stddev)

def Q4(num_samples=5000, burn_in=1000, mu1=20, mu2=20, sigma1=0.5, sigma2=0.5, y=1, option="not_mean"):
    """Gibbs sampler for a single match."""
    s1 = mu1
    s2 = mu2

    s1_samples = []
    s2_samples = []

    Sigma_s = np.diag([sigma1**2, sigma2**2])  
    Sigma_s_post = Sigma_s
    Sigma_t_s = Sigma_s[0][0] + Sigma_s[1][1]
    A = np.array([[1, -1]]) 
    for i in range(num_samples):
        t = s1 - s2

        if y == 1:  
            t_trunc = truncated_normal_sample(t, np.sqrt(Sigma_s_post[0][0] + Sigma_s_post[1][1]), 0, np.inf)
        else:
            t_trunc = truncated_normal_sample(t, np.sqrt(Sigma_s_post[0][0] + Sigma_s_post[1][1]), -np.inf, 0)

        Sigma_t = Sigma_t_s + A @ Sigma_s @ A.T  
        Sigma_t_inv = np.linalg.inv(Sigma_t)  
        Sigma_s_inv = np.linalg.inv(Sigma_s)  
        Sigma_t_s_inv = 1/Sigma_t_s

        Sigma_s_post = np.linalg.inv(Sigma_s_inv + Sigma_t_s_inv * A.T @ A)

        mu_s_post = Sigma_s_post @ (Sigma_s_inv @ np.array([s1, s2]) + Sigma_t_s_inv * A.T @ np.array([t_trunc]))

        s = np.random.multivariate_normal(mu_s_post, Sigma_s_post)
        s1, s2 = s

        #if i > burn_in:
        s1_samples.append(s1)
        s2_samples.append(s2)

    if option == "mean":
        return np.mean(s1_samples), np.mean(s2_samples), Sigma_s_post[0, 0], Sigma_s_post[1, 1]
    else:
        return s1_samples, s2_samples, Sigma_s_post[0, 0], Sigma_s_post[1, 1]


def plot_samples(s1_samples, s2_samples, burn_in=500):
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


def Q5(matches, teams, num_samples=1000):
    """
    Update the skills and variances of teams based on match outcomes.
    """
    skills = {team: (50, 2) for team in teams}  # Initialize (mean, sigma) for each team

    for match in matches:
        team1, team2, score1, score2 = match

        if score1 == score2:  # Skip draws
            continue

        s1_mean, s1_sigma = skills[team1]
        s2_mean, s2_sigma = skills[team2]

        if score1 > score2:
            s1, s2, s1_sigma_new, s2_sigma_new = Q4(num_samples=num_samples, mu1=s1_mean, mu2=s2_mean, sigma1=s1_sigma, sigma2=s2_sigma, y=1, option="mean")
        else:
            s2, s1, s2_sigma_new, s1_sigma_new = Q4(num_samples=num_samples, mu1=s1_mean, mu2=s2_mean, sigma1=s1_sigma, sigma2=s2_sigma, y=-1, option="mean")

        # Update both the mean and the sigma for each team
        skills[team1] = (s1, s1_sigma_new)
        skills[team2] = (s2, s2_sigma_new)

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


def predict_winner(s1, s2):
    """
    Predicts the winner between two players based on their skills.
    Returns +1 if Player 1 is predicted to win, -1 if Player 2 is predicted to win.
    """
    # The decision rule: Player 1 wins if s1 > s2
    return 1 if s1 > s2 else -1


def Q6(matches, skills, num_samples=1000):
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
        
        if actual_result == 1:
            s1_new, s2_new, s1_sigma_new, s2_sigma_new = Q4(num_samples=num_samples, mu1=s1_mean, mu2=s2_mean, sigma1=s1_sigma, sigma2=s2_sigma, y=1, option="mean")
        else:
            s2_new, s1_new, s2_sigma_new, s1_sigma_new = Q4(num_samples=num_samples, mu1=s1_mean, mu2=s2_mean, sigma1=s1_sigma, sigma2=s2_sigma, y=-1, option="mean")

        skills[team1] = (s1_new, s1_sigma_new)
        skills[team2] = (s2_new, s2_sigma_new)

    prediction_rate = correct_predictions / (len(matches) - correct_predictions)
    print(f"Prediction rate: {prediction_rate:.2f}")
    return prediction_rate


def main():
    # Q4
    s1_samples, s2_samples, var_s1, var_s2 = Q4()
    plot_samples(s1_samples, s2_samples)
    
    # Q5
    filename = 'SerieA.csv'  # Change to the correct path of your dataset
    matches = load_dataset(filename)

    teams = set([match[0] for match in matches] + [match[1] for match in matches])
    final_skills = Q5(matches, teams, num_samples=1000)
    rank_teams(final_skills)

    # Q6
    # Load matches from the dataset
    filename = 'SerieA.csv'  # Change to the correct path of your dataset
    matches = load_dataset(filename)

    teams = set([match[0] for match in matches] + [match[1] for match in matches])
    
    # Initialize skills
    skills = {team: (50, 2) for team in teams}  # (mean, sigma)

    # Run the predictions
    prediction_rate = Q6(matches, skills, num_samples=1000)
    
    # Check if better than random guessing
    print("Is prediction better than random guessing? ", "Yes" if prediction_rate > 0.5 else "No")


if __name__ == "__main__":
    main()


