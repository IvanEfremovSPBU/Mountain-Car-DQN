import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def plot_score(data, means=None, filename="run_scores.png"):
    
    if means == None:
        means = np.mean(data) * np.ones(len(data))
    
    plt.figure(figsize=(24, 6))

    plt.ylim(-160, -130)
    t = np.linspace(1, len(data), len(means))

    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.plot(data)
    plt.plot(t, means)

    plt.legend(['Score', 'Mean of last 10 episodes'])

    plt.savefig('./plots/' + filename)

    plt.figure(figsize=(24, 6))

    plt.ylim(-160, -130)

    plt.xlabel("Episode")
    plt.ylabel("Score")


def plot_range(data, means, filename='range.png'):
    
    if means == None:
        means = np.mean(data) * np.ones(len(data))
    
    t = np.linspace(1, len(data), len(means))
    
    means = means - np.mean(data) * np.ones(len(means)) + np.min(data) * np.ones(len(means))
    plt.plot(t, means)

    means = means + np.mean(data) * np.ones(len(means)) - np.min(data) * np.ones(len(means))
    plt.plot(t, means)

    plt.savefig('./plots/' + filename)


def plot_percentiles(data, filename='percentiles.png'):
    zero_percentile = np.percentile(data, 0)
    hundred_percentile = np.percentile(data, 100)

    plt.figure(figsize=(24, 6))

    plt.title(f"0_percentile: {zero_percentile}    100_percentile: {hundred_percentile}")

    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.xlim(0, len(data))
    plt.ylim(-160, -130)

    plt.axline([0, zero_percentile], slope=0)
    plt.axline([0, hundred_percentile], slope=0)

    plt.savefig('./plots/' + filename)


def plot_conf_interval(data, filename='confidence_interval.png'):
    interval = st.norm.interval(0.95, loc=np.mean(data), scale=st.sem(data))
    low_bound = interval[0]
    up_bound = interval[1]

    plt.figure(figsize=(24, 6))

    plt.title(f"Confidence interval: {interval}")

    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.xlim(0, len(data))
    plt.ylim(round(low_bound)-0.5, round(up_bound)+0.5)

    plt.axline([0, low_bound], slope=0)
    plt.axline([0, up_bound], slope=0)

    plt.savefig('./plots/' + filename)