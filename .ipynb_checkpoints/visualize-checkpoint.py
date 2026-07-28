import matplotlib.pyplot as plt
import numpy as np

def plot_constellation(constellation, boundaries, distribution):
    constellation = constellation.detach().cpu().numpy()
    boundaries = boundaries.detach().cpu().numpy()
    
    # distribution histogram + constellation points + dotted line boundaries
    plt.hist(distribution,bins=500, label="Noisy Signals")
    plt.scatter(constellation, np.zeros_like(constellation), color="red", label="Constellation")
    for b in boundaries:
        plt.axvline(b, color='black', linestyle='--', label='Boundary')

    plt.legend()
    plt.show()


def plot_loss(loss):
    plt.plot(loss, label="Loss over Time")

    plt.xlabel("Time", fontsize=12) # model is too small to need epochs so we just used batches
    plt.ylabel("Loss", fontsize=12)
    
    plt.legend()
    plt.show()