import matplotlib.pyplot as plt
import numpy as np
import torch

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


def plot_noise(file_path):
    symbols = []
    noise = [] # actually symbols out
    
    with open(file_path, "r") as f:
        for line in f:
            a, b = map(float, line.split())   # use int instead of float if appropriate
            symbols.append(a)
            noise.append(b)

    plt.figure(figsize=(8,6))
    plt.hexbin(symbols, noise, gridsize=100, cmap="viridis", bins="log")
    plt.colorbar(label="log(count)")
    plt.xlabel("Symbol")
    plt.ylabel("Noise")
    plt.title("Noise through Wire Channel")
    plt.show()

def plot_radio_model(model, x, num_inputs=2000, samples_each=200):
    model.eval()
    device = next(model.parameters()).device

    x = torch.as_tensor(x, dtype=torch.float32).flatten()
    # the full dataset is over a million amplitudes, which is far more than a hexbin needs
    if len(x) > num_inputs:
        x = x[torch.randperm(len(x))[:num_inputs]]
    x = x.to(device)

    with torch.no_grad():
        weights, means, stds = model(x)
        # pick which Gaussian each sample comes from, then draw from that Gaussian
        component = torch.multinomial(weights, samples_each, replacement=True)
        chosen_means = means.gather(1, component)
        chosen_stds = stds.gather(1, component)
        ys = chosen_means + chosen_stds * torch.randn_like(chosen_means)

    xs = x.unsqueeze(1).expand_as(ys)

    plt.hexbin(xs.flatten().cpu(), ys.flatten().cpu(), gridsize=150, bins="log")
    plt.colorbar(label="log(count)")
    plt.xlabel("Transmitted")
    plt.ylabel("Received")
    plt.show()
