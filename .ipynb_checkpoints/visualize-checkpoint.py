import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import Voronoi, voronoi_plot_2d

from digicomm import *

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
    plt.xlabel("Symbol In")
    plt.ylabel("Symbol Out")
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

def plot_qam_symbols(file_name):
    symbols_in = []
    noise = []
    
    with open(file_name, "r") as f:
        for line in f:
            a, b = line.split()
            a = complex(a.strip("()"))
            b = complex(b.strip("()"))
            symbols_in.append(a)
            noise.append(b - a)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    hb1 = ax[0].hexbin(np.real(symbols_in), np.real(noise), gridsize=100, cmap="viridis", bins="log")
    fig.colorbar(hb1, ax=ax[0], label="log(count)")
    ax[0].set_xlabel("Symbol In - Real")
    ax[0].set_ylabel("Noise - Real")
    ax[0].set_title("Real Channel Noise")

    hb2 = ax[1].hexbin(np.imag(symbols_in), np.imag(noise), gridsize=100, cmap="viridis", bins="log")
    fig.colorbar(hb2, ax=ax[1], label="log(count)")
    ax[1].set_xlabel("Symbol In - Imaginary")
    ax[1].set_ylabel("Noise - Imaginary")
    ax[1].set_title("Imaginary Channel Noise")

    plt.tight_layout()
    plt.show()

    
def plot_constellation_with_model(model, constellation, num_repeat=1000):
    model.eval()
    device = next(model.parameters()).device

    rng = np.random.default_rng()

    symbols = np.array(constellation.tolist() * num_repeat, dtype=np.float32)
    rng.shuffle(symbols)

    x = torch.tensor(symbols[:, None], dtype=torch.float32).to(device)

    with torch.no_grad():
        w, m, std = model(x)

        # remove extra dimensions
        w = w.squeeze()
        m = m.squeeze()
        std = std.squeeze()

        c = torch.multinomial(w, 1).squeeze(1)

        received = (
            m[torch.arange(len(x)), c]
            + std[torch.arange(len(x)), c] * torch.randn(len(x), device=device)
        )

    plt.figure(figsize=(6,6))
    plt.hist(received.cpu().numpy(), bins=500)
    plt.xlabel("Symbol Received")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

def plot_qam_constellation(constellation, distribution):
    # use the scipy voronoi thing to find closest neighbor boundaries
    pts = np.column_stack([np.real(constellation), np.imag(constellation)])
    vor = Voronoi(pts)
    fig, ax = plt.subplots() # make sure evreythings on one graph

    ax.hexbin(np.real(distribution), np.imag(distribution), gridsize=100, bins="log")
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='k')
    ax.scatter(pts[:,0], pts[:,1], c='r')

    plt.xlabel("Real Component")
    plt.ylabel("Imaginary Component")
    plt.title("QAM Constellation")
    plt.axis('equal')
    plt.show()

def plot_256qam_standard():
    scale = 1 / np.sqrt(170)

    
    normalized_256QAM = np.array(
        [
            x + 1j*y
            for x in range(-15, 16, 2)
            for y in range(-15, 16, 2)
        ],
        dtype=np.complex64
    ) * scale

    samples_per_symbol = 1000
    snr_db = 25

    Es = 1  # normalized constellation energy
    noise_power = Es / (10**(snr_db / 10))
    noise_std = np.sqrt(noise_power / 2)

    symbols = np.repeat(normalized_256QAM, samples_per_symbol)
    noise = noise_std * (
        np.random.randn(len(symbols)) + 1j*np.random.randn(len(symbols))
    )

    distribution = symbols + noise

    plot_qam_constellation(normalized_256QAM, distribution)