import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import NoiseMDN
from infrastructure import bits_to_symbol_indices
from train import MainDataset, radio_trainer, device
from visualize import *
import numpy as np
from torch.utils.data import random_split, TensorDataset

noise_model = NoiseMDN().to(device)

noise_model = NoiseMDN()
noise_model = noise_model.to(device)
noise_model.load_state_dict(torch.load("noise_mdn.pth", weights_only = True))

features = []
labels = []

with open(file_path, "r") as f:
    for line in f:
        a, b = map(float, line.split())   # use int instead of float if appropriate
        symbols.append(a)
        noise.append(b)


dataset = NoiseDataset(features, labels)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

model = NoiseMDN(num_symbols=8, embed_dim=3, hidden=64, K=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

train_hist, val_hist = trainer(model, train_loader, val_loader, optimizer, scheduler)

plot_loss(train_hist)
#SAVING IS EMBEDDED IN TRAIN FUNCTION

print("Training complete.")