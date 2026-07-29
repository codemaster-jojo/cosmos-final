import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import NoiseMDN
from infrastructure import bits_to_symbol_indices
from train import MainDataset, radio_trainer, device
from visualize import *
import numpy as np
from torch.utils.data import random_split, TensorDataset


features = []
labels = []

with open("/Users/alyang/Downloads/cosmos-final/signals_through_wire.txt", "r") as f:
    for line in f:
        a, b = map(float, line.split())   # use int instead of float if appropriate
        features.append(a)
        labels.append(b)



dataset = MainDataset(features, labels, feature_dtype=torch.float32)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)


#For first run only
#model = NoiseMDN(hidden=64, K=3).to(device)



model = NoiseMDN()
model = model.to(device)
model.load_state_dict(torch.load("noise_mdn.pth", weights_only = True))
# model.load_state_dict(torch.load("noise_mdn.pth", weights_only = True))

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

features = torch.tensor(features, dtype=torch.float32).to(device)

train_hist, val_hist = radio_trainer(model, train_loader, val_loader, optimizer, scheduler)

plot_loss(train_hist)

plot_radio_model(model, features)
plot_constellation_with_model(model, torch.tensor([
                -1.5275252316519465,
                -1.0910894511799618,
                -0.6546536707079771,
                -0.21821789023599236,
                 0.21821789023599236,
                 0.6546536707079771,
                 1.0910894511799618,
                 1.5275252316519465
            ]))

#SAVING IS EMBEDDED IN TRAIN FUNCTION

print("Training complete.")