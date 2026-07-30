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

with open("/Users/alyang/Downloads/cosmos-final/signals_through_wire_complex.txt", "r") as f:
    for line in f:
        a, b = map(complex, line.split())   # use int instead of float if appropriate
        features.append(a)
        labels.append(b)

imag_features = []
imag_labels = []
real_features = []
real_labels = []
for i in range(len(features)):
    real_features.append(features[i].real)
    imag_features.append(features[i].imag)
    real_labels.append(labels[i].real)
    imag_labels.append(labels[i].imag)



real_dataset = MainDataset(real_features, real_labels, feature_dtype=torch.float32)
imag_dataset = MainDataset(imag_features, imag_labels, feature_dtype=torch.float32)
real_train_set, real_val_set, real_test_set = random_split(real_dataset, [0.8, 0.1, 0.1])
imag_train_set, imag_val_set, imag_test_set = random_split(imag_dataset, [0.8, 0.1, 0.1])

real_train_loader = DataLoader(real_train_set, batch_size=1024, shuffle=True)
real_val_loader = DataLoader(real_val_set, batch_size=1024, shuffle=False)
real_test_loader = DataLoader(real_test_set, batch_size=1024, shuffle=False)

imag_train_loader = DataLoader(imag_train_set, batch_size=1024, shuffle=True)
imag_val_loader = DataLoader(imag_val_set, batch_size=1024, shuffle=False)
imag_test_loader = DataLoader(imag_test_set, batch_size=1024, shuffle=False)

#For first run only
# real_model = NoiseMDN(hidden=64, K=3).to(device)
# imag_model = NoiseMDN(hidden=64, K=3).to(device)



real_model = NoiseMDN()
real_model = real_model.to(device)
real_model.load_state_dict(torch.load("real_noise_mdn.pth", weights_only = True))
imag_model = NoiseMDN()
imag_model = imag_model.to(device)
imag_model.load_state_dict(torch.load("imag_noise_mdn.pth", weights_only = True))


real_optimizer = optim.Adam(real_model.parameters(), lr=0.001)
real_scheduler = optim.lr_scheduler.ReduceLROnPlateau(real_optimizer, mode="min", factor=0.5, patience=2)

imag_optimizer = optim.Adam(imag_model.parameters(), lr=0.001)
imag_scheduler = optim.lr_scheduler.ReduceLROnPlateau(imag_optimizer, mode="min", factor=0.5, patience=2)

real_features = torch.tensor(real_features, dtype=torch.float32).to(device)
imag_features = torch.tensor(imag_features, dtype=torch.float32).to(device)

real_train_hist, real_val_hist = radio_trainer(real_model, real_train_loader, real_val_loader, real_optimizer, real_scheduler)
imag_train_hist, imag_val_hist = radio_trainer(imag_model, imag_train_loader, imag_val_loader, imag_optimizer, imag_scheduler)

plot_loss(real_train_hist)
plot_loss(imag_train_hist)

plot_radio_model(real_model, real_features)
plot_radio_model(imag_model, imag_features)
# plot_constellation_with_model(model, torch.tensor([
#                 -1.5275252316519465,
#                 -1.0910894511799618,
#                 -0.6546536707079771,
#                 -0.21821789023599236,
#                  0.21821789023599236,
#                  0.6546536707079771,
#                  1.0910894511799618,
#                  1.5275252316519465
#             ]))


torch.save(real_model.state_dict(), "real_noise_mdn.pth")
torch.save(imag_model.state_dict(), "imag_noise_mdn.pth")
print("Training complete.")