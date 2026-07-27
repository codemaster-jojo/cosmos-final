import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import numpy as np

class MainDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


with open("your_file.txt", "r", encoding="utf-8") as file:
    features = []
    for line in file:
        line_list = []
        for bit in line.strip(): #line.strip removes \n at the end
            line_list.append(bit)
        features.append(line_list)
    features = np.array(features, dtype = np.uint8)
    labels = np.zeros(len(features), dtype = np.uint64)
dataset = MainDataset(features, labels) #uint64 is needed for the loss function input


features = features.to(device)
labels = labels.to(device)
# Compute prediction error
prediction = model(features)
loss = loss_function(prediction, labels) #FIND LOSS FUNCTION!!!
# Backpropagation
# loss.backward()
# optimizer.step()
# optimizer.zero_grad()

#Every 100 batches, calculates the loss
if batch % 100 == 0:
    loss = loss.item()
    current = (batch + 1) * len(X)
    print(f"loss: {round(loss, 2)}  [{current}/{size}]")
