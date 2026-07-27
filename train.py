import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

x = x.to(device)
y = y.to(device)
# Compute prediction error
prediction = model(x)
loss = loss_fn(prediction, y)
# Backpropagation
# loss.backward()
# optimizer.step()
# optimizer.zero_grad()

#Every 100 batches, calculates the loss
if batch % 100 == 0:
    loss = loss.item()
    current = (batch + 1) * len(X)
    print(f"loss: {round(loss, 2)}  [{current}/{size}]")
