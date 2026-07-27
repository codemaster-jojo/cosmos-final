import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 #For images
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
train_dataset = MainDataset(features, labels) #uint64 is needed for the loss function input
train_dataloader = Dataloader(train_dataset, batch_size = 64)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001) #lr is essentially how big of an adjustment we want at the beginning




def trainer(dataloader, optimizer, loss_function, model)
    for batch, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        # Compute prediction error
        prediction = model(features)
        loss = loss_function(prediction, labels) #FIND LOSS FUNCTION!!!
        # Backpropagation
        loss.backward() #Find Gradients
        optimizer.step() #Readjust Weights and Biases
        optimizer.zero_grad() #Reset all the gradients
    
        #Every 50 batches, calculates the loss
        if batch % 50 == 0:
            loss = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {round(loss, 2)}  [{current}/{size}]")
