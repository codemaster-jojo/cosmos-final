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

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(constellation.parameters()) + list(decoder.parameters()), lr = 0.001) #lr is essentially how big of an adjustment we want at the beginning

with open("list_of_bits.txt","r") as file:
    all_symbols = []
    for line in file:
        line_list = []
        for bit in line.strip(): #line.strip removes \n at the end
            line_list.append(bit)
        bits.append(line_list)
        symbols = bits_to_symbol_indices(bits, 8) #Converts the bits in groups of 3 to the INDEX of the symbol (0-7)
        all_symbols.extend(symbols)
all_symbols = np.array(all_symbols, dtype = np.uint8)

dataset = MainDataset(all_symbols, all_symbols) #the first all_symbols will be transformed by the noise
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

def trainer(dataloader, constellation, decoder, optimizer, loss_function):
    constellation.train() #sets the nn into training mode
    decoder.train()
    for batch,(symbol_indices, labels) in enumerate(dataloader):
        symbol_indices = symbol_indices.to(device)
        labels = labels.to(device)

        # returns the amplitudes of all our symbols (they are currently in all the indices)
        transmitted = constellation(symbol_indices)

        # AWGN channel
        noise = 0.1 * torch.randn_like(transmitted)
        received = transmitted + noise

        # Neural decoder
        logits = decoder(received)
        loss = loss_function(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            print(loss.item())
