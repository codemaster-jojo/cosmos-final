import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import Constellation, Decoder
from infrastructure import bits_to_symbol_indices
import numpy as np

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

class MainDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

constellation = Constellation().to(device)
decoder = Decoder().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(list(constellation.parameters()) + list(decoder.parameters()), lr = 0.001) #lr is essentially how big of an adjustment we want at the beginning

with open("list_of_bits.txt","r") as file:
    all_symbols = []
    for line in file:
        bits = []
        for bit in line.strip(): #line.strip removes \n at the end
            bits.append(int(bit))
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

        # # AWGN channel
        # noise = torch.zeros_like(transmitted) # 
        noise = 0.1 * torch.randn_like(transmitted)
        received = transmitted + noise

        # Neural decoder
        prediction = decoder(received)
        loss = loss_function(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 2 == 0:
            print(loss.item())
            print(constellation.normalized_points())
            print(decoder.get_boundaries())