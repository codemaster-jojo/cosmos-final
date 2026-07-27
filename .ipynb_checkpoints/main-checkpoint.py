import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import *
from infrastructure import *
from train import *
import numpy as np

constellation_model = Constellation()
decoder_model = Decoder()

loss_function = nn.MSELoss()
optimizer = optim.Adam(list(constellation_model.parameters()) + list(decoder_model.parameters()), lr = 0.001) #lr is essentially how big of an adjustment we want at the beginning

with open("list_of_bits.txt","r") as file:
    all_symbols = []
    for line in file:
        bits = []
        for bit in line.strip(): #line.strip removes \n at the end
            bits.append(int(bit))
        symbols = bits_to_symbol_indices(bits, 8) #Converts the bits in groups of 3 to the INDEX of the symbol (0-7)
        all_symbols.extend(symbols)
all_symbols = np.array(all_symbols, dtype = np.uint8)[:100]

dataset = MainDataset(all_symbols, all_symbols) #the first all_symbols will be transformed by the noise
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


trainer(dataloader, constellation_model, decoder_model, optimizer, loss_function)