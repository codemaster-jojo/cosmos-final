from torch import nn, optim
from torch.utils.data import DataLoader
from model import Constellation, Decoder
from infrastructure import bits_to_symbol_indices
from train import MainDataset, trainer, device
import numpy as np

constellation_model = Constellation().to(device)
decoder_model = Decoder().to(device)

loss_function = nn.MSELoss()
optimizer = optim.Adam(
    list(constellation_model.parameters()) + list(decoder_model.parameters()),
    lr=0.01,
)

with open("list_of_bits.txt", "r") as file:
    all_symbols = []
    for line in file:
        bits = [int(bit) for bit in line.strip()]
        symbols = bits_to_symbol_indices(bits, 8)
        all_symbols.extend(symbols)
all_symbols = np.array(all_symbols, dtype=np.uint8)

dataset = MainDataset(all_symbols, all_symbols)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

trainer(dataloader, constellation_model, decoder_model, optimizer, loss_function)