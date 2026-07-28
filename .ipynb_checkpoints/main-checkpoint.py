import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Constellation, Decoder
from infrastructure import bits_to_symbol_indices
from train import MainDataset, trainer, device
from visualize import *
import numpy as np

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

'''
DO THIS IF IT IS YOUR FIRST TIME RUNNING THIS PROGRAM
constellation_model = Constellation().to(device)
decoder_model = Decoder().to(device)
'''

decoder_model = Decoder()
decoder_model = decoder_model.to(device)
decoder_model.load_state_dict(torch.load("decoder.pth", weights_only = True))

constellation_model = Constellation()
constellation_model = constellation_model.to(device)
constellation_model.load_state_dict(torch.load("constellation.pth", weights_only = True))

loss_function = nn.MSELoss()
optimizer = optim.Adam(
    list(constellation_model.parameters()) + list(decoder_model.parameters()),
    lr=0.01,
)

with open("list_of_bits.txt", "r") as file:
    all_symbols = []
    for line in file:
        bits = [int(bit) for bit in line.strip()]
        symbol_indices = bits_to_symbol_indices(bits, 8)
        all_symbols.extend(symbol_indices)
all_symbols = np.array(all_symbols, dtype=np.uint8)

dataset = MainDataset(all_symbols, all_symbols)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

data_distrb, loss_over_time = trainer(dataloader, constellation_model, decoder_model, optimizer, loss_function, max_steps=1000, report_every=100)
plot_constellation(constellation_model.normalized_points(), decoder_model.get_boundaries(), data_distrb)
plot_loss(loss_over_time)


torch.save(decoder_model.state_dict(), "decoder.pth")
torch.save(constellation_model.state_dict(), "constellation.pth")
print("Model Saved!")
# SAVING THE MODEL TO DECODER.PTH / CONSTELLATION.PTH
