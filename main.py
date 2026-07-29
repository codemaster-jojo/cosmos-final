import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import *
from infrastructure import bits_to_symbol_indices
from train import MainDataset, trainer, device
from visualize import *
import numpy as np
from torch.utils.data import random_split, TensorDataset


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# DO THIS IF IT IS YOUR FIRST TIME RUNNING THIS PROGRAM
#constellation_model = Constellation().to(device)
#decoder_model = Decoder().to(device)

noise_model = NoiseMDN()
noise_model = noise_model.to(device)
noise_model.load_state_dict(torch.load("noise_mdn.pth", weights_only = True))


decoder_model = Decoder()
decoder_model = decoder_model.to(device)
decoder_model.load_state_dict(torch.load("decoder_irl_test.pth", weights_only = True))

constellation_model = Constellation()
constellation_model = constellation_model.to(device)
constellation_model.load_state_dict(torch.load("constellation_irl_test.pth", weights_only = True))

with torch.no_grad():
    sample_transmitted = constellation_model(torch.tensor([0]).to(device))
    noise_samples = torch.stack([noise_model.sample(sample_transmitted) for _ in range(1000)])
print(noise_samples.std().item())


loss_function = nn.MSELoss()
optimizer = optim.Adam(
    list(constellation_model.parameters()) + list(decoder_model.parameters()),
    lr=0.001,
)


with open("/Users/alyang/Downloads/cosmos-final/list_of_bits.txt", "r") as file:
    all_symbols = []
    for line in file:
        bits = [int(bit) for bit in line.strip()]
        symbol_indices = bits_to_symbol_indices(bits, 8)
        all_symbols.extend(symbol_indices)
all_symbols = np.array(all_symbols, dtype=np.uint8)

dataset = MainDataset(all_symbols, all_symbols)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])
dataloader = DataLoader(train_set, batch_size=1024, shuffle=True)


data_distrb, loss_over_time = trainer(dataloader, constellation_model, decoder_model, noise_model, optimizer, loss_function, max_steps=1000, report_every=100)

plot_constellation(constellation_model.normalized_points(), decoder_model.get_boundaries(), data_distrb)
plot_loss(loss_over_time)


torch.save(decoder_model.state_dict(), "decoder_irl_test.pth")
torch.save(constellation_model.state_dict(), "constellation_irl_test.pth")
print("Model Saved!")
# SAVING THE MODEL TO DECODER.PTH / CONSTELLATION.PTH