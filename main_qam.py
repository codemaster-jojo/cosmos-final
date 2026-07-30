import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Constellation, QAMDecoder, make_qam64_points, NoiseMDN
from infrastructure import bits_to_symbol_indices
from train_qam import MainDataset, trainer, device
from visualize import plot_qam_constellation, plot_loss
import numpy as np
from torch.utils.data import random_split


# device is CPU (imported from train_qam)

# Random start so learning is observable; use init="qam64" for square baseline.
constellation_model = Constellation(
    points=make_qam64_points(init="qam64"),
    M=64,
).to(device)

decoder_model = QAMDecoder(constellation_model).to(device)

'''
decoder_model = QAMDecoder(constellation_model)
decoder_model = decoder_model.to(device)
decoder_model.load_state_dict(torch.load("decoder_qam.pth", weights_only=True))

constellation_model = Constellation(points=make_qam64_points(init="qam64"), M=64)
constellation_model = constellation_model.to(device)
'''

constellation_model.load_state_dict(torch.load("constellation_qam.pth", weights_only=True))


loss_function = nn.CrossEntropyLoss()
# Only constellation params — QAMDecoder registers constellation as a submodule,
# so adding decoder.parameters() would put the same weights in Adam twice.
optimizer = optim.Adam(
    list(constellation_model.parameters()),
    lr=0.001,
)


with open("list_of_bits.txt", "r") as file:
    all_symbols = []
    for line in file:
        bits = [int(bit) for bit in line.strip()]
        symbol_indices = bits_to_symbol_indices(bits, 64)
        all_symbols.extend(symbol_indices)
all_symbols = np.array(all_symbols, dtype=np.uint8)

dataset = MainDataset(all_symbols, all_symbols)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])
dataloader = DataLoader(train_set, batch_size=1024, shuffle=True)

noise_model = NoiseMDN()
noise_model = noise_model.to(device)
noise_model.load_state_dict(torch.load("noise_mdn.pth", weights_only = True))


data_distrb, loss_over_time = trainer(dataloader,constellation_model,decoder_model,noise_model,
                                      optimizer,loss_function,max_steps=10000,report_every=1000,snr_db=20.0)

plot_qam_constellation(
    constellation_model.normalized_points().detach().cpu().numpy(),
    data_distrb,
)
plot_loss(loss_over_time)


torch.save(constellation_model.state_dict(), "constellation_qam.pth")
print("Model Saved!")
