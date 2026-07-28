import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import Constellation, Decoder
from infrastructure import bits_to_symbol_indices, calculate_BER, pam_symbols_to_bits, pam_from_indices
import numpy as np
from data import *

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

def trainer(dataloader, constellation, decoder, optimizer, loss_function, max_steps=1000, report_every=100):
    constellation.train()
    decoder.train()
    data_points = []
    loss_over_time = []
    step = 0
    recent_losses = []  # accumulates batch losses between reports
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

    for symbol_indices, labels in dataloader:
        symbol_indices = symbol_indices.to(device)
        labels = labels.to(device)

        transmitted = constellation(symbol_indices)
        noise = 0.1 * torch.randn_like(transmitted)
        received = transmitted + noise
        
        data_points += received.tolist()
        decoder.temperature = min(50.0, 5.0 + step * 0.05) 
        prediction = decoder(received)
        loss = loss_function(prediction, labels)
        recent_losses.append(loss.item())  # track every step's loss for averaging

        step += 1

        if step % report_every == 0:
            avg_loss = sum(recent_losses) / len(recent_losses)
            scheduler.step(avg_loss)  # let the scheduler react to the averaged, less noisy loss
            loss_over_time.append(avg_loss)
            recent_losses = []  # reset the window for the next report interval

            constellation_np = constellation.normalized_points().detach().cpu().numpy()
            pam_symbols = pam_from_indices(constellation_np, symbol_indices)
            raw_bits = pam_symbols_to_bits(pam_symbols, constellation_np)
            received_symbols = decoder.decode_hard(received)
            received_np = received_symbols.detach().cpu().numpy()
            pam_symbols = pam_from_indices(constellation_np, received_np)
            raw_received_bits = pam_symbols_to_bits(pam_symbols, constellation_np)
            print(f"Step {step} | Avg Loss: {avg_loss:.4f} | BER: {calculate_BER(raw_received_bits, raw_bits):.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if step == max_steps - 1:
            print(constellation.normalized_points())
            print(decoder.get_boundaries())
        if step > max_steps:
            return data_points, np.array(loss_over_time)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return data_points, np.array(loss_over_time)