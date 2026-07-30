import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import *
from infrastructure import *
import numpy as np
from data import *

# QAM path stays on CPU: small model, complex intermediates are unreliable on MPS.
device = torch.device("cpu")


class MainDataset(Dataset):
    def __init__(self, symbol_indices):
        self.symbol_indices = torch.tensor(symbol_indices, dtype=torch.long)

    def __len__(self):
        return len(self.symbol_indices)

    def __getitem__(self, index):
        idx = self.symbol_indices[index]
        return idx


def _index_ber(pred_idx, true_idx, bits_per_symbol):
    """Bit error rate from symbol indices via XOR of their bit patterns."""
    xor = torch.bitwise_xor(pred_idx, true_idx)
    # count set bits in each XOR result
    bit_errors = torch.zeros_like(xor, dtype=torch.float32)
    x = xor
    for _ in range(bits_per_symbol):
        bit_errors += (x & 1).float()
        x >>= 1
    return (bit_errors.sum() / (len(xor) * bits_per_symbol)).item()


def trainer(dataloader, constellation, decoder, optimizer, loss_function, max_steps=1000, report_every=100, snr_db=20.0):
    constellation.train()
    data_points = []
    loss_over_time = []
    step = 0
    recent_losses = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200
    )

    Es = float(constellation.Es)
    # N0 = Es / (10 ** (snr_db / 10.0))
    # sigma = (N0 / 2.0) ** 0.5
    # AWGN log-likelihood temperature
    # decoder.temperature = 1.0 / N0
    bits_per_symbol = int(np.log2(constellation.M))

    while step < max_steps:
        for symbol_indices in dataloader:
            if step >= max_steps:
                break

            symbol_indices = symbol_indices.to(device)
            transmitted = constellation(symbol_indices)
            noise = 0.1 * (
                torch.randn_like(transmitted.real)
                + 1j * torch.randn_like(transmitted.real)
            )
            
            received = transmitted + noise.to(transmitted.dtype)

            logits = decoder(received)
            loss = loss_function(logits, symbol_indices)
            recent_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # only keep samples from the final report window for plotting
            if step > max_steps - report_every:
                data_points += received.detach().cpu().tolist()

            if step % report_every == 0:
                avg_loss = sum(recent_losses) / len(recent_losses)
                scheduler.step(avg_loss)
                loss_over_time.append(avg_loss)
                recent_losses = []

                pred_idx = decoder.decode_hard(received)
                ser = (pred_idx != symbol_indices).float().mean().item()
                ber = _index_ber(pred_idx, symbol_indices, bits_per_symbol)
                print(
                    f"Step {step} | Avg Loss: {avg_loss:.5f} | "
                    f"SER: {ser:.5f} | BER: {ber:.5f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

            if step >= max_steps:
                print(constellation.normalized_points().detach())
                break

    return data_points, np.array(loss_over_time)
