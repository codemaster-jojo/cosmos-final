import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import *
from infrastructure import bits_to_symbol_indices, calculate_BER, qam_symbols_to_bits, qam_from_indices
import numpy as np
from data import *
from collections import Counter

# QAM stays on CPU: complex intermediates are unreliable on MPS.
device = torch.device("cpu")


class MainDataset(Dataset):
    def __init__(self, features, labels, feature_dtype=torch.long):
        # feature_dtype defaults to long because Constellation lookups need integer indices
        self.features = torch.tensor(features, dtype=feature_dtype)
        # long labels for CrossEntropyLoss over symbol indices
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def trainer(dataloader, constellation, decoder, real_noise_model, imag_noise_model, optimizer, loss_function, val_loader = None, max_steps=1000, report_every=100, snr_db=20.0):    
    constellation.train()
    decoder.train()
    data_points = []
    loss_over_time = []
    step = 0
    recent_losses = []  # accumulates batch losses between reports
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    Es = float(constellation.Es)
    N0 = Es / (10 ** (snr_db / 10.0)) #Noise Variance, how much noise power
    sigma = (N0 / 2.0) ** 0.5 #Splits noise variance between complex and real, and then takes sqrt for std
    # AWGN log-likelihood temperature (replaces the PAM annealing schedule)
    decoder.temperature = 1.0 / N0

    best_val_ber = float('inf')
    best_state = None
    
    for symbol_indices, labels in dataloader:
        symbol_indices = symbol_indices.to(device)
        labels = labels.to(device)

        transmitted = constellation(symbol_indices)
        # complex AWGN in place of NoiseMDN (which is real/PAM-only)
        # noise = sigma * (
        #     torch.randn_like(transmitted.real) + 1j * torch.randn_like(transmitted.real)
        # )
        transmitted_real, transmitted_imag = np.real(transmitted), np.imag(transmitted)
        received_real, received_imag = real_noise_model.sample(transmitted_real), imag_noise_model.sample(transmitted_imag)
        received = received_real + 1j * received_imag

        data_points += received.detach().cpu().tolist()
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
            qam_symbols = qam_from_indices(constellation_np, symbol_indices.cpu().numpy())
            raw_bits = qam_symbols_to_bits(qam_symbols, constellation_np)
            received_indices = decoder.decode_hard(received)
            received_np = received_indices.detach().cpu().numpy()
            qam_symbols = qam_from_indices(constellation_np, received_np)
            raw_received_bits = qam_symbols_to_bits(qam_symbols, constellation_np)
            print(f"Step {step} | Avg Loss: {avg_loss:.5f} | BER: {calculate_BER(raw_received_bits, raw_bits):.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_loader is not None:
                constellation.eval()
                decoder.eval()
                val_errors, val_bits_total = 0, 0
                with torch.no_grad():
                    for val_symbols, val_labels in val_loader:
                        val_symbols = val_symbols.to(device)
                        val_transmitted = constellation(val_symbols)
                        vt_real, vt_imag = np.real(val_transmitted), np.imag(val_transmitted)
                        vr_real = real_noise_model.sample(vt_real)
                        vr_imag = imag_noise_model.sample(vt_imag)
                        val_received = vr_real + 1j * vr_imag
                        val_decoded = decoder.decode_hard(val_received)

                        val_const_np = constellation.normalized_points().detach().cpu().numpy()
                        true_syms = qam_from_indices(val_const_np, val_symbols.cpu().numpy())
                        true_bits = qam_symbols_to_bits(true_syms, val_const_np)
                        pred_syms = qam_from_indices(val_const_np, val_decoded.cpu().numpy())
                        pred_bits = qam_symbols_to_bits(pred_syms, val_const_np)

                        val_ber_batch = calculate_BER(pred_bits, true_bits)
                        val_errors += val_ber_batch * len(true_bits)
                        val_bits_total += len(true_bits)
                val_ber = val_errors / val_bits_total
                print(f"               Val BER: {val_ber:.5f}")

                if val_ber < best_val_ber:
                    best_val_ber = val_ber
                    best_state = {
                        'constellation': constellation.state_dict(),
                        'decoder': decoder.state_dict(),
                    }
                    print(f"           ^ new best val BER, checkpoint saved")

                constellation.train()
                decoder.train()

        if step == max_steps - 1:
            print(constellation.normalized_points().detach())
        if step > max_steps:
            return data_points, np.array(loss_over_time)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return data_points, np.array(loss_over_time)
