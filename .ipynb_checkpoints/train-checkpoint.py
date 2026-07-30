import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import *
from infrastructure import bits_to_symbol_indices, calculate_BER, pam_symbols_to_bits, pam_from_indices
import numpy as np
from data import *

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

class MainDataset(Dataset):
    def __init__(self, features, labels, feature_dtype=torch.long):
        #feature_dtype defaults to long because Constellation/Embedding lookups need integer indices,
        #but the wire measurements are continuous amplitudes and must stay float
        self.features = torch.tensor(features, dtype=feature_dtype)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

def trainer(dataloader, constellation, decoder, noiseMDN, optimizer, loss_function, max_steps=1000, report_every=100):
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
        # gain_transmitted = torch.tensor(apply_gain(transmitted)).to(device)
        # noise = 0.1 * torch.randn_like(gain_transmitted)
        #noise = (0.5 + torch.abs(transmitted)**0.8) * 0.05 * torch.randn_like(transmitted)
        received = noiseMDN.sample(transmitted)
        
        # received = gain_transmitted + noise
        # received = transmitted + noise
        
        data_points += received.tolist()
        decoder.temperature = min(10, 5.0 + step * 0.05) 
        prediction = decoder(received)
        loss = loss_function(prediction, labels)
        recent_losses.append(loss.item())  # track every step's loss for averaging

        step += 1

        if step % report_every == 0:
            avg_loss = sum(recent_losses) / len(recent_losses)
            scheduler.step(avg_loss)  # let the scheduler react to the averaged, less noisy loss
            loss_over_time.append(avg_loss)
            recent_losses = []  # reset the window for the next report interval

            #CALCULATES BER
            constellation_np = constellation.normalized_points().detach().cpu().numpy()
            pam_symbols = pam_from_indices(constellation_np, symbol_indices)
            raw_bits = pam_symbols_to_bits(pam_symbols, constellation_np)
            received_symbols = decoder.decode_hard(received)
            received_np = received_symbols.detach().cpu().numpy()
            pam_symbols = pam_from_indices(constellation_np, received_np)
            raw_received_bits = pam_symbols_to_bits(pam_symbols, constellation_np)
            print(f"Step {step} | Avg Loss: {avg_loss:.5f} | BER: {calculate_BER(raw_received_bits, raw_bits):.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            #END OF LONGWINDED CALCULATION

            
            print(constellation.normalized_points())
            print(decoder.get_boundaries())
        
        if step > max_steps:
            return data_points, np.array(loss_over_time)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return data_points, np.array(loss_over_time)


# Loss
def mdn_loss(weights, means, stds, target):
    target = target.unsqueeze(-1)
    log_probs = torch.distributions.Normal(means, stds).log_prob(target)
    weighted = torch.log(weights + 1e-8) + log_probs
    return -torch.logsumexp(weighted, dim=-1).mean()

def radio_trainer(model, train_loader, val_loader, optimizer, scheduler, max_epochs = 20):
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(max_epochs):
        # training pass
        model.train()
        running_loss = 0.0
        n_batches = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            weights, means, stds = model(features)
            loss = mdn_loss(weights, means, stds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches
        train_loss_history.append(train_loss)

        # VALIDATE (LIKE A POP QUIZ after every epoch)
        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        with torch.no_grad(): #don't want to adjust gradient - just evaluate
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                weights, means, stds = model(features)
                loss = mdn_loss(weights, means, stds, labels)
                
                val_running_loss += loss.item()
                val_batches += 1

        val_loss = val_running_loss / val_batches
        val_loss_history.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{max_epochs} | Train NLL: {train_loss:.4f} | "
              f"Val NLL: {val_loss:.4f} | LR: {current_lr:.6f}")

        # --- early stopping + best checkpoint ---
        if val_loss < best_val_loss - 0.0001: #0.0001 checks for REAL improvement, not negligible decreases
            best_val_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= 5:
                print(f"Early stopping at epoch {epoch+1}: no val improvement in 5 epochs")
                break
    return np.array(train_loss_history), np.array(val_loss_history)