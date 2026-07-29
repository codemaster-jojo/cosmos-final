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


import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Dataset ---
class NoiseDataset(Dataset):
    def __init__(self, symbol_indices, noise_values):
        self.symbol_indices = torch.tensor(symbol_indices, dtype=torch.long)
        self.noise_values = torch.tensor(noise_values, dtype=torch.float32)
    def __len__(self):
        return len(self.symbol_indices)
    def __getitem__(self, idx):
        return self.symbol_indices[idx], self.noise_values[idx]


# --- Loss ---
def mdn_loss(weights, means, stds, target):
    target = target.unsqueeze(-1)
    log_probs = torch.distributions.Normal(means, stds).log_prob(target)
    weighted = torch.log(weights + 1e-8) + log_probs
    return -torch.logsumexp(weighted, dim=-1).mean()


def radio_trainer(model, train_loader, val_loader, optimizer, scheduler,
            max_epochs=20, patience=5, min_delta=1e-4, checkpoint_path="noise_mdn_best.pth"):
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(max_epochs):
        # --- training pass ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        for symbol_indices, noise_values in train_loader:
            symbol_indices = symbol_indices.to(device)
            noise_values = noise_values.to(device)

            weights, means, stds = model(symbol_indices)
            loss = mdn_loss(weights, means, stds, noise_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches
        train_loss_history.append(train_loss)

        # --- validation pass ---
        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for symbol_indices, noise_values in val_loader:
                symbol_indices = symbol_indices.to(device)
                noise_values = noise_values.to(device)
                weights, means, stds = model(symbol_indices)
                loss = mdn_loss(weights, means, stds, noise_values)
                val_running_loss += loss.item()
                val_batches += 1

        val_loss = val_running_loss / val_batches
        val_loss_history.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{max_epochs} | Train NLL: {train_loss:.4f} | "
              f"Val NLL: {val_loss:.4f} | LR: {current_lr:.6f}")

        # --- early stopping + best checkpoint ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}: no val improvement in {patience} epochs")
                break

    return np.array(train_loss_history), np.array(val_loss_history)


#PUT IN MAIN
if __name__ == "__main__":
    # symbol_indices: array of ints (0-7), noise_values: array of floats
    # symbol_indices, noise_values = load_your_million_samples(...)

    dataset = NoiseDataset(symbol_indices, noise_values)
    train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    model = NoiseMDN(num_symbols=8, embed_dim=8, hidden=64, K=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_hist, val_hist = trainer(model, train_loader, val_loader, optimizer, scheduler,
                                    max_epochs=30, patience=5)

    # reload best checkpoint (not necessarily the last epoch's weights)
    model.load_state_dict(torch.load("noise_mdn_best.pth", weights_only=True))
    print("Training complete. Best model loaded from checkpoint.")