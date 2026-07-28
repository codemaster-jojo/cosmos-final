import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import Constellation, Decoder
from infrastructure import bits_to_symbol_indices, calculate_BER, pam_symbols_to_bits, pam_from_indices
import numpy as np

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

def trainer(dataloader, constellation, decoder, optimizer, loss_function, max_steps = 15000, report_every = 500):
    constellation.train() #sets the nn into training mode
    decoder.train()
    data_points = [] # to plot, not necessary elsewhere
    loss_over_time = [] # also to plot
    step = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5) #Cuts LR in half every 2000 steps to fine-tune
    # for batch,(symbol_indices, labels) in enumerate(dataloader):
    for symbol_indices, labels in dataloader:
        
        symbol_indices = symbol_indices.to(device)
        labels = labels.to(device)

        # returns the amplitudes of all our symbols (they are currently in all the indices)
        transmitted = constellation(symbol_indices)

        # # AWGN channel
        # Gaussian rn
        noise = 0.1 * torch.randn_like(transmitted)
        '''
        noise = torch.from_numpy(
            np.random.gamma(shape=2, scale=1, size=transmitted.shape)
        ).float().to(transmitted.device)
         
        noise -= torch.mean(noise)
        noise *= 0.1
        '''
        
        received = transmitted + noise
        data_points += received.tolist()

        # Neural decoder
        prediction = decoder(received)
        loss = loss_function(prediction, labels)
        
        
        step += 1
        if step % report_every == 0:
            loss_over_time.append(loss.item())
            
            constellation_np = constellation.normalized_points().detach().cpu().numpy()
            pam_symbols = pam_from_indices(constellation_np, symbol_indices)
            raw_bits = pam_symbols_to_bits(pam_symbols, constellation_np)

            received_symbols = decoder.decode_hard(received)
            received_np = received_symbols.detach().cpu().numpy()
            pam_symbols = pam_from_indices(constellation_np, received_np)
            raw_received_bits = pam_symbols_to_bits(pam_symbols, constellation_np)
            print(f"Step {step} | Loss: {loss.item():.3f} | BER: {calculate_BER(raw_received_bits, raw_bits):.3f}")
        if step == max_steps - 1:
            print(constellation.normalized_points())
            print(decoder.get_boundaries())
        if step > max_steps:
            return data_points, np.array(loss_over_time)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return data_points, np.array(loss_over_time)

# train_loss
# validation_loss
#Find best constellation by seeing which numbers the noise affects the least