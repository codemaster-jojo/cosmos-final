import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import Constellation, Decoder
from infrastructure import bits_to_symbol_indices
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

def trainer(dataloader, constellation, decoder, optimizer, loss_function, max_steps = 15000, report_every = 1000):
    constellation.train() #sets the nn into training mode
    decoder.train()
    data_points = [] # to plot, not necessary elsewhere
    step = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5) #Cuts LR in half every 2000 steps to fine-tune
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step % report_every == 0:
            print(f"Step {step} | Loss: {loss.item()}")
            print(constellation.normalized_points())
            print(decoder.get_boundaries())

    return data_points