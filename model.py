import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Constellation(nn.Module):
    
    def __init__(self, M=8, Es=1.0):
        super().__init__()
        self.M = M
        self.Es = Es
        '''
        self.points = nn.Parameter(
            torch.tensor([
                -1.5275252316519465,
                -1.0910894511799618,
                -0.6546536707079771,
                -0.21821789023599236,
                 0.21821789023599236,
                 0.6546536707079771,
                 1.0910894511799618,
                 1.5275252316519465
            ], dtype=torch.float32)
        )
        '''      
        self.points = nn.Parameter(torch.tensor([-2.5, -2.3, -0.9, -0.1, 0.1,  0.9,  2.3,  2.5], dtype=torch.float32))
    def normalized_points(self):
        energy = (self.points ** 2).mean()
        scale = torch.sqrt(self.Es / energy)
        return self.points * scale


    def forward(self, symbol_indices):
        pts = self.normalized_points()
        return pts[symbol_indices]

    def set_points(self, new_points):
        #Allows external optimizer/LLM to modify constellation.
        with torch.no_grad():
            self.points.copy_(
                torch.tensor(
                    new_points,
                    dtype=torch.float32)
            )


class NoiseMDN(nn.Module):
    def __init__(self, hidden=64, K=3): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(), #Rectified Linear Unit (makes sure not everything is linear)
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        #THREE SEPARATE LAYERS
        self.weight_head = nn.Linear(hidden, K) #How important each Gaussian is, output K weights
        self.mean_head = nn.Linear(hidden, K) #Where is the center of the Gaussian
        self.logstd_head = nn.Linear(hidden, K) #How uncertain each Gaussian is-High log standard deviation=flat distribution

    def forward(self, transmitted): #MAKES THE NOISE PARAMETERS (WEIGHTS, MEANS, STDS)
        h = self.net(transmitted.unsqueeze(-1)) #(B,) -> (B, 1) so Linear sees one input feature
        weights = torch.softmax(self.weight_head(h), dim=-1) #softmax converts numbers into a probability distribution
        means = self.mean_head(h)
        stds = torch.nn.functional.softplus(self.logstd_head(h)) + 1e-3  # keep std > 0, avoid collapse, softplus = better log
        return weights, means, stds

    def sample(self, transmitted):
        weights, means, stds = self.forward(transmitted)
        mixture_mean = (weights * means).sum(dim=-1) #Each Gaussian mean is multiplied by its importance to find center of the mix. Sums this for every single symbol, so if transmitted has 1000 symbols, size [1000]
        mixture_variance = (weights * (stds**2 + means**2)).sum(dim=-1) - mixture_mean**2
        #Variance = E[X^2] - (E[X])^2. E[X] = mixture_mean
    
        mixture_variance = torch.clamp(mixture_variance, min=1e-6) #Keep everything positive
        random_nums = torch.randn_like(mixture_mean) #Generates Gaussian Distribution Matrix the size of mixture_mean
        received = mixture_mean + torch.sqrt(mixture_variance) * random_nums
        return received


class Decoder(nn.Module):
    def __init__(self, constellation, temperature=10.0):
        super().__init__()

        self.constellation = constellation
        self.temperature = temperature
        
    def get_boundaries(self):
        points = self.constellation.normalized_points()
        boundaries = (
            points[:-1] + points[1:]
        ) / 2
        return boundaries
        
    def forward(self, received_values):
        boundaries = self.get_boundaries()
        soft_indicators = torch.sigmoid(
            (
                received_values.unsqueeze(-1)
                - boundaries.unsqueeze(0)
            ) * self.temperature
        )
        soft_symbol_position = soft_indicators.sum(dim=-1)
        return soft_symbol_position

    def decode_hard(self, received_values):
        with torch.no_grad():
            boundaries = self.get_boundaries()
            predicted_symbols = torch.searchsorted(
                boundaries,
                received_values
            )
        return predicted_symbols
