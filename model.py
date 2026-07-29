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
    def __init__(self, num_symbols=8, embed_dim=3, hidden=64, K=3): 
        #K is how many Gaussians we want to sum up to find the probability distribution for the noise of a given number
        super().__init__()
        self.symbol_embed = nn.Embedding(num_symbols, embed_dim)
        #Embedding: if I directly input symbol indices, the Sequential will treat index 7 > index 5. Embedding assigns each symbol to a random vector that is parametrized and learnable, and will con/diverge based on the similarity of their Gaussians
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(), #Rectified Linear Unit (makes sure not everything is linear)
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        #THREE SEPARATE LAYERS
        self.weight_head = nn.Linear(hidden, K) #How important each Gaussian is, output K weights
        self.mean_head = nn.Linear(hidden, K) #Where is the center of the Gaussian
        self.logstd_head = nn.Linear(hidden, K) #How uncertain each Gaussian is-High log standard deviation=flat distribution

    def forward(self, symbol_indices): #MAKES THE NOISE PARAMETERS (WEIGHTS, MEANS, STDS)
        h = self.net(self.symbol_embed(symbol_indices))
        weights = torch.softmax(self.weight_head(h), dim=-1) #softmax converts numbers into a probability distribution
        #softmax is needed because to apply the weights, they have to follow the rules of a probability distribution
        means = self.mean_head(h)
        stds = torch.nn.functional.softplus(self.logstd_head(h)) + 1e-3  # keep std > 0, avoid collapse, softplus = better log
        return weights, means, stds

    def sample(self, symbol_indices): #RETURNS THE SYMBOLS WITH THE NOISE [1, 2, 3, 4, 5]
        with torch.no_grad():
            weights, means, stds = self.forward(symbol_indices) # all three look like [[1, 3, 2, 4, 5]] (2D!!!)
            k = torch.multinomial(weights, num_samples=1).squeeze(-1) #randomly picks one element with the weights as probs
            #weights is a tensor like [0.1, 0.7, 0.2], and picking one sample would have high likelyhood of returning idx 1
            # [1]
            chosen_mean = means.gather(1, k.unsqueeze(1)).squeeze(1) #gather looks along dim=1 (columns) and picks the kth item
            # [[1, 3, 3, 2, 4]].gather(1, [[k]]) will return the kth item on the (only) column. Squeezing turns [[n]] to [n]
            chosen_std = stds.gather(1, k.unsqueeze(1)).squeeze(1)
            #Unsqueezed and then squeezed to satisfy dimension requirements, see above^^
            return torch.normal(chosen_mean, chosen_std) #Picks random item in the normal distribution given the parameters

class Decoder(nn.Module):
    def __init__(self, M=8, temperature=10.0):
        super().__init__()
        self.M = M
        self.temperature = temperature  # controls how "soft" vs "sharp" boundaries are during training
        '''
        # Init at standard midpoints as a reasonable starting guess
        init_boundaries = torch.tensor([
            -1.3093, -0.8729, -0.4364, 0.0, 0.4364, 0.8729, 1.3093
        ], dtype=torch.float32)
        '''
        init_boundaries = torch.tensor([-1.3661, -0.9108, -0.2846, 0.0, 0.2846, 0.9108, 1.3661], dtype=torch.float32)

        self.raw_boundaries = nn.Parameter(init_boundaries)

    def get_boundaries(self):
        # Sort to guarantee increasing order, regardless of how gradients move them individually
        return torch.sort(self.raw_boundaries).values

    def forward(self, received_values):
        boundaries = self.get_boundaries()  # shape (M-1,)
        soft_indicators = torch.sigmoid(
            (received_values.unsqueeze(-1) - boundaries.unsqueeze(0)) * self.temperature
        )  

        # Summing these gives a smooth, differentiable approximation of "how many boundaries has this value passed"
        # but now as a continuous value instead of a hard integer
        soft_symbol_position = soft_indicators.sum(dim=-1)  

        return soft_symbol_position  # continuous prediction, differentiable

    def decode_hard(self, received_values):
        # Use this at eval time / real decoding, no gradients needed here
        with torch.no_grad():
            boundaries = self.get_boundaries()
            predicted_symbols = torch.searchsorted(boundaries, received_values)
        return predicted_symbols
