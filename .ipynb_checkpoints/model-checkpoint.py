import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Constellation(nn.Module):
    
    def __init__(self, M=8, Es=1.0):
        super().__init__()
        self.M = M
        self.Es = Es
<<<<<<< Updated upstream

        
=======
        '''
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        
        #self.points = nn.Parameter(torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1], dtype=torch.float32))

=======
        '''
        self.points = nn.Parameter(torch.tensor([-2.5, -2.3, -0.9, -0.1, 0.1,  0.9,  2.3,  2.5], dtype=torch.float32))
>>>>>>> Stashed changes
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


class Decoder(nn.Module):

    def __init__(self, M=8, temperature=10.0):
        super().__init__()
        self.M = M
        self.temperature = temperature  # controls how "soft" vs "sharp" boundaries are during training
        
        # Init at standard midpoints as a reasonable starting guess
        '''
        init_boundaries = torch.tensor([
            -1.3093, -0.8729, -0.4364, 0.0, 0.4364, 0.8729, 1.3093
        ], dtype=torch.float32)
        '''
<<<<<<< Updated upstream
        init_boundaries = torch.tensor([-1.3, -0.75, -0.5, 0, 0.5, 0.6, 1.2], dtype=torch.float32)
=======
        init_boundaries = torch.tensor([-1.3661, -0.9108, -0.2846, 0.0, 0.2846, 0.9108, 1.3661], dtype=torch.float32)
>>>>>>> Stashed changes

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
