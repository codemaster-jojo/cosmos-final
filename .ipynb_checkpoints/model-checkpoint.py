import torch
<<<<<<< Updated upstream
import torch.nn as nn
=======
from torch import nn
>>>>>>> Stashed changes
from torch.utils.data import DataLoader

class Constellation(nn.Module):
    
    def __init__(self, M=8, Es=1.0):
        super().__init__()
        self.M = M
        self.Es = Es
    
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

    def normalized_points(self):
        energy = (self.points ** 2).mean()
        scale = torch.sqrt(self.Es / energy)
        return self.points * scale


    def forward(self, symbol_indices):
        pts = self.normalized_points()
        return pts[symbol_indices]


<<<<<<< Updated upstream
class Decoder(nn.Module):

    def __init__(self, M=8, hidden_dim = 32):
        super().__init__()
        self.M = M

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, M)
        )

    def forward(self, received_values):
        x = received_values.unsqueeze(-1)
        logits = self.net(x)
        return logits


decoder = Decoder()
fake_received = torch.tensor([-1.4, -0.7, 0.1, 0.9, 1.6])
logits = decoder(fake_received)
probs = torch.softmax(logits, dim=-1)
predicted_symbols = torch.argmax(probs, dim=-1)
confidence = torch.max(probs, dim=-1).values
print("Predicted symbols:", predicted_symbols)
print("Confidence:", confidence)
=======
model = Constellation()

symbols = torch.tensor([0, 2, 5, 7])

print(model(symbols))
>>>>>>> Stashed changes
