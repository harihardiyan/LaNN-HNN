import torch
import torch.nn as nn

class HamiltonianNN(nn.Module):
    def __init__(self, d_dim=1, hidden_dim=64, depth=4):
        super().__init__()
        self.d_dim = d_dim
        layers = []
        layers += [nn.Linear(2*d_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, q, p):
        # q: [B, d], p: [B, d]
        qp = torch.cat([q, p], dim=-1)
        return self.net(qp)   # [B, 1]
