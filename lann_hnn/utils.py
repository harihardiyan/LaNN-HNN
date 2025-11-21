import torch
import numpy as np

def generate_sho_data(N=2048):
    t = np.random.uniform(0, 2*np.pi, N)
    q, p = np.cos(t), -np.sin(t)
    qp = torch.tensor(np.stack([q, p], 1), dtype=torch.float32)
    return qp, torch.tensor(p.reshape(-1,1)), torch.tensor(-q.reshape(-1,1))

# tambah double_pendulum_data(), henon_heiles_data() kalau mau
