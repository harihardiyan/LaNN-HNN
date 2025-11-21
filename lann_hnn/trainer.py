import torch

def augmented_lagrangian(L_theta, c, lam, mu):
    return L_theta + torch.sum(lam * c) + (mu / 2) * torch.sum(c ** 2)
