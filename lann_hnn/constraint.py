import torch

def compute_hnn_constraint(model, qp_batch, q_dot, p_dot, d_dim, requires_grad=True):
    q = qp_batch[:, :d_dim].clone().requires_grad_(requires_grad)
    p = qp_batch[:, d_dim:].clone().requires_grad_(requires_grad)
    H = model(q, p).sum()
    dH_dq, dH_dp = torch.autograd.grad(H, (q, p), create_graph=requires_grad)
    c1 = (q_dot - dH_dp).mean(0)
    c2 = (p_dot + dH_dq).mean(0)
    return torch.cat([c1, c2])
