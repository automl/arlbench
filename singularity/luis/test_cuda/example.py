import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("Using device:", device)

A = torch.randn(500, 400).to(device)
B = torch.randn(400, 200).to(device)
C = torch.sum(torch.mm(A, B)).item()

print("Result:", C)
