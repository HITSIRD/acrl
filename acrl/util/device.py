import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.has_mps:
    device = torch.device('mps')
else:
    device = torch.device('cpu')
