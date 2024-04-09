import torch

device_type = 'cpu'
# device_type = 'cuda:0'

# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
# elif torch.backends.mps.is_built():
#     device = torch.device('mps')
# else:
device = torch.device(device_type)
