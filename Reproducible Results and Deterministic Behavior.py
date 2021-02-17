import torch
import numpy as np
import random

seed = 41
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#if using cuda

torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

x = torch.rand((5,5))
print(torch.einsum('ii->',x))