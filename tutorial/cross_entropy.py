import torch
import torch.nn as nn
from torch.nn import functional as F

from rich import print as pprint

y = torch.tensor([1, 2, 3], dtype=torch.float32)
p = y.softmax(dim=0)
t1 = torch.tensor([1, 0, 0], dtype=torch.float32)
t2 = torch.tensor([0, 1, 0], dtype=torch.float32)
t3 = torch.tensor([0, 0, 1], dtype=torch.float32)

pprint(p)

for t in [t1]:
    cross_entropy = -torch.sum(t * torch.log(p)) # F.cross_entropy(y, t)

    pprint(t, 'x', torch.log(p), '=', t * torch.log(p))
    pprint(cross_entropy)


pprint(torch.log(torch.arange(0.0, 1.1, 0.1)))
pprint(torch.log(torch.tensor([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0])))