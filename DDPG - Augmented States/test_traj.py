from trajectory import Trajectory
from augment import Augment
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = Augment(3,2)

c = torch.tensor([1.,2.,3.]).to(device)
b = torch.tensor([4.,5.]).to(device)


for i in range(50):
    a(c)
    a.update(b)

print(a(c))


