from trajectory import Trajectory
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


t = Trajectory(3,2)

a = torch.tensor([1,2,3]).to(device)
b = torch.tensor([4,5]).to(device)

for i in range(50):
    t(a)
    # t.archive([a,b,a,i])

b = t(a) # list on tensors
print(torch.stack(b))