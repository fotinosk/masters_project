import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

batch_size = 100

class Divisor(nn.Module):
    def __init__(self):
        super(Divisor, self).__init__()
        self.linear1 = nn.Linear(2, 20)
        self.linear2 = nn.Linear(20,1)

    def forward(self, x):
        x = self.linear1(x)
        # print('l1 ', x)
        x = torch.tanh(x)
        # print('tanh ', x)
        x = self.linear2(x)
        # print('l2 ', x)
        return x

div = Divisor()

criterion = nn.MSELoss()
optimizer = optim.Adam(div.parameters(), lr=0.0001)

# training loop
running_loss = 0.0

for step in range(10**8):

    optimizer.zero_grad()
    x = (torch.rand(batch_size, 2) - 0.5) * 200  # range [-100,100]
    ratio = x[:, 0] / x[:, 1]
    for i in range(len(x)):
        if abs(x[i,1]) < 0.000001:
            x = torch.cat([x[:i], x[i+1:]])
    ratio = ratio.unsqueeze(1)
    z = div(x)
    loss = criterion(z, ratio)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()


    if (step+1) % 10000 == 0:
        print(f"Epoch {step // 10000 + 1}| Average Loss: {running_loss / 10000:.2f}")
        running_loss = 0.0
        print("Now displaying some results")
        for ex in range(10):
            y = (torch.rand(2) - 0.5) * 200
            r = y[0] / y[1]
            q = div(y)
            print(f"Right Answer: {r.item():.2f} | Net Output: {q.item():.2f}")
        print("\n\n")

