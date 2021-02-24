import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Divisor(nn.Module):
    def __init__(self):
        super(Divisor, self).__init__()
        self.linear1 = nn.Linear(2, 20)
        self.linear2 = nn.Linear(20,1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        return x

div = Divisor()

criterion = nn.MSELoss()
optimizer = optim.Adam(div.parameters(), lr=0.0001)

# training loop
running_loss = 0.0

for step in range(10**6):

    optimizer.zero_grad()
    x = (torch.rand(2) - 0.5) * 200  # range [-100,100]
    ratio = x[0] / x[1]
    ratio = ratio.unsqueeze(0)
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
            print(f"Ratio: {y[0].item():.2f}/{y[1].item():.2f} | Right Answer: {r.item():.2f} | Net Output: {q.item():.2f}")

"""
TODO:
1. Surround training loop so that it happens multiple times
2. Display results (split into epochs and show results after each)
3. Add functionality to save model
"""