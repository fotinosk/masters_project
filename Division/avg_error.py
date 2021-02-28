# assuming that the network always outputs 0

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

# for 10000 steps

means = []
criterion = nn.MSELoss()

for k in range(5):
    running_loss = 0.0

    for j in tqdm(range(10000)):
        x = (torch.rand(100, 2) - 0.5) * 200  # range [-100,100]
        for i in range(len(x)):
            if abs(x[i,1]) < 0.000001:
                x = torch.cat([x[:i], x[i+1:]])
        ratio = x[:, 0] / x[:, 1]
        loss = criterion(ratio, torch.zeros_like(ratio))
        running_loss += loss.item()

    means.append(running_loss / 10000)


print(np.mean(means))