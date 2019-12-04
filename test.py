import torch.nn as nn
import torch

cnn = nn.Conv2d(3, 16, 3, 1, 1)
x = torch.rand(10, 3, 75, 100)
drop = nn.Dropout2d(0.3)
print(drop(cnn(x)).parameters())