from torch import nn
import torch


class light_point_net(nn.Module):
    def __init__(self):
        super(light_point_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Dropout2d(0.5),
            nn.Sigmoid()
        )

    def forward(self, x):
        layer = self.layers(x)
        batch = layer.shape[0]
        out = layer.expand(batch, 3, 75, 100)
        return out


class ClsNet(nn.Module):
    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = nn.Conv2d(3, 16, 3, 2)
        self.drop = nn.Dropout2d(0.5)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(16 * 37 * 49, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        cnn = self.cnn(x)
        cnn = self.drop(cnn)
        cnn = self.relu(cnn)
        x = cnn.reshape(-1, 16 * 37 * 49)
        lin = self.lin(x)
        out = self.sig(lin)
        return out


class MAIN(nn.Module):
    def __init__(self):
        super(MAIN, self).__init__()
        self.light = light_point_net()
        self.cls = ClsNet()

    def forward(self, x):
        light = self.light(x)
        attention = (x + 0.5) * 255 * light
        A_output = self.cls((attention / 255) - 0.5)
        return A_output, attention, light


if __name__ == '__main__':
    cls = MAIN()
    a = torch.rand(10, 3, 75, 100)
    a, b, c = cls(a)
    print(a.shape, b.shape, c.shape)
