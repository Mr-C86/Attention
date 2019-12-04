from torch import nn
import torch
import time


class light_point_net(nn.Module):
    def __init__(self):
        super(light_point_net, self).__init__()
        self.layers = nn.Sequential(
            nn.LSTM(225, 75, 1)
        )
        self.lstm = nn.LSTM(75, 75, 2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(-1, 225, 100).permute(0, 2, 1)
        layer, (_, _) = self.layers(x)
        lstm, (_, _) = self.lstm(layer)
        sig = self.sig(lstm)
        sig = sig.permute(0, 2, 1)
        batch = sig.shape[0]
        sig = sig.reshape(batch, 1, 75, 100)
        out = sig.expand(batch, 3, 75, 100)
        return out


class ClsNet(nn.Module):
    def __init__(self):
        super(ClsNet, self).__init__()
        self.lstm = nn.LSTM(225, 1, 1, batch_first=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(-1, 225, 100).permute(0, 2, 1)
        out, (_, _) = self.lstm(x)
        out = out[:, -1, :].reshape(-1, 1)
        out = self.sig(out)
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
        # output = self.cls(x)
        return A_output, attention, light


if __name__ == '__main__':
    net = MAIN().cuda()
    a = torch.rand(100, 3, 75, 100).cuda()
    y = torch.rand(100, 1).cuda()
    loss = nn.MSELoss()
    optimer = torch.optim.Adam(net.parameters())

    start = time.time()
    out = net(a)[0]
    loss = loss(out, y)
    optimer.zero_grad()
    loss.backward()
    optimer.step()
    end = time.time()
    print(end - start)
