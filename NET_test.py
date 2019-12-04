import torch.nn as nn
import torch
import os
import torch.utils.data as data
import MKDataset


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2),
            nn.ReLU()
        )
        self.lin = nn.Linear(16 * 37 * 49, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        layer = self.layers(x)
        x = layer.reshape(-1, 16 * 37 * 49)
        lin = self.lin(x)
        out = self.sig(lin)
        return out

def trainer():
    path = r'light_point_data'
    module = r'module\20190821_NONA.pkl'
    batch = 1

    net = NET().cuda()
    optimer = torch.optim.Adam(net.parameters())
    A_loss_func = nn.MSELoss()
    dataset = MKDataset.Dataset(path)

    dataloader = data.DataLoader(dataset, batch_size=256, shuffle=True)

    if os.path.exists(module):
        net.load_state_dict(torch.load(module))
        print('module is loaded !')

    while True:
        for i, (img_data, target) in enumerate(dataloader):
            img_data = img_data.cuda()
            y = net(img_data)
            y = y.reshape(-1)
            loss = A_loss_func(y, target.float().cuda())

            optimer.zero_grad()
            loss.backward()
            optimer.step()
        print('batch: {} is completed!'.format(batch))
        batch += 1

        torch.save(net.state_dict(), module)
        acc_A = (y.round() == target.float().cuda()).float().mean()
        print('loss_A: {} acc_A: {}'.format(loss, acc_A))
        print('module is saved!')


if __name__ == '__main__':
    trainer()