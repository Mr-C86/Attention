import torch
import torch.nn as nn
from torch.utils import data
import os
import NET_C
import MKDataset


def trainer():
    path = r'light_point_data'
    module = r'module\20190821M_C_D.pkl'
    batch = 1

    net = NET_C.MAIN().cuda()
    optimer = torch.optim.Adam(net.parameters())
    A_loss_func = nn.MSELoss()
    # loss_func = nn.MSELoss()
    dataset = MKDataset.Dataset(path)

    dataloader = data.DataLoader(dataset, batch_size=256, shuffle=True)

    if os.path.exists(module):
        net.load_state_dict(torch.load(module))
        print('module is loaded !')

    while True:
        for i, (img_data, target) in enumerate(dataloader):
            img_data = img_data.cuda()
            y, _, _ = net(img_data)
            y = y.reshape(-1)
            # y_ = y_.reshape(-1)
            loss = A_loss_func(y, target.float().cuda())
            # loss_ = loss_func(y_, target.float().cuda())

            optimer.zero_grad()
            loss.backward()
            # loss_.backward()
            optimer.step()
        print('batch: {} is completed!'.format(batch))
        batch += 1

        torch.save(net.state_dict(), module)
        acc_A = (y.round() == target.float().cuda()).float().mean()
        # acc = (y_.round() == target.float().cuda()).float().mean()
        print('loss_A: {} acc_A: {}'.format(loss, acc_A))
        print('module is saved!')


if __name__ == '__main__':
    trainer()
