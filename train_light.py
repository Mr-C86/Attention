import torch
import torch.nn as nn
from torch.utils import data
import os
import NETS
import MKDataset


def trainer():

    path = r'light_point_data'
    module_cls = r'module\20190820.pkl'
    module_light = r'module\20190820_light.pkl'
    batch = 1

    net_ligth = NETS.light_point_net().cuda()
    net_cls = NETS.ClsNet().cuda()
    optimer = torch.optim.Adam(net_ligth.parameters())
    loss_func = nn.MSELoss()
    dataset = MKDataset.Dataset(path)

    dataloader = data.DataLoader(dataset, batch_size=256, shuffle=True)

    if os.path.exists(module_cls):
        net_cls.load_state_dict(torch.load(module_cls))
        print('cls_module is loaded !')

    if os.path.exists(module_light):
        net_ligth.load_state_dict(torch.load(module_light))
        print('light_module is loaded !')

    while True:
        for i, (img_data, target) in enumerate(dataloader):
            target = target.float().cuda()
            img_data = img_data.cuda()
            w = net_ligth(img_data)
            out_ = torch.zeros(w.shape).cuda()
            out_[w >= 0.3] = 1
            out_[w < 0.3] = 0
            img_data = ((img_data + 0.5) * 255 * w) / 255 - 0.5
            y = net_cls(img_data)
            y = y.reshape(-1)
            loss = loss_func(y, target)

            optimer.zero_grad()
            loss.backward()
            optimer.step()
        print('batch: {} is completed!'.format(batch))
        batch += 1

        torch.save(net_ligth.state_dict(), module_light)
        acc = (y.round() == target.float().cuda()).float().mean()
        print('loss: {} acc: {}'.format(loss, acc))
        print('module is saved!')


if __name__ == '__main__':
    trainer()