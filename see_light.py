from PIL import Image
import torch
import torchvision.transforms as transforms
import NET_C
import os

transforms0 = transforms.Compose([transforms.Resize((75, 100)),
                                  transforms.ToTensor()])

for x in open(r'light_point_data\target.txt').readlines()[30:50]:
    module_cls = r'module\20190821M_C.pkl'
    path = os.path.join(r'light_point_data', x.split()[0])
    img = Image.open(path)
    # img.show()
    net_cls = NET_C.MAIN().cuda()
    net_cls.load_state_dict(torch.load(module_cls))
    net_c = net_cls.eval()

    img_data = transforms0(img) - 0.5
    img_data_4 = img_data.unsqueeze(0).cuda()
    img_data = net_c(img_data_4)[1]
    img_data = img_data.squeeze(0).detach().cpu()
    img = transforms.ToPILImage()(img_data.long().float())
    img.show()
    # img_data = net_c(img_data_4)[2]
    # img_data = img_data * 255
    # img_data = img_data.squeeze(0).detach().cpu()
    # img = transforms.ToPILImage()(img_data.long().float())
    # img.show()

# p = net_c(img_data_4)
# print(p)


# img = transforms.ToPILImage()(img_data.cpu().detach().numpy())
# img.show()
