import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms

transforms = transforms.Compose([transforms.ToTensor()])


class Dataset(data.Dataset):
    def __init__(self, path):
        self.path = path
        file = open(os.path.join(path, r'target.txt'))
        self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        img = Image.open(os.path.join(self.path, line.split()[0]))
        img_data = transforms(img)-0.5
        target = line.split()[1]
        return img_data, int(target)


if __name__ == '__main__':

    dataset = Dataset(r'test')
    data = data.DataLoader(dataset, batch_size=100, shuffle=True)
    for i, tar in data:
        print(i.shape)