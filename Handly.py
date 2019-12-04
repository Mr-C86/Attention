import os
from PIL import Image
import numpy as np


def handly(path):
    i = 0
    j = 1
    k = 0
    cls = '0'
    name = 'cat'
    save_path = r'light_point_data'
    target = r'light_point_data\target.txt'
    file = open(target,'w')

    while j:
        try:
            get = name + '.{}.jpg'.format(i)
            img = Image.open(os.path.join(path, get))
            i += 1
        except FileNotFoundError:
            if name.split('.')[0] == 'cat':
                name = 'dog'
                k = 0
                i = 0
                cls = '1'
            else:
                j = 0,
        else:
            img_data = np.array(img)
            print(img_data.shape)
            h, w, c = img_data.shape

            if w > h:
                save = name + '.{}.jpg'.format(k)
                img.resize((200, 150)).save(os.path.join(save_path, save))
                file.write(save+' {}\n'.format(cls))
                k += 1
    file.close()


if __name__ == '__main__':
    path = r'D:\kaggle\train'
    handly(path)