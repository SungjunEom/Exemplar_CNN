from stl_preprocessing import *
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch


class TrainDataset(Dataset):
    def __init__(self, k=150, n=8000, data_path='./data/stl10_binary/unlabeled_X.bin'):
        self.data_path = data_path
        self.all_images = read_all_images(self.data_path)
        self.img_w = self.all_images.shape[-1]
        self.img_h = self.all_images.shape[-2]
        self.k = k
        self.n = n

    def __len__(self):
        return self.all_images.shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(self.all_images[idx])
        imgs = [self.transform(img) for _ in range(self.k)]
        return imgs

    def transform(self, img):
        adjust_color = v2.ColorJitter()
        rotate = v2.RandomRotation(20)
        resize = v2.RandomResize(self.img_w * 0.7, self.img_w * 1.4)
        crop = v2.RandomCrop((32,32))

        colored = adjust_color(img)
        rotated = rotate(colored)
        resized = resize(rotated)
        img = crop(resized)

        return img
        
    
class TestDataset(Dataset):
    def __init__(
            self, 
            data_path='./data/stl10_binary/test_X.bin',
            label_path='./data/stl10_binary/test_y.bin'
            ):
        self.data_path = data_path
        self.label_path = label_path
        self.all_images = read_all_images(self.data_path)

    def __len__(self):
        return self.all_images.shape[0]

    def __getitem__(self, idx):
        return self.all_images[idx]


if __name__ == '__main__':
    print('train:',read_all_images('./data/stl10_binary/train_X.bin').shape)
    print('test:',read_all_images('./data/stl10_binary/test_X.bin').shape)
    print('unlabeled:',read_all_images('./data/stl10_binary/unlabeled_X.bin').shape)