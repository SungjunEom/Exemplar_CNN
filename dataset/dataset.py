from stl_preprocessing import *
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self):
        self.data_path = './data/stl10_binary/train_X.bin'
        self.label_path = './data/stl10_binary/train_y.bin'

    def __len__(self):
        pass

    def __getitem__(self):
        pass


if __name__ == '__main__':
    print(read_all_images('./data/stl10_binary/train_X.bin').shape)
    print(read_all_images('./data/stl10_binary/test_X.bin').shape)
    print(read_all_images('./data/stl10_binary/unlabeled_X.bin').shape)