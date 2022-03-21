# -*- coding: utf-8 -*-
# @project：image_caption_base
# @author:caojinlei
# @file: dataload.py
# @time: 2022/02/11
import os
import torch
import h5py
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class CaptionDataset(Dataset):
    def __init__(self, data_folder, split_name, transform=None):
        self.split_name = split_name
        assert self.split_name in {'TRAIN', 'VAL', 'TEST'}, 'Please write in one of "TRAIN"|"VAL"|"TEST"'

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split_name + '_IMAGES' + '.hdf5'), 'r')
        self.images = self.h['image']
        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']
        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split_name + '_CAPTIONS_' + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split_name + '_CAPLENS_' + '.json'), 'r') as j:
            self.caplens = json.load(j)

        self.transform = transform

        self.dataset_size = len(self.captions)

    def __getitem__(self, item):
        img = torch.FloatTensor(self.images[item // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        caption = torch.LongTensor(self.captions[item])

        caplen = torch.LongTensor([self.caplens[item]])

        if self.split_name == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((item // self.cpi) * self.cpi):(((item // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    dataset = 'Flicker8k'
    data_folder = f'./inputs/{dataset}/'
    split_name = 'TRAIN'
    batch_size = 64
    shuffle = True
    workers = 1
    normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    my_dataset = CaptionDataset(data_folder, split_name,normalize)
    train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    for i, (img, caption, caplen) in enumerate(train_loader):
        if i==0:
            print(len(caplen))
            print(f'item:{i} caplen{caplen} ')
            break
