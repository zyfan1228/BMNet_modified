import os
import random

# import cv2
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from pathlib import Path


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(train_dir, test_dir, cr, ycrcb=False, name=False, seed=42):

    train_dataset = DATADataset(train_dir, cr=cr, ycrcb=ycrcb, name=name)
    test_dataset = DATADataset(test_dir, cr=cr, ycrcb=ycrcb, name=name)

    return train_dataset, test_dataset


class DATADataset(data.Dataset):
    def __init__(self, imgs_path, cr, ycrcb=False, name=False, defocus=True):
        self.img_path = os.path.join(imgs_path, 'defocus')
        self.gt_path = os.path.join(imgs_path, 'gt')
        self.img_list = sorted(os.listdir(self.img_path))
        self.gt_list = sorted(os.listdir(self.gt_path))

        self.cr = cr
        torch.manual_seed(42)
        self.tfs = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.ycrcb = ycrcb
        self.name = name
        self.defocus = defocus

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.img_list[index])
        if self.defocus:
            gt_path = os.path.join(self.gt_path, self.gt_list[index])
        # name = Path(path).stem

        img = Image.open(img_path)
        img = self.tfs(img)
        if self.defocus:
            gt = Image.open(gt_path)
            gt = self.tfs(gt)
        else:
            gt = img
        # img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # img_ycrcb = self.tfs(img_ycrcb)
        # img_y = img_ycrcb[0, :, :][None, :, :]

        # if self.ycrcb:
        #     img_y = [img_y, img_ycrcb]
        # if self.name:
        #     return img_y, name
        # else:
        #     return img_y

        return img, gt

    def __len__(self):
        return len(self.img_list)
