import os
import random

# import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image
from pathlib import Path


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(train_dir, test_dir, cr, ycrcb=False, name=False, defocus=True):

    train_dataset = DATADataset(train_dir, cr=cr, ycrcb=ycrcb, name=name, defocus=defocus)
    test_dataset = DATADataset(test_dir, cr=cr, ycrcb=ycrcb, name=name, defocus=defocus)

    print(f"Do defocus SCI is -{defocus}-")

    return train_dataset, test_dataset


class DATADataset(Dataset):
    def __init__(self, imgs_path, cr, ycrcb=False, name=False, defocus=True):
        self.ycrcb = ycrcb
        self.name = name
        self.defocus = defocus

        if self.defocus:
            # if do defocus sci task
            self.img_path = os.path.join(imgs_path, 'defocus')
            self.gt_path = os.path.join(imgs_path, 'gt')
            self.gt_list = sorted(os.listdir(self.gt_path))
        else:
            self.img_path = os.path.join(imgs_path, 'gt')

        self.img_list = sorted(os.listdir(self.img_path))
        
        # self.cr = cr
        # torch.manual_seed(42)
        self.tfs = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.img_list[index])
        # name = Path(path).stem
        img = Image.open(img_path)
        img = self.tfs(img)

        if self.defocus:
            gt_path = os.path.join(self.gt_path, self.gt_list[index])
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


# -- for dota v1 train --
# 在你的 PyTorch Dataset 或 DataLoader 中使用 transforms.Normalize:
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.321886], std=[0.179375])
# ])

class MAEDataset(Dataset):
    def __init__(self, 
                 imgs_path, 
                 blur, 
                 kernel_size, 
                 sigma, 
                 img_size=224, 
                 random_crop_scale=(0.05, 0.5)):
        
        # self.img_dir = os.path.join(imgs_path, 'gt') # Assuming 'gt' subdirectory exists
        self.img_dir = imgs_path
        if not os.path.isdir(self.img_dir):
             raise FileNotFoundError(f"Directory not found: {self.img_dir}. Ensure imgs_path points to parent of 'gt'.")
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])
        if not self.img_list:
            raise ValueError(f"No images found in {self.img_dir}")

        self.tfs = transforms.Compose([
            # transforms.CenterCrop(img_size),
            # transforms.RandomCrop(img_size),
            transforms.RandomResizedCrop(
                img_size, 
                scale=random_crop_scale, 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.330694], std=[0.176989]), # mean and var for sim_v2_train_gt!
        ])
        # Apply blur transformation only if blur is True
        self.blur_transform = transforms.GaussianBlur(kernel_size, sigma=sigma) if blur else transforms.Lambda(lambda x: x)
        print(f"Dataset initialized for {imgs_path}. Found {len(self.img_list)} images. Blur: {blur}")

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        try:
            img_clean = Image.open(img_path)
        except Exception as e:
            print(f"Error opening or converting image {img_path}: {e}")

        img_clean_tensor = self.tfs(img_clean)
        img_input = self.blur_transform(img_clean_tensor) # Apply blur to the tensor

        return img_input, img_clean_tensor

    def __len__(self):
        return len(self.img_list)


class MAEDatasetEval(Dataset):
    def __init__(self, imgs_path, blur, kernel_size, sigma, img_size=224):
        # self.img_dir = os.path.join(imgs_path, 'gt') # Assuming 'gt' subdirectory exists
        self.img_dir = imgs_path
        if not os.path.isdir(self.img_dir):
             raise FileNotFoundError(f"Directory not found: {self.img_dir}. Ensure imgs_path points to parent of 'gt'.")
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])
        if not self.img_list:
            raise ValueError(f"No images found in {self.img_dir}")

        self.tfs = transforms.Compose([
            # transforms.RandomCrop(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.330694], std=[0.176989]), # mean and var for sim_v2_train_gt!
        ])
        self.blur_transform = transforms.GaussianBlur(kernel_size, sigma=sigma) if blur else transforms.Lambda(lambda x: x)
        print(f"Eval Dataset initialized for {imgs_path}. Found {len(self.img_list)} images. Blur: {blur}")

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        try:
            img_clean = Image.open(img_path)
        except Exception as e:
            print(f"Error opening or converting image {img_path}: {e}")

        img_clean_tensor = self.tfs(img_clean)
        img_input = self.blur_transform(img_clean_tensor)

        return img_input, img_clean_tensor

    def __len__(self):
        return len(self.img_list)
# --- End Placeholder Dataset ---