import os
import torch
from torch.utils.data import Dataset
# 确保你的 torchvision 版本支持 transforms.v2
try:
    import torchvision.transforms.v2 as transformsV2
    from torchvision.transforms.v2 import functional as F_v2
    TV_V2_AVAILABLE = True
except ImportError:
    print("Warning: torchvision.transforms.v2 not available. Falling back to functional transforms approach.")
    # 如果 v2 不可用，需要 fallback 到方法二
    import torchvision.transforms as transforms
    from torchvision.transforms import functional as F
    TV_V2_AVAILABLE = False
import torchvision.datasets as datasets
import torchvision.transforms as transforms # Keep standard transforms for non-geometric ones
from PIL import Image
import torch.nn as nn
import random # For fallback method


CUSTOM_DATASET_MEAN = [0.330694]
CUSTOM_DATASET_STD = [0.176989]

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
                 apply_blur_to_input,
                 kernel_size,
                 sigma,
                 img_size=224,
                 random_crop_scale=(0.05, 0.5),
                 random_crop_ratio=(3./4., 4./3.),
                 hflip_prob=0.5,
                 vflip_prob=0.0, # Usually off
                 color_jitter_strength=0.0
                 ):

        self.img_dir = imgs_path
        if not os.path.isdir(self.img_dir): raise FileNotFoundError(f"Directory not found: {self.img_dir}")
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])
        if not self.img_list: raise ValueError(f"No images found in {self.img_dir}")

        self.apply_blur_to_input = apply_blur_to_input
        self.img_size = img_size

        # --- Define Non-Geometric Transforms ---
        # Blur (applied only to data if needed)
        self.blur_transform = transforms.GaussianBlur(kernel_size, sigma=sigma) if apply_blur_to_input else nn.Identity()
        # Color Jitter (applied only to data if needed)
        self.color_jitter = transforms.ColorJitter(
            brightness=color_jitter_strength, 
            contrast=color_jitter_strength
        ) if color_jitter_strength > 0 else nn.Identity()
        # ToTensor and Normalize (applied to both data and gt)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=CUSTOM_DATASET_MEAN, std=CUSTOM_DATASET_STD)

        # --- Define Geometric Transforms (Using v2 if available) ---
        if TV_V2_AVAILABLE:
            # v2 transforms handle synchronization automatically when called
            self.geometric_transform = transformsV2.Compose([
                transformsV2.RandomResizedCrop(img_size, 
                                               scale=random_crop_scale, 
                                               ratio=random_crop_ratio, 
                                               interpolation=transforms.InterpolationMode.BICUBIC),
                transformsV2.RandomHorizontalFlip(p=hflip_prob),
                transformsV2.RandomVerticalFlip(p=vflip_prob),
                # Important: Wrap ToTensor and Normalize for v2 compatibility if needed AFTER geometric
                # Or apply them manually later. Here we apply manually later.
            ])
            print("Using torchvision.transforms.v2 for synchronized geometric transforms.")
        else:
            # Fallback: Store parameters for functional application
            self.random_crop_scale = random_crop_scale
            self.random_crop_ratio = random_crop_ratio
            self.hflip_prob = hflip_prob
            self.vflip_prob = vflip_prob
            print("Using functional transforms for geometric synchronization (torchvision.transforms.v2 recommended).")

        print(f"Train Dataset (Spatial Sync): {imgs_path}, Input Blur: {apply_blur_to_input}")

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        try:
            img_original = Image.open(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}") from e

        # --- Apply Geometric Transforms Synchronously ---
        if TV_V2_AVAILABLE:
            # Apply the same geometric transform to the original image once
            # Note: v2 Compose expects and returns tensors by default for some ops,
            # but RandomResizedCrop etc. work on PIL. Check v2 docs carefully.
            # A safer way might be to apply to PIL then convert.
            # Let's assume self.geometric_transform is defined to work on PIL for simplicity here:
            img_geom_transformed = self.geometric_transform(img_original)

            # Create two copies for data and gt BEFORE applying non-geometric transforms
            data_intermediate = img_geom_transformed
            gt_intermediate = img_geom_transformed.copy() # Make a copy if transforms modify in-place (PIL usually doesn't)

        else: # Fallback using functional transforms
            # 1. RandomResizedCrop parameters
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img_original, scale=self.random_crop_scale, ratio=self.random_crop_ratio
            )
            # Apply the *same* crop to both data and gt paths initially
            img_cropped = F.resized_crop(img_original, i, j, h, w, [self.img_size, self.img_size], interpolation=transforms.InterpolationMode.BICUBIC)

            # 2. RandomHorizontalFlip state
            apply_hflip = random.random() < self.hflip_prob
            # Apply the *same* flip state
            img_hflipped = F.hflip(img_cropped) if apply_hflip else img_cropped

            # 3. RandomVerticalFlip state
            apply_vflip = random.random() < self.vflip_prob
            # Apply the *same* flip state
            img_geom_transformed = F.vflip(img_hflipped) if apply_vflip else img_hflipped

            # Create intermediate copies
            data_intermediate = img_geom_transformed
            gt_intermediate = img_geom_transformed.copy()


        # --- Apply Non-Geometric Transforms Separately ---
        # Apply blur and jitter only to 'data' path (still PIL Images at this stage)
        data_intermediate = self.blur_transform(data_intermediate)
        data_intermediate = self.color_jitter(data_intermediate)

        # Apply ToTensor and Normalize to both
        data = self.normalize(self.to_tensor(data_intermediate))
        gt = self.normalize(self.to_tensor(gt_intermediate)) # gt is based on the same geom transform but without blur/jitter

        return data, gt

    def __len__(self):
        return len(self.img_list)


class MAEDatasetEval(Dataset):
    def __init__(self,
                 imgs_path,
                 apply_blur_to_input,
                 kernel_size,
                 sigma,
                 img_size=224):

        self.img_dir = imgs_path
        if not os.path.isdir(self.img_dir): raise FileNotFoundError(f"Directory not found: {self.img_dir}")
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])
        if not self.img_list: raise ValueError(f"No images found in {self.img_dir}")

        # Define the single transform pipeline for evaluation
        eval_transforms_list = [
            #  transforms.Resize(int(img_size * 256 / 224), 
            #                    interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(img_size),
             transforms.GaussianBlur(kernel_size, sigma=sigma) if apply_blur_to_input else nn.Identity(), # Blur applied to input if needed
             transforms.ToTensor(),
             transforms.Normalize(mean=CUSTOM_DATASET_MEAN, std=CUSTOM_DATASET_STD)
        ]
        self.eval_input_transform = transforms.Compose(eval_transforms_list)

        # GT transform is the same deterministic crop without blur
        self.eval_target_transform = transforms.Compose([
            #  transforms.Resize(int(img_size * 256 / 224), 
            #                    interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(img_size),
             transforms.ToTensor(),
             transforms.Normalize(mean=CUSTOM_DATASET_MEAN, std=CUSTOM_DATASET_STD)
        ])

        print(f"Eval Dataset (Spatial Sync): {imgs_path}, Input Blur: {apply_blur_to_input}")


    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        try:
            img_original = Image.open(img_path).convert('L')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}") from e

        data = self.eval_input_transform(img_original) # Potentially blurred input
        gt = self.eval_target_transform(img_original)  # Always clear GT from same crop

        return data, gt

    def __len__(self):
        return len(self.img_list)
# --- End Placeholder Dataset ---


# -- for ImageNet-1k --
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # ImageNet

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

valid_transforms = transforms.Compose([
    transforms.Resize(int(224 * 256 / 224),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

class MAEImageNetDataset(Dataset):
    def __init__(self, root, train=True, target_transform=None):
        # 使用 ImageFolder 来处理文件查找和基本加载
        self.image_folder = datasets.ImageFolder(root, transform=None) # 初始不应用 transform
        self.transform = train_transforms if train else valid_transforms
        self.target_transform = target_transform # 用于生成目标 gt 的 transform (可能与 transform 不同)

    def __getitem__(self, index):
        # 从 ImageFolder 获取原始图像路径和标签 (标签在这里不用)
        path, _ = self.image_folder.samples[index]
        # 加载原始图像
        original_img = self.image_folder.loader(path) # 使用 ImageFolder 的加载器

        # 应用不同的 transform 生成 data 和 gt
        # 注意：需要确保 transform 返回的是 Tensor
        data = self.transform(original_img)
        gt = self.target_transform(original_img) if self.target_transform else data # 如果没有 target_transform，gt 可以等于 data

        # MAE 通常需要 (data, gt)
        return data, gt

    def __len__(self):
        return len(self.image_folder)