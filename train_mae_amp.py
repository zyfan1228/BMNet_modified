import os
import subprocess
import time
import shutil
import datetime
import argparse
import random
import sys

import torch
import numpy as np
from tqdm import tqdm
import tensorboardX
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from timm.scheduler import CosineLRScheduler
from PIL import Image
# from functools import partial

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- AMP: Import GradScaler and autocast ---
from torch.cuda.amp import GradScaler, autocast

# Import your models
from model import mae_vit_tiny_mine, mae_vit_base_patch16

# -- Your MAEImageNetDataset Definition (Keep as is) --
# ... (MAEImageNetDataset class definition remains exactly the same) ...
# Helper function to check if code is running in the main process (useful for DDP)
def is_main_process():
    # Simple check, might need adjustment based on your DDP setup
    # (e.g., using torch.distributed.get_rank() == 0 if initialized)
    import torch.distributed as dist
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

class MAEImageNetDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 img_size=224,
                 apply_blur_to_input=False,
                 kernel_size=3,
                 sigma=0.5,
                 random_crop_scale=(0.2, 1.0),
                 random_crop_ratio=(3./4., 4./3.),
                 hflip_prob=0.5,
                 norm_mean=[0.485, 0.456, 0.406],
                 norm_std=[0.229, 0.224, 0.225]
                 ):
        self.root = root
        self.train = train
        self.img_size = img_size
        self.apply_blur_to_input = apply_blur_to_input

        try:
            # Initialize ImageFolder just to easily get samples and loader
            # We won't use its built-in transform directly in __getitem__
            self.image_folder = datasets.ImageFolder(self.root)
            if not self.image_folder.samples and is_main_process():
                print(f"Warning: No images found in {self.root} or its subdirectories.")
        except FileNotFoundError:
            if is_main_process():
                print(f"Error: Root directory not found: {self.root}")
            raise
        except Exception as e:
             if is_main_process():
                 print(f"Error initializing ImageFolder at {self.root}: {e}")
             raise


        # --- Store Geometric Transform Parameters ---
        self.random_crop_scale = random_crop_scale
        self.random_crop_ratio = random_crop_ratio
        self.hflip_prob = hflip_prob
        # Validation transforms (resize/crop) parameters
        self.val_resize_size = int(img_size * 256 / 224) # Standard scaling

        # --- Define Non-Geometric Transforms ---
        # Blur (applied only to data path if needed, after geometric)
        self.blur_transform = transforms.GaussianBlur(kernel_size, sigma=sigma) if apply_blur_to_input else nn.Identity()

        # ToTensor and Normalize (applied to both data and gt at the end)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

        if is_main_process():
            print(f"Initialized MAEImageNetDataset for {'Training' if train else 'Validation'}")
            print(f"  Root: {root}")
            print(f"  Image Size: {img_size}")
            print(f"  Apply Blur to Input: {apply_blur_to_input}")
            if apply_blur_to_input:
                print(f"    Blur Kernel Size: {kernel_size}")
                print(f"    Blur Sigma: {sigma}")
            print(f"  Random Crop Scale: {random_crop_scale}")
            print(f"  Random Crop Ratio: {random_crop_ratio}")
            print(f"  Horizontal Flip Prob: {hflip_prob}")
            print(f"  Normalization Mean: {norm_mean}")
            print(f"  Normalization Std: {norm_std}")


    def __getitem__(self, index):
        # Error handling for index out of bounds (less likely with standard loaders but good practice)
        if index >= len(self.image_folder):
             if is_main_process():
                  print(f"Warning: Index {index} out of bounds for dataset size {len(self.image_folder)}.")
             # Return dummy data or raise error
             return torch.zeros(3, self.img_size, self.img_size), torch.zeros(3, self.img_size, self.img_size)

        path, _ = self.image_folder.samples[index]
        try:
            original_img = self.image_folder.loader(path) # Loads as PIL RGB
        except Exception as e:
             # Propagate error for DataLoader to handle potentially
             raise RuntimeError(f"Failed to load image {path}") from e

        # --- Apply Geometric Transforms Synchronously ---
        if self.train:
            # 1. RandomResizedCrop
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                original_img, scale=self.random_crop_scale, ratio=self.random_crop_ratio
            )
            img_geom_transformed = F.resized_crop(original_img, 
                                                  i, j, h, w, 
                                                  [self.img_size, self.img_size], 
                                                  interpolation=transforms.InterpolationMode.BICUBIC)

            # 2. RandomHorizontalFlip
            if random.random() < self.hflip_prob:
                img_geom_transformed = F.hflip(img_geom_transformed)

        else: # Validation transforms
            img_resized = F.resize(
                original_img, self.val_resize_size, interpolation=transforms.InterpolationMode.BICUBIC
            )
            img_geom_transformed = F.center_crop(img_resized, [self.img_size, self.img_size])

        # --- Create Branches for data and gt ---
        # Apply transforms after this point separately
        # No need for .copy() as PIL transforms typically return new objects
        data_intermediate = img_geom_transformed
        gt_intermediate = img_geom_transformed

        # --- Apply Input-Specific Transforms (to data path only) ---
        # Blur happens AFTER geometric transforms, BEFORE ToTensor
        data_intermediate = self.blur_transform(data_intermediate)
        # Add other input-specific transforms like ColorJitter here if needed

        # --- Apply Final Transforms (ToTensor, Normalize) ---
        data = self.normalize(self.to_tensor(data_intermediate))
        gt = self.normalize(self.to_tensor(gt_intermediate)) # gt uses the same geometric transform result

        return data, gt

    def __len__(self):
        # Return the number of samples found by ImageFolder
        return len(self.image_folder)


# --- Import Custom Dataset (Keep as is) ---
try:
    from data.dotadataset import MAEDataset, MAEDatasetEval
except ImportError:
    # Print only on rank 0
    if 'RANK' not in os.environ or int(os.environ['RANK']) == 0:
         print("Warning: Could not import custom MAEDataset/MAEDatasetEval. Custom dataset functionality will not work.")
    MAEDataset = None
    MAEDatasetEval = None

# --- Normalization Constants (Keep as is) ---
CUSTOM_DATASET_MEAN = [0.330694]
CUSTOM_DATASET_STD = [0.176989]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
ORIGINAL_DATA_RANGE = 1.0

# --- Utility Functions ---

# --- DDP Helper functions (Keep as is) ---
def init_distributed_mode(args):
    # ... (function definition remains the same) ...
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ['SLURM_NPROCS'])
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['MASTER_PORT'] = str(29500) # Default port
        args.dist_url = 'env://'
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0 # Default to GPU 0 for single process
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}, world {args.world_size}, gpu {args.gpu}): {args.dist_url}', flush=True)
    dist.init_process_group(backend=args.dist_backend, 
                            init_method=args.dist_url,
                            world_size=args.world_size, 
                            rank=args.rank)
    dist.barrier() # Wait for all processes to synchronize
    print("Distributed training initialized.")

def cleanup():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    if not dist.is_available(): return False
    if not dist.is_initialized(): return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized(): return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized(): return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


# --- Batch PSNR (Keep as is) ---
def batch_PSNR(img, imclean, data_range):
    # ... (function definition remains the same) ...
    if not isinstance(img, torch.Tensor) or not isinstance(imclean, torch.Tensor): raise TypeError("Inputs must be torch.Tensor")
    if img.shape != imclean.shape: raise ValueError(f"Input shapes must match: img {img.shape}, imclean {imclean.shape}")
    squared_error = (img - imclean) ** 2
    if img.ndim == 4: mse_per_image = torch.mean(squared_error, dim=(1, 2, 3))
    elif img.ndim == 3: mse_per_image = torch.mean(squared_error, dim=(1, 2))
    else: raise ValueError(f"Unsupported input dimensions: {img.ndim}. Expected 3 or 4.")
    epsilon = 1e-10
    psnr_per_image = 10.0 * torch.log10((data_range**2) / (mse_per_image + epsilon))
    return torch.mean(psnr_per_image)


# --- Masked PSNR (Keep as is) ---
@torch.no_grad()
def calculate_masked_psnr_original_scale(pred_original_img, gt_original_img, mask, patchify_func, in_chans, original_data_range):
    # ... (function definition remains the same) ...
    device = pred_original_img.device
    pred_original_patch = patchify_func(pred_original_img, in_chan=in_chans)
    target_original_patch = patchify_func(gt_original_img, in_chan=in_chans)
    squared_error_original = (pred_original_patch - target_original_patch) ** 2
    mse_original_per_patch = torch.mean(squared_error_original, dim=-1)
    epsilon = 1e-10
    mse_original_per_patch_safe = mse_original_per_patch + epsilon
    psnr_original_per_patch = 10.0 * torch.log10((original_data_range**2) / mse_original_per_patch_safe)
    num_masked = mask.sum()
    if num_masked == 0: return torch.tensor(0.0, device=device)
    mask_float = mask.to(device=psnr_original_per_patch.device, dtype=torch.float32)
    masked_psnr_sum = (psnr_original_per_patch * mask_float).sum()
    return masked_psnr_sum / num_masked


# --- time2file_name (Keep as is) ---
def time2file_name(time_str): return time_str.replace(" ", "_").replace(":", "-").split('.')[0]

# --- AverageMeter (Keep as is) ---
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count if self.count != 0 else 0

# --- save_checkpoint (Modified for Rank 0 and Scaler) ---
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_main_process(): # Only save on rank 0
        save_dir = os.path.dirname(filename)
        if not os.path.exists(save_dir):
             os.makedirs(save_dir, exist_ok=True) # Ensure directory exists
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(save_dir, 'model_best.pth')
            shutil.copyfile(filename, best_filename)
            print(f"Rank 0: Best model updated: {best_filename}")
        # print(f"Rank 0: Checkpoint saved to {filename}") # Optional: confirmation

# --- Main Training Function ---
def main(args):
    # Initialize DDP
    init_distributed_mode(args)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Rank {args.rank} using device: {device}")

    # --- Setup Save Directory and Logger (Rank 0 Only) ---
    writer = None
    save_dir = None
    if is_main_process():
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        dataset_tag = "imagenet" if args.imagenet else "custom"
        save_dir = os.path.join(args.save_dir, f"{dataset_tag}_{args.model}_{args.patch_size}_{date_time}")
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
                print(f"Rank 0: Created save directory: {save_dir}")
            except FileExistsError:
                 print(f"Rank 0: Save directory already exists (likely race condition, ignoring): {save_dir}") # Handle potential race condition
        else:
             print(f"Rank 0: Save directory exists: {save_dir}")
        writer = tensorboardX.SummaryWriter(log_dir=save_dir)

    # --- Determine Dataset Parameters & Load Datasets ---
    num_input_channels = 3 if args.imagenet else 1
    mean = IMAGENET_DEFAULT_MEAN if args.imagenet else CUSTOM_DATASET_MEAN
    std = IMAGENET_DEFAULT_STD if args.imagenet else CUSTOM_DATASET_STD

    train_dataset = None
    test_dataset = None
    if args.imagenet:
        if is_main_process(): print("Using ImageNet Dataset")
        train_dir = os.path.join(args.imagenet_path, 'train')
        val_dir = os.path.join(args.imagenet_path, 'val')
        if is_main_process():
             if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
                  print(f"ERROR: ImageNet train ({train_dir}) or val ({val_dir}) directory not found under {args.imagenet_path}")
                  if args.distributed: dist.barrier()
                  sys.exit(1)

        # Critical: Ensure dataset objects are created *before* checking their length or content,
        # especially with multiple workers. Path checks are done above.
        try:
            train_dataset = MAEImageNetDataset(train_dir, 
                                               train=True, 
                                               img_size=args.img_size,
                                               apply_blur_to_input=args.blur,
                                               kernel_size=args.kernel_size,
                                               sigma=args.sigma)
            test_dataset = MAEImageNetDataset(val_dir, 
                                              train=False, 
                                              img_size=args.img_size,
                                              apply_blur_to_input=args.blur,
                                              kernel_size=args.kernel_size,
                                              sigma=args.sigma)
            # Perform a basic check after creation on rank 0
            if is_main_process():
                if len(train_dataset) == 0 or len(test_dataset) == 0:
                    print(f"ERROR: ImageNet dataset loaded but is empty. Train: {len(train_dataset)}, Val: {len(test_dataset)}. Check content of {train_dir} and {val_dir}")
                    if args.distributed: dist.barrier()
                    sys.exit(1)
        except Exception as e:
            if is_main_process():
                print(f"\nERROR: Failed to initialize ImageNet Datasets: {e}")
                print(f"Checked paths: Train={train_dir}, Val={val_dir}")
            if args.distributed: dist.barrier()
            sys.exit(1)

    else: # Custom Dataset
        if is_main_process(): print("Using Custom Dataset")
        if MAEDataset is None or MAEDatasetEval is None:
             if is_main_process(): print("ERROR: Custom dataset classes not imported.")
             if args.distributed: dist.barrier()
             sys.exit(1)
        try:
            # Keep your existing custom dataset path logic
            train_dataset = MAEDataset(args.data_path, 
                                       args.blur, 
                                       args.kernel_size, 
                                       args.sigma, 
                                       img_size=args.img_size)
            val_data_path = args.data_path.replace('train', 'valid')
            if not os.path.isdir(val_data_path):
                 if is_main_process(): print(f"Validation path {val_data_path} not found, using train path for validation.")
                 val_data_path = args.data_path
            test_dataset = MAEDatasetEval(val_data_path, args.blur, args.kernel_size, args.sigma, img_size=args.img_size)
            # Basic check on rank 0
            if is_main_process():
                 if len(train_dataset) == 0 or len(test_dataset) == 0:
                     print(f"ERROR: Custom dataset loaded but is empty. Train: {len(train_dataset)}, Val: {len(test_dataset)}. Check paths and content.")
                     if args.distributed: dist.barrier()
                     sys.exit(1)
        except Exception as e:
            if is_main_process():
                 print(f"\nERROR: Failed to initialize custom Datasets: {e}")
                 print("Please ensure MAEDataset/MAEDatasetEval are correctly defined and paths are valid.")
            if args.distributed: dist.barrier()
            sys.exit(1)

    if train_dataset is None or test_dataset is None:
         if is_main_process(): print("ERROR: Datasets were not initialized.")
         if args.distributed: dist.barrier()
         sys.exit(1)

    # --- Create Samplers and DataLoaders ---
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, seed=args.seed) # Add seed
        test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    def seed_worker(worker_id):
        """Sets the random seed for each dataloader worker."""
        # Combine base seed, rank, and worker ID for a unique seed
        # Ensure different seeds across ranks and workers within ranks
        rank = get_rank() if is_dist_avail_and_initialized() else 0
        # Use args.num_workers to ensure seeds are unique across ranks
        worker_seed = args.seed + rank * args.num_workers + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Define generator for workers based on seed+rank for reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed + get_rank()) # Rank-specific seed for workers

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  worker_init_fn=seed_worker,
                                  generator=g, # Use generator for shuffling
                                  persistent_workers=True if args.num_workers > 0 else False)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 sampler=test_sampler,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False)

    if is_main_process():
        print(f"Train Dataset: {len(train_dataset)} samples")
        print(f"Test Dataset: {len(test_dataset)} samples")
        print(f"Train Dataloader: {len(train_dataloader)} batches/GPU ({len(train_dataloader) * args.batch_size * args.world_size} total samples approx)")
        print(f"Test Dataloader: {len(test_dataloader)} batches/GPU ({len(test_dataloader) * args.eval_batch_size * args.world_size} total samples approx)")


    # --- Model Definition ---
    if is_main_process(): print(f"Loading MAE model: {args.model}, Patch Size: {args.patch_size}, Input Channels: {num_input_channels}")
    # model_args = {'patch_size': args.patch_size, 'in_chans': num_input_channels}
    if args.model == 'tiny': model = mae_vit_tiny_mine() # Pass args correctly
    elif args.model == 'base': model = mae_vit_base_patch16() # Pass args correctly
    else: raise ValueError(f"Unknown model type: {args.model}")

    # Remove Sigmoid
    if hasattr(model, 'decoder_pred') and isinstance(model.decoder_pred, nn.Sequential):
         if len(model.decoder_pred) > 0 and isinstance(model.decoder_pred[-1], nn.Sigmoid):
              if is_main_process(): print("Removing Sigmoid layer from decoder_pred.")
              model.decoder_pred = model.decoder_pred[:-1]

    model.to(device)

    # --- Wrap Model with DDP ---
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
        if is_main_process(): print("Model wrapped with DDP.")
    else:
        model_without_ddp = model

    if is_main_process():
        print(f"Number of trainable params (M): {sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6:.2f}")


    # --- Optimizer ---
    ref_batch_size = 256
    batch_size_per_gpu = args.batch_size
    total_batch_size = batch_size_per_gpu * args.world_size
    effective_lr = args.learning_rate * total_batch_size / ref_batch_size

    if is_main_process():
        print(f"Base LR: {args.learning_rate}, Ref BS: {ref_batch_size}")
        print(f"Batch Size/GPU: {batch_size_per_gpu}, World Size: {args.world_size}, Total Batch Size: {total_batch_size}")
        print(f"Applying LR Scaling: Effective LR = {args.learning_rate} * {total_batch_size} / {ref_batch_size} = {effective_lr:.2e}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=effective_lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
    if is_main_process(): print(f"Optimizer: AdamW, LR: {effective_lr:.2e}, Weight Decay: {args.weight_decay}")

    # --- LR Scheduler ---
    num_training_steps = args.end_epoch * len(train_dataloader) * args.world_size
    warmup_steps = args.warmup_epochs * len(train_dataloader)
    if is_main_process():
        print(f"Total training steps (per GPU): {num_training_steps}")
        print(f"Warmup steps (per GPU): {warmup_steps} ({args.warmup_epochs} epochs)")
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=num_training_steps,
        lr_min=1e-6,
        warmup_t=warmup_steps,
        warmup_lr_init=1e-6,
        warmup_prefix=True,
        cycle_decay=1.0,
        t_in_epochs=False
    )

    # --- AMP: Initialize GradScaler ---
    scaler = GradScaler()
    if is_main_process(): print("AMP GradScaler initialized.")

    # --- Resume / Finetune ---
    start_epoch = args.start_epoch
    best_val_masked_psnr = 0.
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.gpu} if args.distributed else device

    # --- AMP: Modified load_checkpoint to include scaler ---
    def load_checkpoint(path, model_to_load, optimizer_to_load=None, scheduler_to_load=None, scaler_to_load=None, is_finetune=False):
        nonlocal start_epoch, best_val_masked_psnr # Allow modification
        if os.path.isfile(path):
            if is_main_process(): print(f"=> loading {'weights' if is_finetune else 'checkpoint'} '{path}'")
            checkpoint = torch.load(path, map_location=map_location)

            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
            else: state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            msg = model_to_load.load_state_dict(new_state_dict, strict=False)
            if is_main_process(): print(f"   Model loaded with status: {msg}")

            if not is_finetune:
                if optimizer_to_load and 'optimizer' in checkpoint:
                    try:
                        optimizer_to_load.load_state_dict(checkpoint['optimizer'])
                        if is_main_process(): print("   Optimizer state loaded.")
                    except ValueError as e:
                        if is_main_process(): print(f"   Warning: Could not load optimizer state (likely parameter mismatch): {e}")
                if scheduler_to_load and 'scheduler' in checkpoint:
                    try:
                        scheduler_to_load.load_state_dict(checkpoint['scheduler'])
                        if is_main_process(): print("   Scheduler state loaded.")
                    except Exception as e:
                        if is_main_process(): print(f"   Warning: Could not load scheduler state: {e}")
                # --- AMP: Load scaler state ---
                if scaler_to_load and 'scaler' in checkpoint:
                    try:
                        scaler_to_load.load_state_dict(checkpoint['scaler'])
                        if is_main_process(): print("   GradScaler state loaded.")
                    except Exception as e:
                        if is_main_process(): print(f"   Warning: Could not load GradScaler state: {e}")
                # --- End AMP ---
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    if is_main_process(): print(f"   Resuming from epoch {start_epoch}")
                if 'best_val_masked_psnr' in checkpoint:
                    best_val_masked_psnr = checkpoint['best_val_masked_psnr']
                    if is_main_process(): print(f"   Previous best validation masked PSNR: {best_val_masked_psnr:.4f}")
                elif 'best_psnr' in checkpoint:
                    best_val_masked_psnr = checkpoint['best_psnr']
                    if is_main_process(): print(f"   Loaded previous best (likely full) PSNR: {best_val_masked_psnr:.4f}")
            else: # Finetuning
                 start_epoch = 0
                 best_val_masked_psnr = 0.

        else:
            if is_main_process(): print(f"=> no {'weights' if is_finetune else 'checkpoint'} found at '{path}'")


    # --- AMP: Pass scaler to load_checkpoint when resuming ---
    if args.resume:
        load_checkpoint(args.resume, model_without_ddp, optimizer, scheduler, scaler, is_finetune=False) # Pass scaler
    elif args.finetune:
        load_checkpoint(args.finetune, model_without_ddp, is_finetune=True) # Scaler not loaded for finetune

    # --- Barrier after loading checkpoint ---
    if args.distributed:
        print(f"Rank {args.rank}: Waiting at barrier before training loop...")
        dist.barrier()
        print(f"Rank {args.rank}: Passed barrier.")

    # --- Prepare mean/std tensors ---
    mean_tensor = torch.tensor(mean, device=device).view(1, num_input_channels, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, num_input_channels, 1, 1)

    # --- Training Loop ---
    if is_main_process(): print(f"Starting training from epoch {start_epoch + 1} to {args.end_epoch}")

    for epoch_i in range(start_epoch, args.end_epoch):
        model.train()
        epoch_loss_meter = AverageMeter()
        epoch_train_psnr_full = AverageMeter()
        epoch_train_psnr_masked = AverageMeter()
        epoch_start_time = time.time()

        # --- Set epoch for Distributed Sampler ---
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch_i)

        # --- TQDM setup ---
        training_bar = tqdm(train_dataloader,
                            desc=f"[Epoch {epoch_i + 1}/{args.end_epoch}]Train",
                            colour='yellow', ncols=140,
                            disable=not is_main_process())

        for idx, (data, gt) in enumerate(training_bar):
            current_iter = epoch_i * len(train_dataloader) + idx
            data = data.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()

            # --- AMP: Forward pass with autocast ---
            with autocast():
                loss, pred_patches, mask = model(data,
                                                 gt=gt,
                                                 mask_ratio=args.mask_ratio)
                # Ensure loss is float32 for scaler, though autocast usually handles it.
                # Redundant if loss calculation is stable in FP16.
                # loss = loss.float()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING Rank {get_rank()}: NaN or Inf loss detected BEFORE scaling at epoch {epoch_i+1}, iter {idx}. Loss: {loss.item()}. Skipping step.")
                optimizer.zero_grad() # Reset gradients if skipping
                continue # Skip backward, step, update

            # --- AMP: Scale loss and call backward ---
            scaler.scale(loss).backward()

            # --- AMP: Unscale gradients before clipping ---
            scaler.unscale_(optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # --- AMP: Optimizer step ---
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called.
            # Otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # --- AMP: Update scaler for next iteration ---
            scaler.update()

            # Scheduler update
            scheduler.step_update(current_iter)

            # --- Calculate PSNR (outside autocast, typically in FP32) ---
            with torch.no_grad():
                gt_original = gt * std_tensor + mean_tensor
                gt_original_clamped = torch.clamp(gt_original, 0., ORIGINAL_DATA_RANGE)

                pred_img_standardized = model_without_ddp.unpatchify(pred_patches.float(), in_chan=num_input_channels) # Ensure FP32 input if needed
                pred_original = pred_img_standardized * std_tensor + mean_tensor
                pred_original_clamped = torch.clamp(pred_original, 0., ORIGINAL_DATA_RANGE)

                psnr_train_full = batch_PSNR(pred_original_clamped,
                                             gt_original_clamped,
                                             ORIGINAL_DATA_RANGE)
                psnr_train_masked = calculate_masked_psnr_original_scale(
                    pred_original_clamped,
                    gt_original_clamped, mask,
                    model_without_ddp.patchify,
                    num_input_channels,
                    ORIGINAL_DATA_RANGE
                )

            batch_size = data.size(0)
            epoch_loss_meter.update(loss.item(), batch_size) # Loss is already on CPU via .item()
            epoch_train_psnr_full.update(psnr_train_full.item(), batch_size)
            epoch_train_psnr_masked.update(psnr_train_masked.item(), batch_size)

            # --- Logging ---
            if is_main_process():
                lr = optimizer.param_groups[0]['lr']
                training_bar.set_postfix({
                    "Loss": f"{epoch_loss_meter.val:.4f}",
                    "Avg Loss": f"{epoch_loss_meter.avg:.4f}",
                    "Tr PSNR Full": f"{epoch_train_psnr_full.avg:.2f}",
                    "Tr PSNR Mask": f"{epoch_train_psnr_masked.avg:.2f}",
                    "LR": f"{lr:.2e}",
                    # --- AMP: Log scaler state if needed ---
                    "Scale": f"{scaler.get_scale():.1f}"
                     })
                if current_iter % 100 == 0:
                     writer.add_scalar('Train/iter_loss', loss.item(), current_iter)
                     writer.add_scalar('Train/iter_psnr_full', psnr_train_full.item(), current_iter)
                     writer.add_scalar('Train/iter_psnr_masked', psnr_train_masked.item(), current_iter)
                     writer.add_scalar('lr', lr, current_iter)
                     # --- AMP: Log scaler scale ---
                     writer.add_scalar('AMP/scale', scaler.get_scale(), current_iter)

        # --- End of Epoch Training Summary ---
        if is_main_process():
            print(f"Epoch {epoch_i + 1} Train Summary: Loss: {epoch_loss_meter.avg:.4f}, "
                  f"PSNR Full: {epoch_train_psnr_full.avg:.2f} dB, PSNR Masked: {epoch_train_psnr_masked.avg:.2f} dB")
            writer.add_scalar('Train/epoch_avg_loss', epoch_loss_meter.avg, epoch_i + 1)
            writer.add_scalar('Train/epoch_avg_psnr_full', epoch_train_psnr_full.avg, epoch_i + 1)
            writer.add_scalar('Train/epoch_avg_psnr_masked', epoch_train_psnr_masked.avg, epoch_i + 1)
            epoch_end_time = time.time(); print(f'Epoch {epoch_i + 1} Time: {epoch_end_time - epoch_start_time:.2f}s')

        # --- Evaluation ---
        if (epoch_i + 1) % args.eval_interval == 0:
            model.eval()
            val_loss_meter = AverageMeter()
            val_psnr_full = AverageMeter()
            val_psnr_masked = AverageMeter()

            if is_main_process(): print(f'--- Validation Epoch {epoch_i + 1} ---')
            val_bar = tqdm(test_dataloader, 
                           desc=f"Val", 
                           colour='blue', 
                           ncols=140,
                           disable=not is_main_process())
            visible_batch_idx = random.randint(0, len(test_dataloader) - 1) if len(test_dataloader) > 0 else -1

            for idx, (data, gt) in enumerate(val_bar):
                data = data.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                with torch.no_grad():
                    # --- AMP: Use autocast for evaluation forward pass ---
                    with autocast():
                        loss, pred_patches, mask = model(data, gt=gt, mask_ratio=args.eval_mask_ratio)
                        # loss = loss.float() # Ensure loss is FP32 if needed for meter

                    # --- Calculations outside autocast (FP32) ---
                    gt_original = gt * std_tensor + mean_tensor
                    gt_original_clamped = torch.clamp(gt_original, 0., ORIGINAL_DATA_RANGE)

                    pred_img_standardized = model_without_ddp.unpatchify(pred_patches.float(), 
                                                                         in_chan=num_input_channels) # Ensure FP32
                    pred_original = pred_img_standardized * std_tensor + mean_tensor
                    pred_original_clamped = torch.clamp(pred_original, 0., ORIGINAL_DATA_RANGE)

                    psnr_val_full = batch_PSNR(pred_original_clamped,
                                               gt_original_clamped,
                                               ORIGINAL_DATA_RANGE)
                    psnr_val_masked = calculate_masked_psnr_original_scale(
                        pred_original_clamped,
                        gt_original_clamped,
                        mask,
                        model_without_ddp.patchify,
                        num_input_channels,
                        ORIGINAL_DATA_RANGE
                    )

                    # Save images on rank 0
                    if is_main_process() and idx == visible_batch_idx and save_dir is not None:
                        try:
                            # Ensure tensors are detached, moved to CPU, and converted before saving
                            img_to_save_tensor = pred_original_clamped[0].detach().float().cpu()
                            gt_to_save_tensor = gt_original_clamped[0].detach().float().cpu()

                            img_to_save = (img_to_save_tensor * 255).clamp(0, 255).byte()
                            gt_to_save = (gt_to_save_tensor * 255).clamp(0, 255).byte()

                            if num_input_channels == 1:
                                img_pil = Image.fromarray(img_to_save.squeeze().numpy(), mode='L')
                                gt_pil = Image.fromarray(gt_to_save.squeeze().numpy(), mode='L')
                            else:
                                img_pil = Image.fromarray(img_to_save.permute(1, 2, 0).numpy(), mode='RGB')
                                gt_pil = Image.fromarray(gt_to_save.permute(1, 2, 0).numpy(), mode='RGB')
                            img_pil.save(os.path.join(save_dir, f'{epoch_i + 1}_pred.png'))
                            gt_pil.save(os.path.join(save_dir, f'{epoch_i + 1}_gt.png'))
                        except Exception as e:
                             print(f"Rank 0 WARNING: Failed to save validation images: {e}")

                batch_size = data.size(0)
                val_loss_meter.update(loss.item(), batch_size)
                val_psnr_full.update(psnr_val_full.item(), batch_size)
                if not torch.isnan(psnr_val_masked) and not torch.isinf(psnr_val_masked):
                     val_psnr_masked.update(psnr_val_masked.item(), batch_size)
                elif is_main_process():
                     print(f"Warning: NaN/Inf encountered in val_psnr_masked (val={psnr_val_masked.item()}), skipping update.")

                if is_main_process():
                    val_bar.set_postfix({ "Loss": f"{val_loss_meter.val:.4f}",
                                          "Avg Loss": f"{val_loss_meter.avg:.4f}",
                                          "PSNR Full": f"{val_psnr_full.avg:.2f}",
                                          "PSNR Mask": f"{val_psnr_masked.avg:.2f}"})

            # --- Synchronize validation meters ---
            val_loss_meter.synchronize_between_processes()
            val_psnr_full.synchronize_between_processes()
            val_psnr_masked.synchronize_between_processes()

            avg_val_loss = val_loss_meter.avg
            avg_val_psnr_full = val_psnr_full.avg
            avg_val_psnr_masked = val_psnr_masked.avg

            # --- Log and Save Checkpoint (Rank 0 Only) ---
            if is_main_process():
                print(f"Epoch {epoch_i + 1} Validation Summary (Aggregated): Loss: {avg_val_loss:.4f}, "
                      f"PSNR Full: {avg_val_psnr_full:.2f} dB, PSNR Masked: {avg_val_psnr_masked:.2f} dB")
                writer.add_scalar("Val/epoch_avg_loss", avg_val_loss, epoch_i + 1)
                writer.add_scalar("Val/epoch_avg_psnr_full", avg_val_psnr_full, epoch_i + 1)
                writer.add_scalar("Val/epoch_avg_psnr_masked", avg_val_psnr_masked, epoch_i + 1)

                is_best = avg_val_psnr_masked > best_val_masked_psnr
                if is_best:
                    best_val_masked_psnr = avg_val_psnr_masked
                    print(f"*** Rank 0: New Best Val Masked PSNR: {best_val_masked_psnr:.4f} ***")

                # --- AMP: Include scaler state in checkpoint ---
                save_dict = {
                    'epoch': epoch_i,
                    'state_dict': model_without_ddp.state_dict(),
                    'best_val_masked_psnr': best_val_masked_psnr,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), # Add scaler state
                    'args': args
                }
                filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch_i + 1}.pth')
                save_checkpoint(save_dict, is_best, filename=filename)

        # --- Barrier at end of epoch ---
        if args.distributed:
            dist.barrier()


    if is_main_process():
        print(f"Training finished. Best Validation Masked PSNR: {best_val_masked_psnr:.4f}")
        if writer: writer.close()

    # Clean up DDP
    cleanup()


# --- Argument Parser (Keep as is) ---
def parse_args():
    parser = argparse.ArgumentParser(description='MAE Training (Multi-GPU DDP + AMP)') # Updated description
    # Keep all your existing arguments
    parser.add_argument('--imagenet', action='store_true', help='Use ImageNet-1k dataset.')
    parser.add_argument('--data_path', type=str, default=None, help='Path to custom training data directory.')
    parser.add_argument('--imagenet_path', type=str, default=None, help='Path to ImageNet-1k root directory.')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--blur', action='store_true', help='Apply Gaussian blur (custom dataset only)')
    parser.add_argument('--kernel_size', type=int, default=3, help='Gaussian kernel size (custom dataset only)')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian sigma (custom dataset only)')
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'base'], help='MAE model size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Masking ratio for training')
    parser.add_argument('--eval_mask_ratio', type=float, default=0.0, help='Masking ratio for evaluation (0.0 means no masking)')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size PER GPU')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size PER GPU')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--end_epoch', type=int, default=400, help='Total number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='Base learning rate (scaled in script)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='Warmup epochs')
    parser.add_argument('--save_dir', type=str, default='./mae_checkpoints_ddp_amp', help='Save directory') # Changed default slightly
    parser.add_argument('--num_workers', type=int, default=8, help='Data loading workers per GPU')
    parser.add_argument('--seed', type=int, default=717, help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval (in epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint path')
    parser.add_argument('--finetune', type=str, default=None, help='Finetune weights path')

    args = parser.parse_args()

    if args.imagenet:
        if args.imagenet_path is None: parser.error("--imagenet requires --imagenet_path.")
    else:
        if args.data_path is None: parser.error("Custom dataset requires --data_path.")

    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    args.dist_url = 'env://'

    return args

if __name__ == "__main__":
    args = parse_args()

    # --- Seeding (Before DDP init is usually fine) ---
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # These can impact reproducibility with AMP/DDP, but often boost performance
    torch.backends.cudnn.deterministic = False # As per original, might hinder AMP perf
    torch.backends.cudnn.benchmark = True   # As per original, set True for potential speedup if input sizes fixed

    main(args)