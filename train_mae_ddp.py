import os
import subprocess
import time
import shutil
import datetime
import argparse
import random
import sys # Added for sys.exit

import torch
import numpy as np
from tqdm import tqdm
import tensorboardX
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # Import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from timm.models.vision_transformer import PatchEmbed, Block
from timm.scheduler import CosineLRScheduler
from PIL import Image
from functools import partial

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import your models
from model import mae_vit_tiny_mine, mae_vit_base_patch16 # Ensure these are importable

# -- Your MAEImageNetDataset Definition (Keep as is) --
# ... (MAEImageNetDataset class definition remains exactly the same as in your input) ...
class MAEImageNetDataset(Dataset):
    def __init__(self, root, train=True, img_size=224): # Add img_size param
        self.image_folder = datasets.ImageFolder(root, transform=None)
        # Define transforms inside __init__ using img_size
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, 
                                         scale=(0.2, 1.0), 
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.valid_transforms = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224), 
                              interpolation=transforms.InterpolationMode.BICUBIC), # Scale resize based on img_size
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # The transform applied to get the input 'data'
        self.transform = self.train_transforms if train else self.valid_transforms
        # The transform applied to get the target 'gt' (usually simpler)
        # For MAE GT, often just resize/crop + normalize, same as validation transform
        self.target_transform = self.valid_transforms # Use validation transform for GT

        if not self.image_folder.samples:
             # Check only on rank 0 to avoid multiple prints
             if is_main_process():
                 print(f"Warning: No images found in {root} or its subdirectories (checked on rank 0).")
             # Allow empty dataset for DDP initialization if path is wrong, error later
             # raise ValueError(f"No images found in {root} or its subdirectories.")

    def __getitem__(self, index):
        # Ensure index is valid even if dataset is temporarily empty during setup
        if index >= len(self.image_folder):
            print(f"Warning: Index {index} out of bounds for dataset size {len(self.image_folder)}. This might happen during DDP setup with incorrect paths.")
            # Return dummy data or handle appropriately, maybe raise error later
            # For now, let it potentially fail later in loader
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224) # Adjust shape/channels if needed

        path, _ = self.image_folder.samples[index]
        try:
            original_img = self.image_folder.loader(path) # Loads as PIL RGB by default
        except Exception as e:
             # Print error only once
             if is_main_process():
                 print(f"Rank {dist.get_rank()} Error loading image {path}: {e}")
             # Propagate error to potentially stop training
             raise RuntimeError(f"Rank {dist.get_rank()} Failed to load image {path}") from e

        data = self.transform(original_img)
        gt = self.target_transform(original_img)
        return data, gt

    def __len__(self):
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

# --- DDP Helper functions ---
def init_distributed_mode(args):
    """Initializes the distributed environment."""
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
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
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
    # Ensure num_masked is broadcast correctly if needed, though sum is usually fine
    # This calculation should be correct per-GPU, aggregation happens later via AverageMeter sync
    # However, num_masked itself might need aggregation if mask generation is non-deterministic per GPU
    # Assuming mask generation is deterministic or we average the PSNR values:
    return masked_psnr_sum / num_masked # Return per-GPU masked PSNR sum / per-GPU masked count

# --- time2file_name (Keep as is) ---
def time2file_name(time_str): return time_str.replace(" ", "_").replace(":", "-").split('.')[0]

# --- AverageMeter (Modified for DDP) ---
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
        # Calculate local average immediately
        self.avg = self.sum / self.count if self.count != 0 else 0

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the self.val field.
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count if self.count != 0 else 0

# --- save_checkpoint (Modified for Rank 0) ---
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
        # Ensure save_dir is created before any process might need it (though saving is rank 0)
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
    num_input_channels = 3 if args.imagenet else 1 # Assuming custom is single channel
    mean = IMAGENET_DEFAULT_MEAN if args.imagenet else CUSTOM_DATASET_MEAN
    std = IMAGENET_DEFAULT_STD if args.imagenet else CUSTOM_DATASET_STD

    # Use args directly for paths
    train_dataset = None
    test_dataset = None
    if args.imagenet:
        if is_main_process(): print("Using ImageNet Dataset")
        train_dir = os.path.join(args.imagenet_path, 'train')
        val_dir = os.path.join(args.imagenet_path, 'val')
        # Check paths only on rank 0 to avoid redundant errors/warnings
        if is_main_process():
             if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
                  print(f"ERROR: ImageNet train ({train_dir}) or val ({val_dir}) directory not found under {args.imagenet_path}")
                  # Consider signaling other processes to exit gracefully
                  dist.barrier() # Wait for others
                  sys.exit(1) # Exit if paths are wrong

        # All processes create the dataset instance
        train_dataset = MAEImageNetDataset(train_dir, train=True, img_size=args.img_size)
        test_dataset = MAEImageNetDataset(val_dir, train=False, img_size=args.img_size)

    else: # Custom Dataset (Ensure MAEDataset/Eval are DDP-safe if they do complex init)
        if is_main_process(): print("Using Custom Dataset")
        if MAEDataset is None or MAEDatasetEval is None:
             if is_main_process(): print("ERROR: Custom dataset classes not imported.")
             dist.barrier()
             sys.exit(1)

        # Logic for custom dataset paths (similar rank 0 checks might be needed)
        # ... (keep your existing custom dataset path logic) ...
        try:
            train_dataset = MAEDataset(args.data_path, args.blur, args.kernel_size, args.sigma, img_size=args.img_size)
            val_data_path = args.data_path.replace('train', 'valid')
            if not os.path.isdir(val_data_path):
                 if is_main_process(): print(f"Validation path {val_data_path} not found, using train path for validation.")
                 val_data_path = args.data_path
            test_dataset = MAEDatasetEval(val_data_path, args.blur, args.kernel_size, args.sigma, img_size=args.img_size)
        except Exception as e:
            if is_main_process():
                 print(f"\nERROR: Failed to initialize custom Datasets: {e}")
                 print("Please ensure MAEDataset/MAEDatasetEval are correctly defined and paths are valid.")
            dist.barrier()
            sys.exit(1)

    # Check if datasets were created successfully before proceeding
    if train_dataset is None or test_dataset is None:
         if is_main_process(): print("ERROR: Datasets were not initialized.")
         dist.barrier()
         sys.exit(1)

    # --- Create Samplers and DataLoaders ---
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        # For evaluation, usually shuffle=False. Using sampler distributes evaluation load.
        test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        # Set epoch for sampler to ensure shuffling changes each epoch
        # train_sampler.set_epoch(epoch_i) # Set this inside the epoch loop
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size, # Batch size PER GPU
                                  shuffle=(train_sampler is None), # Shuffle only if no sampler
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  persistent_workers=True if args.num_workers > 0 else False)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.eval_batch_size, # Eval batch size PER GPU
                                 shuffle=False, # Never shuffle test set
                                 sampler=test_sampler,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False) # Usually keep all validation samples

    if is_main_process():
        print(f"Train Dataset: {len(train_dataset)} samples")
        print(f"Test Dataset: {len(test_dataset)} samples")
        print(f"Train Dataloader: {len(train_dataloader)} batches/GPU ({len(train_dataset)} total samples)")
        print(f"Test Dataloader: {len(test_dataloader)} batches/GPU ({len(test_dataset)} total samples)")

    # --- Model Definition ---
    if is_main_process(): print(f"Loading MAE model: {args.model}, Patch Size: {args.patch_size}, Input Channels: {num_input_channels}")

    # Model creation should be identical across processes
    model_args = {'patch_size': args.patch_size, 'in_chans': num_input_channels}
    if args.model == 'tiny': model = mae_vit_tiny_mine() # Pass args correctly
    elif args.model == 'base': model = mae_vit_base_patch16() # Pass args correctly
    else: raise ValueError(f"Unknown model type: {args.model}")

    # Remove Sigmoid if exists (Done before DDP wrapping)
    # Accessing decoder_pred assumes it's a direct attribute or accessible path
    if hasattr(model, 'decoder_pred') and isinstance(model.decoder_pred, nn.Sequential):
         if len(model.decoder_pred) > 0 and isinstance(model.decoder_pred[-1], nn.Sigmoid):
              if is_main_process(): print("Removing Sigmoid layer from decoder_pred.")
              model.decoder_pred = model.decoder_pred[:-1]

    model.to(device)

    # --- Wrap Model with DDP ---
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False) # Set find_unused_parameters=True if you encounter issues
        model_without_ddp = model.module # Access original model attributes via .module
        if is_main_process(): print("Model wrapped with DDP.")
    else:
        model_without_ddp = model # No wrapping needed

    if is_main_process():
        print(f"Number of trainable params (M): {sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6:.2f}")


    # --- Optimizer ---
    # LR scaling based on TOTAL batch size
    ref_batch_size = 256 # Your reference batch size
    batch_size_per_gpu = args.batch_size
    total_batch_size = batch_size_per_gpu * args.world_size
    effective_lr = args.learning_rate * total_batch_size / ref_batch_size

    if is_main_process():
        print(f"Base LR: {args.learning_rate}, Ref BS: {ref_batch_size}")
        print(f"Batch Size/GPU: {batch_size_per_gpu}, World Size: {args.world_size}, Total Batch Size: {total_batch_size}")
        print(f"Applying LR Scaling: Effective LR = {args.learning_rate} * {total_batch_size} / {ref_batch_size} = {effective_lr:.2e}")

    # Optimizer uses parameters from the DDP-wrapped model or original model
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=effective_lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
    if is_main_process(): print(f"Optimizer: AdamW, LR: {effective_lr:.2e}, Weight Decay: {args.weight_decay}")


    # --- LR Scheduler (Same logic, uses optimizer) ---
    num_training_steps = args.end_epoch * len(train_dataloader) # Steps per epoch *per GPU*
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

    # --- Resume / Finetune (Load state dict BEFORE DDP wrapping conceptually, but handled correctly here) ---
    start_epoch = args.start_epoch
    best_val_masked_psnr = 0.
    # Load checkpoint on the specific device for this rank
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.gpu} if args.distributed else device

    def load_checkpoint(path, model_to_load, optimizer_to_load=None, scheduler_to_load=None, is_finetune=False):
        nonlocal start_epoch, best_val_masked_psnr # Allow modification
        if os.path.isfile(path):
            if is_main_process(): print(f"=> loading {'weights' if is_finetune else 'checkpoint'} '{path}'")
            checkpoint = torch.load(path, map_location=map_location)

            # Determine the state dict key
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model'] # Common in timm/huggingface
            else: state_dict = checkpoint # Assume the file only contains the state dict

            # Handle 'module.' prefix (from DDP saving or single GPU loading)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k # remove `module.` prefix
                new_state_dict[name] = v

            # Load into the non-DDP model structure first
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
                    except Exception as e: # Catch broader scheduler state issues
                        if is_main_process(): print(f"   Warning: Could not load scheduler state: {e}")
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    if is_main_process(): print(f"   Resuming from epoch {start_epoch}")
                # Handle different potential keys for best metric
                if 'best_val_masked_psnr' in checkpoint:
                    best_val_masked_psnr = checkpoint['best_val_masked_psnr']
                    if is_main_process(): print(f"   Previous best validation masked PSNR: {best_val_masked_psnr:.4f}")
                elif 'best_psnr' in checkpoint: # Legacy key check
                    best_val_masked_psnr = checkpoint['best_psnr']
                    if is_main_process(): print(f"   Loaded previous best (likely full) PSNR: {best_val_masked_psnr:.4f}")
            else: # Finetuning, reset epoch
                 start_epoch = 0
                 best_val_masked_psnr = 0. # Reset best metric for finetuning

        else:
            if is_main_process(): print(f"=> no {'weights' if is_finetune else 'checkpoint'} found at '{path}'")

    if args.resume:
        load_checkpoint(args.resume, model_without_ddp, optimizer, scheduler, is_finetune=False)
    elif args.finetune:
        load_checkpoint(args.finetune, model_without_ddp, is_finetune=True)

    # --- Barrier after loading checkpoint and before starting training ---
    if args.distributed:
        print(f"Rank {args.rank}: Waiting at barrier before training loop...")
        dist.barrier()
        print(f"Rank {args.rank}: Passed barrier.")

    # --- Prepare mean/std tensors (On correct device) ---
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

        # --- TQDM setup (Rank 0 only) ---
        training_bar = tqdm(train_dataloader,
                            desc=f"[Epoch {epoch_i + 1}/{args.end_epoch}]Train",
                            colour='yellow', ncols=140,
                            disable=not is_main_process()) # Disable bar if not rank 0

        for idx, (data, gt) in enumerate(training_bar):
            current_iter = epoch_i * len(train_dataloader) + idx
            data = data.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass through DDP model
            loss, pred_patches, mask = model(data,
                                             gt=gt,
                                             mask_ratio=args.mask_ratio)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR Rank {args.rank}: NaN or Inf loss detected at epoch {epoch_i+1}, iter {idx}. Skipping batch.")
                # Consider signaling other processes or implementing robust skipping
                continue # Skip this batch

            loss.backward()
            # Gradient clipping (applied to DDP model parameters)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # Scheduler update based on iterations
            scheduler.step_update(current_iter)

            # --- Calculate PSNR (on each GPU's data) ---
            # Use model_without_ddp (or model.module) for methods like unpatchify/patchify
            with torch.no_grad():
                gt_original = gt * std_tensor + mean_tensor
                gt_original_clamped = torch.clamp(gt_original, 0., ORIGINAL_DATA_RANGE)

                # Use module to access unpatchify/patchify
                pred_img_standardized = model_without_ddp.unpatchify(pred_patches, in_chan=num_input_channels)
                pred_original = pred_img_standardized * std_tensor + mean_tensor
                pred_original_clamped = torch.clamp(pred_original, 0., ORIGINAL_DATA_RANGE)

                psnr_train_full = batch_PSNR(pred_original_clamped,
                                             gt_original_clamped,
                                             ORIGINAL_DATA_RANGE)
                psnr_train_masked = calculate_masked_psnr_original_scale(
                    pred_original_clamped,
                    gt_original_clamped, mask,
                    model_without_ddp.patchify, # Use module.patchify
                    num_input_channels,
                    ORIGINAL_DATA_RANGE
                )

            batch_size = data.size(0) # Per-GPU batch size
            # Update local meters
            epoch_loss_meter.update(loss.item(), batch_size)
            epoch_train_psnr_full.update(psnr_train_full.item(), batch_size)
            epoch_train_psnr_masked.update(psnr_train_masked.item(), batch_size)

            # --- Logging (Rank 0 Only) ---
            if is_main_process():
                lr = optimizer.param_groups[0]['lr'] # Get current LR
                training_bar.set_postfix({
                    "Loss": f"{epoch_loss_meter.val:.4f}", # Show current val for responsiveness
                    "Avg Loss": f"{epoch_loss_meter.avg:.4f}",
                    "Tr PSNR Full": f"{epoch_train_psnr_full.avg:.2f}",
                    "Tr PSNR Mask": f"{epoch_train_psnr_masked.avg:.2f}",
                    "LR": f"{lr:.2e}"
                     })
                # Log to TensorBoard less frequently
                if current_iter % 100 == 0:
                     writer.add_scalar('Train/iter_loss', loss.item(), current_iter)
                     writer.add_scalar('Train/iter_psnr_full', psnr_train_full.item(), current_iter)
                     writer.add_scalar('Train/iter_psnr_masked', psnr_train_masked.item(), current_iter)
                     writer.add_scalar('lr', lr, current_iter)

        # --- End of Epoch Training Summary (Rank 0 Only) ---
        # No need to synchronize training meters, avg is usually sufficient for monitoring
        if is_main_process():
            print(f"Epoch {epoch_i + 1} Train Summary: Loss: {epoch_loss_meter.avg:.4f}, "
                  f"PSNR Full: {epoch_train_psnr_full.avg:.2f} dB, PSNR Masked: {epoch_train_psnr_masked.avg:.2f} dB")
            writer.add_scalar('Train/epoch_avg_loss', epoch_loss_meter.avg, epoch_i + 1)
            writer.add_scalar('Train/epoch_avg_psnr_full', epoch_train_psnr_full.avg, epoch_i + 1)
            writer.add_scalar('Train/epoch_avg_psnr_masked', epoch_train_psnr_masked.avg, epoch_i + 1)
            epoch_end_time = time.time(); print(f'Epoch {epoch_i + 1} Time: {epoch_end_time - epoch_start_time:.2f}s')

        # --- Evaluation (Performed on all GPUs, aggregated) ---
        if (epoch_i + 1) % args.eval_interval == 0:
            model.eval()
            val_loss_meter = AverageMeter()
            val_psnr_full = AverageMeter()
            val_psnr_masked = AverageMeter()

            if is_main_process(): print(f'--- Validation Epoch {epoch_i + 1} ---')
            # TQDM for validation (Rank 0 only)
            val_bar = tqdm(test_dataloader, desc=f"Val", colour='blue', ncols=140,
                           disable=not is_main_process())
            # Logic for saving images only needed on rank 0
            visible_batch_idx = random.randint(0, len(test_dataloader) - 1) if len(test_dataloader) > 0 else -1

            for idx, (data, gt) in enumerate(val_bar):
                data = data.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                with torch.no_grad():
                    # Use eval_mask_ratio for validation if specified
                    loss, pred_patches, mask = model(data, gt=gt, mask_ratio=args.eval_mask_ratio)
                    gt_original = gt * std_tensor + mean_tensor
                    gt_original_clamped = torch.clamp(gt_original, 0., ORIGINAL_DATA_RANGE)

                    # Use module for unpatchify/patchify
                    pred_img_standardized = model_without_ddp.unpatchify(pred_patches, in_chan=num_input_channels)
                    pred_original = pred_img_standardized * std_tensor + mean_tensor
                    pred_original_clamped = torch.clamp(pred_original, 0., ORIGINAL_DATA_RANGE)

                    psnr_val_full = batch_PSNR(pred_original_clamped,
                                               gt_original_clamped,
                                               ORIGINAL_DATA_RANGE)
                    psnr_val_masked = calculate_masked_psnr_original_scale(
                        pred_original_clamped,
                        gt_original_clamped,
                        mask, # Pass the validation mask
                        model_without_ddp.patchify, # Use module.patchify
                        num_input_channels,
                        ORIGINAL_DATA_RANGE
                    )

                    # Save images on rank 0 only
                    if is_main_process() and idx == visible_batch_idx and save_dir is not None:
                        try:
                            img_to_save = (pred_original_clamped[0].detach().cpu() * 255).byte()
                            gt_to_save = (gt_original_clamped[0].detach().cpu() * 255).byte()
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


                batch_size = data.size(0) # Per-GPU batch size
                # Update local meters
                val_loss_meter.update(loss.item(), batch_size)
                val_psnr_full.update(psnr_val_full.item(), batch_size)
                # Handle potential NaN/Inf in PSNR if masked count is zero or error is extreme
                if not torch.isnan(psnr_val_masked) and not torch.isinf(psnr_val_masked):
                     val_psnr_masked.update(psnr_val_masked.item(), batch_size)
                elif is_main_process(): # Print warning only once
                     print(f"Warning: NaN/Inf encountered in val_psnr_masked (val={psnr_val_masked.item()}), skipping update for this batch.")

                # Update TQDM bar only on rank 0 with local values for responsiveness
                if is_main_process():
                    val_bar.set_postfix({ "Loss": f"{val_loss_meter.val:.4f}",
                                          "Avg Loss": f"{val_loss_meter.avg:.4f}", # Show local avg
                                          "PSNR Full": f"{val_psnr_full.avg:.2f}",
                                          "PSNR Mask": f"{val_psnr_masked.avg:.2f}"})

            # --- Synchronize validation meters across all processes ---
            val_loss_meter.synchronize_between_processes()
            val_psnr_full.synchronize_between_processes()
            val_psnr_masked.synchronize_between_processes()

            # Now meters contain the aggregated global averages
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

                # Save checkpoint only on rank 0
                # Ensure state dict is saved from the underlying model (.module)
                save_dict = {
                    'epoch': epoch_i,
                    'state_dict': model_without_ddp.state_dict(), # Save non-DDP model state
                    'best_val_masked_psnr': best_val_masked_psnr,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), # Save scheduler state
                    'args': args # Save args for reference
                }
                filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch_i + 1}.pth')
                save_checkpoint(save_dict, is_best, filename=filename)
                # save_checkpoint handles printing confirmation/best model update

        # Barrier at end of epoch ensure all processes finished before next epoch/final cleanup
        if args.distributed:
            dist.barrier()


    if is_main_process():
        print(f"Training finished. Best Validation Masked PSNR: {best_val_masked_psnr:.4f}")
        if writer: writer.close()

    # Clean up DDP
    cleanup()


# --- Argument Parser (Keep as is, DDP args added implicitly/via env) ---
def parse_args():
    parser = argparse.ArgumentParser(description='MAE Training (Multi-GPU DDP)')
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
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size PER GPU') # Clarified help text
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size PER GPU') # Clarified help text
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--end_epoch', type=int, default=400, help='Total number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='Base learning rate (scaled in script based on total batch size)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='Warmup epochs')
    parser.add_argument('--save_dir', type=str, default='./mae_checkpoints_ddp', help='Save directory') # Changed default slightly
    parser.add_argument('--num_workers', type=int, default=8, help='Data loading workers per GPU') # Clarified help text
    parser.add_argument('--seed', type=int, default=114514, help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval (in epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint path')
    parser.add_argument('--finetune', type=str, default=None, help='Finetune weights path')

    # DDP arguments (usually set by launcher, but can be added for manual launch)
    # parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training') # Handled by init_distributed_mode from env
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    # Validate paths based on dataset choice
    if args.imagenet:
        if args.imagenet_path is None: parser.error("--imagenet requires --imagenet_path.")
    else:
        if args.data_path is None: parser.error("Custom dataset requires --data_path.")

    # Add placeholder distributed args for init_distributed_mode
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    args.dist_url = 'env://' # Default

    return args

if __name__ == "__main__":
    args = parse_args()

    # --- Seeding (Done before model/data creation, potentially per-rank variation) ---
    # It's often good practice to ensure the base seed is the same,
    # and add rank for variations in things like data augmentation order if needed.
    # Model initialization MUST be identical across ranks.
    # The current setup seeds globally before DDP init, which is generally okay if
    # model init uses torch's global generator.
    seed = args.seed # Use the base seed first
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # Seed current GPU
    np.random.seed(seed)
    random.seed(seed)
    # If using DDP, can add rank for further diversification AFTER DDP init if desired
    # seed = args.seed + get_rank() # Optional: for rank-specific seeding later if needed

    # Consider setting deterministic=False, benchmark=True for potential speedup
    # DDP often works better with benchmark=True if input sizes are fixed
    torch.backends.cudnn.deterministic = True # Keep as per original script request
    torch.backends.cudnn.benchmark = False   # Keep as per original script request

    main(args)