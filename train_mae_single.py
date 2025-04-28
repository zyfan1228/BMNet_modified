import os
import time
import shutil
import datetime
import argparse
import random

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

# Import your models
from model import mae_vit_tiny_mine, mae_vit_base_patch16


# Import your specific ImageNet Dataset class
# (Make sure this definition is available, either imported or defined above main)
# -- Your MAEImageNetDataset Definition (as provided) --
# -- for ImageNet-1k --
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # ImageNet

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224), # Assuming args.img_size is 224, otherwise use args.img_size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

# Validation transform usually doesn't use RandomResizedCrop
valid_transforms = transforms.Compose([
    transforms.Resize(256), # Standard practice: resize then center crop
    transforms.CenterCrop(224), # Assuming args.img_size is 224
    transforms.ToTensor(),
    normalize,
])

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
             raise ValueError(f"No images found in {root} or its subdirectories.")

    def __getitem__(self, index):
        path, _ = self.image_folder.samples[index]
        try:
            original_img = self.image_folder.loader(path) # Loads as PIL RGB by default
        except Exception as e:
             print(f"Error loading image {path}: {e}")
             raise RuntimeError(f"Failed to load image {path}") from e

        data = self.transform(original_img)
        gt = self.target_transform(original_img)
        return data, gt

    def __len__(self):
        return len(self.image_folder)
# -- End MAEImageNetDataset Definition --


# Import your custom dataset (ensure it exists and works)
try:
    from data.dotadataset import MAEDataset, MAEDatasetEval
except ImportError:
    print("Warning: Could not import custom MAEDataset/MAEDatasetEval. Custom dataset functionality will not work.")
    MAEDataset = None
    MAEDatasetEval = None

# --- Normalization Constants ---
CUSTOM_DATASET_MEAN = [0.330694] # Assuming single channel
CUSTOM_DATASET_STD = [0.176989]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
ORIGINAL_DATA_RANGE = 1.0 # We de-normalize back to [0, 1] for PSNR/SSIM

# --- Utility Functions (batch_PSNR, AverageMeter, etc.) ---
# (Keep the definitions of batch_PSNR, calculate_masked_psnr_original_scale,
#  time2file_name, AverageMeter, save_checkpoint as they were in the previous response)
def batch_PSNR(img, imclean, data_range):
    if not isinstance(img, torch.Tensor) or not isinstance(imclean, torch.Tensor): raise TypeError("Inputs must be torch.Tensor")
    if img.shape != imclean.shape: raise ValueError(f"Input shapes must match: img {img.shape}, imclean {imclean.shape}")
    squared_error = (img - imclean) ** 2
    if img.ndim == 4: mse_per_image = torch.mean(squared_error, dim=(1, 2, 3))
    elif img.ndim == 3: mse_per_image = torch.mean(squared_error, dim=(1, 2))
    else: raise ValueError(f"Unsupported input dimensions: {img.ndim}. Expected 3 or 4.")
    epsilon = 1e-10
    psnr_per_image = 10.0 * torch.log10((data_range**2) / (mse_per_image + epsilon))
    return torch.mean(psnr_per_image)

@torch.no_grad()
def calculate_masked_psnr_original_scale(pred_original_img, gt_original_img, mask, patchify_func, in_chans, original_data_range):
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

def time2file_name(time_str): return time_str.replace(" ", "_").replace(":", "-").split('.')[0]

class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        save_dir = os.path.dirname(filename)
        best_filename = os.path.join(save_dir, 'model_best.pth')
        shutil.copyfile(filename, best_filename)


# --- Main Training Function ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Save Directory and Logger ---
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    dataset_tag = "imagenet" if args.imagenet else "custom"
    save_dir = os.path.join(args.save_dir, f"{dataset_tag}_{args.model}_{args.patch_size}_{date_time}") # Include patch_size
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created save directory: {save_dir}")
    else:
        print(f"Save directory exists: {save_dir}")
    writer = tensorboardX.SummaryWriter(log_dir=save_dir)

    # --- Determine Dataset Parameters & Load Datasets ---
    if args.imagenet:
        print("Using ImageNet-1k Dataset")
        num_input_channels = 3
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        train_dir = os.path.join(args.imagenet_path, 'train')
        val_dir = os.path.join(args.imagenet_path, 'val')
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
             raise FileNotFoundError(f"ImageNet train ({train_dir}) or val ({val_dir}) directory not found under {args.imagenet_path}")

        # Use your MAEImageNetDataset class
        train_dataset = MAEImageNetDataset(train_dir, train=True, img_size=args.img_size)
        test_dataset = MAEImageNetDataset(val_dir, train=False, img_size=args.img_size) # Use train=False for validation transforms

    else: # Use Custom Dataset
        print("Using Custom Dataset")
        if MAEDataset is None or MAEDatasetEval is None:
             raise ImportError("Custom dataset classes MAEDataset/MAEDatasetEval could not be imported.")

        num_input_channels = 1 # Assuming custom is single channel
        mean = CUSTOM_DATASET_MEAN
        std = CUSTOM_DATASET_STD

        try:
            # Assumes MAEDataset/Eval takes transform/target_transform
            # You NEED to modify your MAEDataset to accept and use these transforms!
            train_dataset = MAEDataset(args.data_path, 
                                       args.blur, 
                                       args.kernel_size, 
                                       args.sigma,
                                       img_size=args.img_size)

            val_data_path = args.data_path.replace('train', 'valid')
            if not os.path.isdir(val_data_path):
                 print(f"Validation path {val_data_path} not found, using train path for validation.")
                 val_data_path = args.data_path
            # Assumes MAEDatasetEval takes transform applied to both data & gt
            test_dataset = MAEDatasetEval(val_data_path, 
                                          args.blur, 
                                          args.kernel_size, 
                                          args.sigma,
                                          img_size=args.img_size)
        except TypeError as e:
             print("\nERROR: Failed to initialize custom Datasets.")
             print("Please ensure MAEDataset/MAEDatasetEval accept 'transform'/'target_transform' arguments")
             print("and apply them (including ToTensor and Normalize) internally.")
             print(f"Original error: {e}")
             return

    # --- Create DataLoaders (Same logic) ---
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  drop_last=True,
                                  pin_memory=True, 
                                  num_workers=args.num_workers,
                                  persistent_workers=True if args.num_workers > 0 else False)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.eval_batch_size, 
                                 shuffle=False,
                                 num_workers=args.num_workers, 
                                 pin_memory=True)
    print(f"Train Dataloader: {len(train_dataloader)} batches ({len(train_dataset)} samples)")
    print(f"Test Dataloader: {len(test_dataloader)} batches ({len(test_dataset)} samples)")


    # --- Model Definition ---
    print(f"Loading MAE model: {args.model}, Patch Size: {args.patch_size}, Input Channels: {num_input_channels}")
    # Pass patch_size and in_chans to model constructor
    # Ensure your model functions use these arguments!
    model_args = {'patch_size': args.patch_size, 'in_chans': num_input_channels}
    if args.model == 'tiny':
        # Example: Assuming mae_vit_tiny_mine accepts these args
        model = mae_vit_tiny_mine()
    elif args.model == 'base':
        # Example: Assuming mae_vit_base_patch16 accepts these args
        model = mae_vit_base_patch16()
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Remove Sigmoid if exists (Keep this check)
    if isinstance(model.decoder_pred, nn.Sequential) and isinstance(model.decoder_pred[-1], nn.Sigmoid):
        print("Removing Sigmoid layer from decoder_pred.")
        model.decoder_pred = model.decoder_pred[:-1]

    model.to(device)
    print(f"Number of trainable params (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.e6:.2f}")


    # --- Optimizer ---
    # Apply LR scaling based on a reference batch size (e.g., 256 or 512 for ImageNet MAE)
    # linear scaling rule: lr = base_lr * total_batch_size / ref_batch_size
    ref_batch_size = 64 # Common reference for scaling
    effective_lr = args.learning_rate * args.batch_size / ref_batch_size
    print(f"Base LR: {args.learning_rate}, Ref BS: {ref_batch_size}, Actual BS: {args.batch_size}")
    print(f"Applying LR Scaling: Effective LR = {args.learning_rate} * {args.batch_size} / {ref_batch_size} = {effective_lr:.2e}")

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=effective_lr, # Use scaled LR
                                  weight_decay=args.weight_decay, 
                                  betas=(0.9, 0.95))
    print(f"Optimizer: AdamW, LR: {effective_lr:.2e}, Weight Decay: {args.weight_decay}")


    # --- LR Scheduler (Keep as is) ---
    num_training_steps = args.end_epoch * len(train_dataloader)
    warmup_steps = args.warmup_epochs * len(train_dataloader)
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {warmup_steps} ({args.warmup_epochs} epochs)")
    scheduler = CosineLRScheduler(
        optimizer=optimizer, 
        t_initial=num_training_steps, 
        lr_min=5e-5,
        warmup_t=warmup_steps, 
        warmup_lr_init=1e-6, 
        warmup_prefix=True,
        cycle_decay=1.0, 
        t_in_epochs=False
    )

    # --- Resume / Finetune (Keep as is, uses 'best_val_masked_psnr') ---
    start_epoch = args.start_epoch
    best_val_masked_psnr = 0. # Track best validation masked PSNR
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items(): name = k[7:] if k.startswith('module.') else k; new_state_dict[name] = v
            msg = model.load_state_dict(new_state_dict, strict=False); print(f"   Model loaded with status: {msg}")
            if 'optimizer' in checkpoint and not args.finetune:
                try: optimizer.load_state_dict(checkpoint['optimizer']); print("   Optimizer state loaded.")
                except ValueError as e: print(f"   Warning: Could not load optimizer state: {e}")
            if 'scheduler' in checkpoint and not args.finetune:
                 try: scheduler.load_state_dict(checkpoint['scheduler']); print("   Scheduler state loaded.")
                 except Exception as e: print(f"   Warning: Could not load scheduler state: {e}")
            if 'epoch' in checkpoint and not args.finetune:
                 start_epoch = checkpoint['epoch'] + 1; print(f"   Resuming from epoch {start_epoch}")
            if 'best_val_masked_psnr' in checkpoint: best_val_masked_psnr = checkpoint['best_val_masked_psnr']; print(f"   Previous best validation masked PSNR: {best_val_masked_psnr:.4f}")
            elif 'best_psnr' in checkpoint: best_val_masked_psnr = checkpoint['best_psnr']; print(f"   Loaded previous best (likely full) PSNR: {best_val_masked_psnr:.4f}")
        else: print(f"=> no checkpoint found at '{args.resume}'")
    elif args.finetune:
         if os.path.isfile(args.finetune):
            print(f"=> loading weights for finetuning '{args.finetune}'")
            checkpoint = torch.load(args.finetune, map_location='cpu')
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
            else: state_dict = checkpoint
            new_state_dict = {}
            for k, v in state_dict.items(): name = k[7:] if k.startswith('module.') else k; new_state_dict[name] = v
            msg = model.load_state_dict(new_state_dict, strict=False); print(f"   Weights loaded with status: {msg}")
            start_epoch = 0
         else: print(f"=> no weights found at '{args.finetune}'")


    # --- Prepare mean/std tensors (Keep as is) ---
    mean_tensor = torch.tensor(mean, device=device).view(1, num_input_channels, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, num_input_channels, 1, 1)


    # --- Training Loop (Keep mostly as is, uses correct de-normalization) ---
    print(f"Starting training from epoch {start_epoch + 1} to {args.end_epoch}")
    for epoch_i in range(start_epoch, args.end_epoch):
        model.train()
        epoch_loss_meter = AverageMeter()
        epoch_train_psnr_full = AverageMeter()
        epoch_train_psnr_masked = AverageMeter()
        epoch_start_time = time.time()

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', lr, current_iter if 'current_iter' in locals() else epoch_i * len(train_dataloader)) # Log LR

        training_bar = tqdm(train_dataloader, desc=f"[Epoch {epoch_i + 1}/{args.end_epoch}] Train", colour='yellow', ncols=125)

        for idx, (data, gt) in enumerate(training_bar):
            current_iter = epoch_i * len(train_dataloader) + idx
            data = data.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()

            loss, pred_patches, mask = model(data, 
                                             gt=gt, 
                                             mask_ratio=args.mask_ratio)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: NaN or Inf loss detected at epoch {epoch_i+1}, iter {idx}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step_update(current_iter)

            with torch.no_grad():
                gt_original = gt * std_tensor + mean_tensor
                gt_original_clamped = torch.clamp(gt_original, 0., ORIGINAL_DATA_RANGE)

                pred_img_standardized = model.unpatchify(pred_patches, in_chan=num_input_channels)
                pred_original = pred_img_standardized * std_tensor + mean_tensor
                pred_original_clamped = torch.clamp(pred_original, 0., ORIGINAL_DATA_RANGE)

                psnr_train_full = batch_PSNR(pred_original_clamped, 
                                             gt_original_clamped, 
                                             ORIGINAL_DATA_RANGE)
                psnr_train_masked = calculate_masked_psnr_original_scale(
                    pred_original_clamped, 
                    gt_original_clamped, mask,
                    model.patchify, 
                    num_input_channels, 
                    ORIGINAL_DATA_RANGE
                )

            batch_size = data.size(0)
            epoch_loss_meter.update(loss.item(), batch_size)
            epoch_train_psnr_full.update(psnr_train_full.item(), batch_size)
            epoch_train_psnr_masked.update(psnr_train_masked.item(), batch_size)

            training_bar.set_postfix({
                "Loss": f"{epoch_loss_meter.avg:.4f}",
                "Tr PSNR Full": f"{epoch_train_psnr_full.avg:.2f}",
                "Tr PSNR Mask": f"{epoch_train_psnr_masked.avg:.2f}" })
            if current_iter % 100 == 0:
                 writer.add_scalar('Train/iter_loss', loss.item(), current_iter)
                 writer.add_scalar('Train/iter_psnr_full', psnr_train_full.item(), current_iter)
                 writer.add_scalar('Train/iter_psnr_masked', psnr_train_masked.item(), current_iter)

        print(f"Epoch {epoch_i + 1} Train Summary: Loss: {epoch_loss_meter.avg:.4f}, "
              f"PSNR Full: {epoch_train_psnr_full.avg:.2f} dB, PSNR Masked: {epoch_train_psnr_masked.avg:.2f} dB")
        writer.add_scalar('Train/epoch_avg_loss', epoch_loss_meter.avg, epoch_i + 1)
        writer.add_scalar('Train/epoch_avg_psnr_full', epoch_train_psnr_full.avg, epoch_i + 1)
        writer.add_scalar('Train/epoch_avg_psnr_masked', epoch_train_psnr_masked.avg, epoch_i + 1)
        epoch_end_time = time.time(); print(f'Epoch {epoch_i + 1} Time: {epoch_end_time - epoch_start_time:.2f}s')

        # --- Evaluation (Keep as is, uses correct de-normalization & masked PSNR) ---
        if (epoch_i + 1) % args.eval_interval == 0:
            model.eval()
            val_loss_meter = AverageMeter()
            val_psnr_full = AverageMeter()
            val_psnr_masked = AverageMeter()
            print(f'--- Validation Epoch {epoch_i + 1} ---')
            val_bar = tqdm(test_dataloader, desc=f" Val", colour='blue', ncols=125)
            visible_batch_idx = random.randint(0, len(test_dataloader) - 1)

            for idx, (data, gt) in enumerate(val_bar):
                data = data.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                with torch.no_grad():
                    loss, pred_patches, mask = model(data, gt=gt, mask_ratio=args.eval_mask_ratio) # Get mask here too
                    gt_original = gt * std_tensor + mean_tensor
                    gt_original_clamped = torch.clamp(gt_original, 0., ORIGINAL_DATA_RANGE)
                    pred_img_standardized = model.unpatchify(pred_patches, in_chan=num_input_channels)
                    pred_original = pred_img_standardized * std_tensor + mean_tensor
                    pred_original_clamped = torch.clamp(pred_original, 0., ORIGINAL_DATA_RANGE)

                    psnr_val_full = batch_PSNR(pred_original_clamped, 
                                               gt_original_clamped, 
                                               ORIGINAL_DATA_RANGE)
                    # Pass the mask obtained from validation forward pass
                    psnr_val_masked = calculate_masked_psnr_original_scale(
                        pred_original_clamped, 
                        gt_original_clamped, 
                        mask, # Pass the validation mask
                        model.patchify, 
                        num_input_channels, 
                        ORIGINAL_DATA_RANGE
                    )

                    if idx == visible_batch_idx: # Save images logic (keep as is)
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

                batch_size = data.size(0)
                val_loss_meter.update(loss.item(), batch_size)
                val_psnr_full.update(psnr_val_full.item(), batch_size)
                val_psnr_masked.update(psnr_val_masked.item(), batch_size)
                val_bar.set_postfix({ "Loss": f"{val_loss_meter.avg:.4f}", "PSNR Full": f"{val_psnr_full.avg:.2f}", "PSNR Mask": f"{val_psnr_masked.avg:.2f}"})

            avg_val_loss = val_loss_meter.avg
            avg_val_psnr_full = val_psnr_full.avg
            avg_val_psnr_masked = val_psnr_masked.avg
            print(f"Epoch {epoch_i + 1} Validation Summary: Loss: {avg_val_loss:.4f}, "
                  f"PSNR Full: {avg_val_psnr_full:.2f} dB, PSNR Masked: {avg_val_psnr_masked:.2f} dB")
            writer.add_scalar("Val/epoch_avg_loss", avg_val_loss, epoch_i + 1)
            writer.add_scalar("Val/epoch_avg_psnr_full", avg_val_psnr_full, epoch_i + 1)
            writer.add_scalar("Val/epoch_avg_psnr_masked", avg_val_psnr_masked, epoch_i + 1)

            # --- Save Checkpoint Logic (Based on Val Masked PSNR) ---
            is_best = avg_val_psnr_masked > best_val_masked_psnr
            if is_best:
                best_val_masked_psnr = avg_val_psnr_masked
                print(f"*** New Best Val Masked PSNR: {best_val_masked_psnr:.4f} ***")
            save_dict = { 'epoch': epoch_i, 'state_dict': model.state_dict(), 'best_val_masked_psnr': best_val_masked_psnr,
                          'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'args': args }
            filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch_i + 1}.pth')
            save_checkpoint(save_dict, is_best, filename=filename)
            if is_best: 
                print(f"Best model updated: {os.path.join(save_dir, 'model_best.pth')}")

    print(f"Training finished. Best Validation Masked PSNR: {best_val_masked_psnr:.4f}")
    writer.close()


# --- Argument Parser (Keep as is) ---
def parse_args():
    parser = argparse.ArgumentParser(description='MAE Training on Single GPU')
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
    parser.add_argument('--eval_mask_ratio', type=float, default=0.0, help='Masking ratio for evaluation')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--end_epoch', type=int, default=400, help='Total number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='Base learning rate (scaled in script)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='Warmup epochs')
    parser.add_argument('--save_dir', type=str, default='./mae_checkpoints', help='Save directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Data loading workers')
    parser.add_argument('--seed', type=int, default=114514, help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint path')
    parser.add_argument('--finetune', type=str, default=None, help='Finetune weights path')
    args = parser.parse_args()
    if args.imagenet:
        if args.imagenet_path is None: parser.error("--imagenet requires --imagenet_path.")
    else:
        if args.data_path is None: parser.error("Custom dataset requires --data_path.")
    return args

if __name__ == "__main__":
    args = parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Consider setting deterministic False for speed unless strict reproducibility is needed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)