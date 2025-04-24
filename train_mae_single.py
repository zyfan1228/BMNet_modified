import os
import time
import shutil
import datetime
import argparse 

import torch
import numpy as np
from tqdm import tqdm
import tensorboardX
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler

from model import mae_vit_tiny_mine, mae_vit_base_patch16
from data.dotadataset import MAEDataset, MAEDatasetEval

# --- Assume these are in separate files or defined here ---
# from data import MAEDataset, MAEDatasetEval # Or define below
# from model import mae_vit_tiny_mine, mae_vit_base_patch16 # Or define below
# from utils import batch_PSNR, time2file_name, AverageMeter # Or define below
# --- End Assumptions ---

# --- Placeholder for Utilities (if not imported) ---
def batch_PSNR(img, imclean, data_range):
    # Basic PSNR calculation (replace with your actual implementation)
    # Ensure img and imclean are torch tensors in the same range [0, data_range]
    mse = torch.mean((img - imclean) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))

def time2file_name(time_str):
    # Basic time formatting (replace with yours)
    return time_str.replace(" ", "_").replace(":", "-").split('.')[0]

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
        self.avg = self.sum / self.count
# --- End Placeholder Utilities ---

# --- Placeholder for Model (if not imported) ---
# IMPORTANT: Make sure the actual model definition in model.py includes the fixes!
# e.g. from model import mae_vit_tiny_mine, mae_vit_base_patch16
# Defining a dummy here just for script structure:
# from model import mae_vit_tiny_mine, mae_vit_base_patch16 # Assuming you have this
# --- End Placeholder Model ---


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Save Directory
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    save_dir = os.path.join(args.save_dir, date_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created save directory: {save_dir}")
    else:
        print(f"Save directory exists: {save_dir}")

    # TensorBoard Writer
    writer = tensorboardX.SummaryWriter(log_dir=save_dir)

    # Datasets and DataLoaders
    train_dir = args.data_path
    # Try to find validation path, default to using train path if not easily replaceable
    test_dir = train_dir.replace('train', 'valid')
    if not os.path.exists(os.path.join(test_dir, 'gt')):
        print(f"Validation path {test_dir} not found, using train path {train_dir} for validation.")
        test_dir = train_dir # Fallback to train data for validation if valid path doesn't exist

    train_dataset = MAEDataset(train_dir,
                               args.blur,
                               args.kernel_size,
                               args.sigma,
                               img_size=args.img_size) # Pass img_size
    test_dataset = MAEDatasetEval(test_dir,
                                  args.blur,
                                  args.kernel_size,
                                  args.sigma,
                                  img_size=args.img_size) # Pass img_size

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  persistent_workers=True if args.num_workers > 0 else False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.eval_batch_size, # Use separate batch size for eval
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
    print(f"Train Dataloader: {len(train_dataloader)} batches")
    print(f"Test Dataloader: {len(test_dataloader)} batches")


    # Model Definition
    print(f"Loading MAE model: {args.model}")
    if args.model == 'tiny':
        model = mae_vit_tiny_mine() # Ensure this uses in_chans=1 and norm_pix_loss=False
    elif args.model == 'base':
         # WARNING: ViT-Base is likely too large for 1300 images!
        print("WARNING: Using mae_vit_base_patch16, which is likely too large for the dataset.")
        model = mae_vit_base_patch16() # Ensure this uses in_chans=1 and norm_pix_loss=False
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # --- CRITICAL FIX REMINDER ---
    print("Reminder: Ensure the MaskedAutoencoderViT class in model.py has the fix:")
    print("  1. `forward_loss` uses `gt` (clean image) to compute the target.")
    print("  2. `forward_loss` has divide-by-zero protection for `mask.sum()`.")
    # ---

    model.to(device)

    # Print model parameter count
    tot_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable params (M): {tot_grad_params / 1.e6:.2f}')

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate * args.batch_size / 256,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
    print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")


    # LR Scheduler
    num_training_steps = args.end_epoch * len(train_dataloader)
    warmup_steps = args.warmup_epochs * len(train_dataloader)
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {warmup_steps} ({args.warmup_epochs} epochs)")
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=num_training_steps,
        lr_min=1e-6,  # Keep lr_min relatively standard
        warmup_t=warmup_steps,
        warmup_lr_init=1e-6, # Start warmup from a low value
        warmup_prefix=True,
        cycle_decay=0.1, # Standard decay for cosine cycle, 1 means no decay within cycle
        t_in_epochs=False # Step based on iterations
    )

    # Resume / Finetune (Simplified for single GPU)
    start_epoch = args.start_epoch
    best_psnr = 0.
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            # Load checkpoint to CPU first to avoid GPU memory issues
            checkpoint = torch.load(args.resume, map_location='cpu')
            # Basic state dict loading, remove 'module.' prefix if saved from DDP
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False) # Allow non-strict loading

            # Load optimizer and scheduler state if available
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and isinstance(scheduler, CosineLRScheduler): # Check type if needed
                 scheduler.load_state_dict(checkpoint['scheduler']) # Load scheduler state too
            if 'epoch' in checkpoint:
                 start_epoch = checkpoint['epoch'] + 1 # Start from next epoch
                 print(f"   Resuming from epoch {start_epoch}")
            if 'best_psnr' in checkpoint:
                 best_psnr = checkpoint['best_psnr']
                 print(f"   Previous best PSNR: {best_psnr:.4f}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            start_epoch = 0 # Start from scratch if resume path invalid
    elif args.finetune: # Basic finetune (load weights only)
         if os.path.isfile(args.finetune):
            print(f"=> loading weights for finetuning '{args.finetune}'")
            checkpoint = torch.load(args.finetune, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"   Weights loaded with status: {msg}")
            start_epoch = 0 # Start training from epoch 0 for finetuning
         else:
             print(f"=> no weights found at '{args.finetune}'")
             start_epoch = 0


    # Training Loop
    print(f"Starting training from epoch {start_epoch + 1} to {args.end_epoch}")
    for epoch_i in range(start_epoch, args.end_epoch):
        model.train()
        epoch_loss_meter = AverageMeter()
        epoch_psnr_meter = AverageMeter()
        epoch_start_time = time.time()

        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch_i + 1}/{args.end_epoch} - LR: {lr:.2e}")
        writer.add_scalar('lr', lr, epoch_i + 1)

        training_bar = tqdm(train_dataloader,
                            desc=f"[Epoch {epoch_i + 1}/{args.end_epoch}] Train",
                            colour='yellow',
                            ncols=120) # Adjust ncols as needed

        for idx, (data, gt) in enumerate(training_bar):
            current_iter = epoch_i * len(train_dataloader) + idx
            data = data.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()

            # --- Model Forward ---
            # Ensure your model's forward method handles unpatch_pred=True correctly
            # and that the loss calculation uses 'gt' as the target (FIXED IN MODEL FILE)
            loss, out_train, _ = model(data, gt, unpatch_pred=True, mask_ratio=args.mask_ratio)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: NaN or Inf loss detected at epoch {epoch_i+1}, iter {idx}. Skipping batch.")
                # Consider logging more info or stopping training
                continue # Skip this batch

            loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Step the scheduler after each iteration
            scheduler.step_update(current_iter)

            # Clamp output for PSNR calculation
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, gt, 1.0) # data_range is 1.0 for [0,1]

            epoch_loss_meter.update(loss.item(), data.size(0))
            epoch_psnr_meter.update(psnr_train.item(), data.size(0))

            # Logging to tqdm
            training_bar.set_postfix({
                "Loss": f"{epoch_loss_meter.avg:.4f}",
                "PSNR": f"{epoch_psnr_meter.avg:.2f}"
            })

            # Logging to TensorBoard (maybe less frequently)
            if current_iter % 50 == 0: # Log every 50 iterations
                 writer.add_scalar('Train/iter_loss', loss.item(), current_iter)
                 writer.add_scalar('Train/iter_psnr', psnr_train.item(), current_iter)


        print(f"Epoch {epoch_i + 1} Train Summary: Avg Loss: {epoch_loss_meter.avg:.4f}, Avg PSNR: {epoch_psnr_meter.avg:.2f} dB")
        writer.add_scalar('Train/epoch_avg_loss', epoch_loss_meter.avg, epoch_i + 1)
        writer.add_scalar('Train/epoch_avg_psnr', epoch_psnr_meter.avg, epoch_i + 1)
        epoch_end_time = time.time()
        print(f'Epoch {epoch_i + 1} Time: {epoch_end_time - epoch_start_time:.2f}s')


        # Evaluation (run every 'eval_interval' epochs)
        if (epoch_i + 1) % args.eval_interval == 0:
            model.eval()
            val_psnr_meter = AverageMeter()
            print(f'--- Running Validation for Epoch {epoch_i + 1} ---')
            val_bar = tqdm(test_dataloader,
                           desc=f"[Epoch {epoch_i + 1}/{args.end_epoch}] Val",
                           colour='blue',
                           ncols=120)
            
            valid_avg_loss = 0

            for data, gt in val_bar:
                data = data.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

                with torch.no_grad():
                    # Use eval_mask_ratio for evaluation consistency or 0.0 for best reconstruction
                    loss, model_out, _ = model(data, gt, 
                                               unpatch_pred=True, 
                                               mask_ratio=args.eval_mask_ratio)
                    model_out = torch.clamp(model_out, 0., 1.)
                
                valid_avg_loss += loss.cpu().item()
                psnr_test = batch_PSNR(model_out, gt, 1.0)
                val_psnr_meter.update(psnr_test.item(), data.size(0))
                val_bar.set_postfix({"Val PSNR": f"{val_psnr_meter.avg:.2f}"})

            valid_avg_loss = valid_avg_loss / len(val_bar)
            avg_val_psnr = val_psnr_meter.avg
            print(f"Epoch {epoch_i + 1} Validation Summary: Avg PSNR: {avg_val_psnr:.4f} dB, Avg Loss: {valid_avg_loss:.4f}")
            writer.add_scalar("Val/epoch_avg_psnr", avg_val_psnr, epoch_i + 1)

            # Save Checkpoint Logic
            is_best = avg_val_psnr > best_psnr
            if is_best:
                best_psnr = avg_val_psnr
                print(f"*** New Best PSNR: {best_psnr:.4f} ***")

            save_dict = {
                'epoch': epoch_i,
                'state_dict': model.state_dict(), # No need for module. prefix on single GPU
                'best_psnr': best_psnr,
                # 'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(), # Save scheduler state
                'args': args # Save args for reference
            }
            filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch_i + 1}.pth')
            save_checkpoint(save_dict, is_best, filename=filename)
            print(f"Checkpoint saved to {filename}")
            if is_best:
                 print(f"Best model checkpoint updated in {save_dir}/model_best.pth")

    print(f"Training finished. Best Validation PSNR: {best_psnr:.4f}")
    writer.close()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Saves checkpoint, and optionally creates/updates 'model_best.pth'"""
    torch.save(state, filename)
    if is_best:
        # Get the directory of the filename
        save_dir = os.path.dirname(filename)
        best_filename = os.path.join(save_dir, 'model_best.pth')
        shutil.copyfile(filename, best_filename)


def parse_args():
    parser = argparse.ArgumentParser(description='MAE Training on Single GPU')

    # Data Parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data directory (containing a "gt" subfolder)')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing/cropping')

    # MAE Specific Data Parameters (Dataset related)
    parser.add_argument('--blur', action='store_true', help='Apply Gaussian blur to input images')
    parser.add_argument('--kernel_size', type=int, default=3, help='Gaussian kernel size if --blur')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian sigma if --blur')

    # Model Parameters
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'base'],
                        help="MAE model size ('tiny' or 'base')")
    parser.add_argument('--mask_ratio', type=float, default=0.6, help='Masking ratio for training')
    parser.add_argument('--eval_mask_ratio', type=float, default=0.0,
                        help='Masking ratio for evaluation (0.0 for best reconstruction)')

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch (for resuming)')
    parser.add_argument('--end_epoch', type=int, default=500, help='Total number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Initial learning rate for AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for AdamW') # Adjusted default
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')

    # Environment/Logging Parameters
    parser.add_argument('--save_dir', type=str, default='./mae_model_ckpt_single', help='Directory to save checkpoints and logs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluate on validation set every N epochs') # Changed from 10

    # Resume/Finetune Parameters
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--finetune', type=str, default=None, help='Path to weights to start finetuning from (loads weights only)')


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Set Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU (but this script is for single)
    np.random.seed(args.seed)
    # random.seed(args.seed) # If using python random

    # Optional: Set deterministic behavior (can slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    main(args)