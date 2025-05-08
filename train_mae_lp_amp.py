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
import tensorboardX # Or torch.utils.tensorboard
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# import torchvision.transforms.functional as F # Not directly used in this version of dataset
from timm.scheduler import CosineLRScheduler
from PIL import Image

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- AMP: Import GradScaler and autocast ---
from torch.cuda.amp import GradScaler, autocast

# --- Import YOUR MAE model definitions ---
# Make sure 'model.py' is in your PYTHONPATH or in the same directory
try:
    import model as model_module
except ImportError:
    print("ERROR: Could not import 'model.py'. Make sure it's in your Python path.")
    print("Please place your MAE model definitions (MaskedAutoencoderViT, mae_vit_base_patch16, etc.) in a file named 'model.py'.")
    sys.exit(1)


# --- DDP Helper functions (Copied from your MAE script) ---
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'SLURM_PROCID' in os.environ: # SLURM DDP
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
        args.rank = 0; args.world_size = 1; args.gpu = 0
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}, world {args.world_size}, gpu {args.gpu}): {args.dist_url}', flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    print("Distributed training initialized.")

def cleanup():
    if dist.is_initialized():
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

def time2file_name(time_str): return time_str.replace(" ", "_").replace(":", "-").split('.')[0]

# --- Dataset for Linear Probing ---
class LinearProbeImageNetDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 img_size=224,
                 random_crop_scale=(0.08, 1.0), # Typical for ImageNet classification
                 random_crop_ratio=(3./4., 4./3.),
                 hflip_prob=0.5,
                 norm_mean=[0.485, 0.456, 0.406], # ImageNet defaults
                 norm_std=[0.229, 0.224, 0.225]
                 ):
        self.root = root
        self.train = train
        self.img_size = img_size

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, 
                                             scale=random_crop_scale, 
                                             ratio=random_crop_ratio, 
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=hflip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
        else: # Validation/Test transforms
            self.transform = transforms.Compose([
                transforms.Resize(int(img_size * 256 / 224), 
                                  interpolation=transforms.InterpolationMode.BICUBIC), # Standard val resize
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
        try:
            self.image_folder = datasets.ImageFolder(self.root, transform=self.transform)
            if not self.image_folder.samples and is_main_process():
                print(f"Warning: No images found in {self.root} or its subdirectories for LinearProbeImageNetDataset.")
            if is_main_process():
                print(f"Initialized LinearProbeImageNetDataset for {'Training' if train else 'Validation'}")
                print(f"  Root: {root}, Image Size: {img_size}, Num classes: {len(self.image_folder.classes)}")
        except FileNotFoundError:
            if is_main_process(): print(f"Error: Root directory not found for Linear Probe: {self.root}")
            raise
        except Exception as e:
            if is_main_process(): print(f"Error initializing ImageFolder for Linear Probe at {self.root}: {e}")
            raise

    def __getitem__(self, index):
        return self.image_folder[index] # Returns (image_tensor, label)

    def __len__(self):
        return len(self.image_folder)

    def get_num_classes(self):
        return len(self.image_folder.classes)


# --- AverageMeter (Copied from your MAE script) ---
class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count if self.count != 0 else 0
    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized(): return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier(); dist.all_reduce(t)
        t = t.tolist(); self.sum = t[0]; self.count = t[1]
        self.avg = self.sum / self.count if self.count != 0 else 0

# --- save_checkpoint (Modified for Linear Probing: best_val_accuracy) ---
def save_checkpoint(state, is_best, filename='linprobe_checkpoint.pth', save_dir='.'):
    if is_main_process():
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        torch.save(state, filepath)
        if is_best:
            best_filename = os.path.join(save_dir, 'linprobe_model_best.pth')
            shutil.copyfile(filepath, best_filename)
            print(f"Rank 0: Best linear probe model updated: {best_filename}")

# --- Accuracy Calculation ---
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# --- MAE Encoder Wrapper for Linear Probing ---
class MAEEncoderForLinprobe(nn.Module):
    def __init__(self, mae_full_model):
        super().__init__()
        self.mae_full_model = mae_full_model # This is the complete MaskedAutoencoderViT instance

    def forward(self, imgs):
        # For linear probing, mask_ratio should be 0.0, meaning no masking.
        # The forward_encoder method of MaskedAutoencoderViT returns: latent_tokens, mask, ids_restore
        # We are interested in the latent_tokens, specifically the CLS token representation.
        latent_tokens, _, _ = self.mae_full_model.forward_encoder(imgs, mask_ratio=0.0)
        # CLS token is typically the first token in the sequence
        cls_token_feature = latent_tokens[:, 0] # Shape: [N, embed_dim]
        return cls_token_feature


# --- Main Linear Probing Function ---
def main_linear_probe(args):
    init_distributed_mode(args)
    device = torch.device(f"cuda:{args.gpu}")
    if is_main_process(): print(f"Rank {args.rank} using device: {device} for Linear Probing.")

    writer = None
    save_dir_linprobe = None
    if is_main_process():
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        dataset_tag = "imagenet100" if "100" in args.imagenet_path.lower() else "imagenet1k" # Basic check
        blur_sigma_tag = f"_sigma{args.sigma_for_tag}" if args.sigma_for_tag is not None else "_baseline"
        model_tag = args.model
        save_dir_linprobe = os.path.join(args.save_dir_linprobe, f"linprobe_{dataset_tag}_{model_tag}{blur_sigma_tag}_{date_time}")
        if not os.path.exists(save_dir_linprobe): os.makedirs(save_dir_linprobe, exist_ok=True)
        writer = tensorboardX.SummaryWriter(log_dir=save_dir_linprobe)
        print(f"Rank 0: Linear probe save directory: {save_dir_linprobe}")

    # --- Datasets for Linear Probing ---
    train_dir = os.path.join(args.imagenet_path, 'train')
    val_dir = os.path.join(args.imagenet_path, 'val')
    if is_main_process():
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            print(f"ERROR: ImageNet train ({train_dir}) or val ({val_dir}) directory not found under {args.imagenet_path} for linprobe")
            if args.distributed: dist.barrier(); sys.exit(1)
    try:
        train_dataset_linprobe = LinearProbeImageNetDataset(
            train_dir, 
            train=True, 
            img_size=args.img_size
        )
        val_dataset_linprobe = LinearProbeImageNetDataset(
            val_dir, 
            train=False, 
            img_size=args.img_size
        )
        args.num_classes = train_dataset_linprobe.get_num_classes()
        if is_main_process():
            if len(train_dataset_linprobe) == 0 or len(val_dataset_linprobe) == 0:
                print(f"ERROR: Linear probe dataset loaded but is empty. Check {train_dir} and {val_dir}")
                if args.distributed: dist.barrier(); sys.exit(1)
            print(f"Number of classes for linear probing: {args.num_classes}")
    except Exception as e:
        if is_main_process(): print(f"\nERROR: Failed to initialize Linear Probe Datasets: {e}")
        if args.distributed: dist.barrier(); sys.exit(1)

    if args.distributed:
        train_sampler_linprobe = DistributedSampler(train_dataset_linprobe, num_replicas=args.world_size, rank=args.rank, shuffle=True, seed=args.seed)
        val_sampler_linprobe = DistributedSampler(val_dataset_linprobe, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        train_sampler_linprobe = None; val_sampler_linprobe = None

    def seed_worker_linprobe(worker_id):
        rank_lp = get_rank() if is_dist_avail_and_initialized() else 0
        worker_seed_lp = args.seed + rank_lp * args.num_workers + worker_id # Ensure unique seed per worker
        np.random.seed(worker_seed_lp); random.seed(worker_seed_lp)

    g_lp = torch.Generator(); g_lp.manual_seed(args.seed + get_rank())

    train_dataloader_linprobe = DataLoader(
        train_dataset_linprobe, 
        batch_size=args.batch_size_linprobe, 
        shuffle=(train_sampler_linprobe is None),
        sampler=train_sampler_linprobe, 
        drop_last=True, 
        pin_memory=True, 
        num_workers=args.num_workers,
        worker_init_fn=seed_worker_linprobe, 
        generator=g_lp, 
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_dataloader_linprobe = DataLoader(
        val_dataset_linprobe, 
        batch_size=args.batch_size_linprobe, 
        shuffle=False,
        sampler=val_sampler_linprobe, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False
    )
    if is_main_process():
        print(f"Linprobe Train Dataloader: {len(train_dataloader_linprobe)} batches/GPU")
        print(f"Linprobe Val Dataloader: {len(val_dataloader_linprobe)} batches/GPU")

    # --- Model Definition for Linear Probing ---
    if is_main_process(): print(f"Defining PRETRAINED MAE model structure: {args.model} for linear probing")
    if args.model == 'tiny': pretrained_mae_full_model = model_module.mae_vit_tiny_patch16_mine()
    elif args.model == 'small': pretrained_mae_full_model = model_module.mae_vit_small_patch16_dec192d4b()
    elif args.model == 'base': pretrained_mae_full_model = model_module.mae_vit_base_patch16()
    elif args.model == 'large': pretrained_mae_full_model = model_module.mae_vit_large_patch16()
    elif args.model == 'huge': pretrained_mae_full_model = model_module.mae_vit_huge_patch14()
    else: raise ValueError(f"Unknown MAE model type: {args.model}")

    if args.pretrained_weights_path and os.path.isfile(args.pretrained_weights_path):
        if is_main_process(): print(f"=> loading pretrained MAE weights from '{args.pretrained_weights_path}'")
        checkpoint = torch.load(args.pretrained_weights_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if state_dict_key not in checkpoint:
            if 'model' in checkpoint: state_dict_key = 'model'
            elif 'module' in checkpoint: state_dict_key = None
            else: state_dict_key = None
        mae_state_dict = checkpoint[state_dict_key] if state_dict_key else checkpoint
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in mae_state_dict.items()}
        msg = pretrained_mae_full_model.load_state_dict(new_state_dict, strict=True)
        if is_main_process(): print(f"   Pretrained MAE weights loaded into full MAE model with status: {msg}")
    else:
        if is_main_process(): print(f"ERROR: Pretrained MAE weights path '{args.pretrained_weights_path}' not found or not specified.")
        if args.distributed: dist.barrier(); sys.exit(1)

    encoder_embed_dim = pretrained_mae_full_model.embed_dim
    model_linprobe = nn.Sequential(
        MAEEncoderForLinprobe(pretrained_mae_full_model),
        nn.Linear(encoder_embed_dim, args.num_classes)
    )
    for param in model_linprobe[0].parameters(): param.requires_grad = False # Freeze MAE
    for param in model_linprobe[1].parameters(): param.requires_grad = True  # Train Linear head
    model_linprobe.to(device)

    if args.distributed:
        model_linprobe = DDP(model_linprobe, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp_linprobe = model_linprobe.module
    else:
        model_without_ddp_linprobe = model_linprobe

    if is_main_process():
        trainable_params = sum(p.numel() for p in model_linprobe.parameters() if p.requires_grad) / 1.e6
        total_params = sum(p.numel() for p in model_linprobe.parameters()) / 1.e6
        print(f"Linear Probe Model: Total params: {total_params:.2f}M, Trainable (classifier): {trainable_params:.2f}M")

    classifier_params = model_without_ddp_linprobe[1].parameters()
    optimizer_linprobe = torch.optim.SGD(classifier_params, lr=args.lr_linprobe, momentum=0.9, weight_decay=0)
    if is_main_process(): print(f"Optimizer for Linear Probe: SGD, LR: {args.lr_linprobe:.1e}, Momentum: 0.9")

    num_training_steps_linprobe = args.epochs_linprobe * len(train_dataloader_linprobe)
    warmup_steps_linprobe = args.warmup_epochs_linprobe * len(train_dataloader_linprobe)
    if is_main_process():
        print(f"Linprobe total training steps (per GPU): {num_training_steps_linprobe}")
        print(f"Linprobe warmup steps (per GPU): {warmup_steps_linprobe} ({args.warmup_epochs_linprobe} epochs)")

    scheduler_linprobe = CosineLRScheduler(
        optimizer=optimizer_linprobe, 
        t_initial=num_training_steps_linprobe, 
        lr_min=1e-6,
        warmup_t=warmup_steps_linprobe, 
        warmup_lr_init=1e-7, 
        warmup_prefix=True, 
        t_in_epochs=False
    )
    criterion_linprobe = nn.CrossEntropyLoss().to(device)
    scaler_linprobe = GradScaler()
    if is_main_process(): print("AMP GradScaler for Linear Probe initialized.")

    start_epoch_linprobe = 0; best_val_accuracy = 0.0
    if args.distributed: dist.barrier()
    if is_main_process(): print(f"Starting Linear Probing from epoch {start_epoch_linprobe + 1} to {args.epochs_linprobe}")

    for epoch_i in range(start_epoch_linprobe, args.epochs_linprobe):
        model_linprobe.train()
        epoch_loss_meter_linprobe = AverageMeter()
        epoch_top1_acc_meter_linprobe = AverageMeter()
        epoch_top5_acc_meter_linprobe = AverageMeter()

        if args.distributed and train_sampler_linprobe is not None:
            train_sampler_linprobe.set_epoch(epoch_i)

        training_bar_linprobe = tqdm(train_dataloader_linprobe,
                                     desc=f"[LinProbe Epoch {epoch_i + 1}/{args.epochs_linprobe}]Train",
                                     colour='cyan', ncols=160, disable=not is_main_process())

        for idx, (images, target_labels) in enumerate(training_bar_linprobe):
            current_iter_linprobe = epoch_i * len(train_dataloader_linprobe) + idx
            images = images.to(device, non_blocking=True)
            target_labels = target_labels.to(device, non_blocking=True)
            optimizer_linprobe.zero_grad()
            with autocast():
                output_logits = model_linprobe(images)
                loss_linprobe = criterion_linprobe(output_logits, target_labels)
            if torch.isnan(loss_linprobe) or torch.isinf(loss_linprobe):
                print(f"WARNING Rank {get_rank()}: NaN/Inf linprobe loss detected BEFORE scaling. Skipping step.")
                optimizer_linprobe.zero_grad(); continue
            scaler_linprobe.scale(loss_linprobe).backward()
            scaler_linprobe.step(optimizer_linprobe)
            scaler_linprobe.update()
            scheduler_linprobe.step_update(current_iter_linprobe)

            acc1, acc5 = accuracy(output_logits.float(), target_labels, topk=(1, 5))
            batch_size_linprobe = images.size(0)
            epoch_loss_meter_linprobe.update(loss_linprobe.item(), batch_size_linprobe)
            epoch_top1_acc_meter_linprobe.update(acc1.item(), batch_size_linprobe)
            epoch_top5_acc_meter_linprobe.update(acc5.item(), batch_size_linprobe)

            if is_main_process():
                lr_linprobe = optimizer_linprobe.param_groups[0]['lr']
                training_bar_linprobe.set_postfix({
                    "Loss": f"{epoch_loss_meter_linprobe.val:.4f} ({epoch_loss_meter_linprobe.avg:.4f})",
                    "Acc@1": f"{epoch_top1_acc_meter_linprobe.val:.2f}% ({epoch_top1_acc_meter_linprobe.avg:.2f}%)",
                    "LR": f"{lr_linprobe:.2e}", "Scale": f"{scaler_linprobe.get_scale():.1f}"
                })
                if current_iter_linprobe > 0 and current_iter_linprobe % 100 == 0: # Avoid logging at step 0
                    writer.add_scalar('LinProbe_Train/iter_loss', loss_linprobe.item(), current_iter_linprobe)
                    writer.add_scalar('LinProbe_Train/iter_acc1', acc1.item(), current_iter_linprobe)
                    writer.add_scalar('LinProbe_Train/lr', lr_linprobe, current_iter_linprobe)
                    writer.add_scalar('LinProbe_AMP/scale', scaler_linprobe.get_scale(), current_iter_linprobe)
        
        epoch_loss_meter_linprobe.synchronize_between_processes()
        epoch_top1_acc_meter_linprobe.synchronize_between_processes()
        epoch_top5_acc_meter_linprobe.synchronize_between_processes()
        if is_main_process():
            print(f"LinProbe Epoch {epoch_i + 1} Train Summary: Loss: {epoch_loss_meter_linprobe.avg:.4f}, "
                  f"Acc@1: {epoch_top1_acc_meter_linprobe.avg:.2f}%, Acc@5: {epoch_top5_acc_meter_linprobe.avg:.2f}%")
            writer.add_scalar('LinProbe_Train/epoch_avg_loss', epoch_loss_meter_linprobe.avg, epoch_i + 1)
            writer.add_scalar('LinProbe_Train/epoch_avg_acc1', epoch_top1_acc_meter_linprobe.avg, epoch_i + 1)
            writer.add_scalar('LinProbe_Train/epoch_avg_acc5', epoch_top5_acc_meter_linprobe.avg, epoch_i + 1)

        model_linprobe.eval()
        val_loss_meter_linprobe = AverageMeter()
        val_top1_acc_meter_linprobe = AverageMeter()
        val_top5_acc_meter_linprobe = AverageMeter()
        if is_main_process(): print(f'--- LinProbe Validation Epoch {epoch_i + 1} ---')
        val_bar_linprobe = tqdm(val_dataloader_linprobe, desc="LinProbe Val", colour='magenta', ncols=160, disable=not is_main_process())
        for images, target_labels in val_bar_linprobe:
            images = images.to(device, non_blocking=True)
            target_labels = target_labels.to(device, non_blocking=True)
            with torch.no_grad():
                with autocast():
                    output_logits = model_linprobe(images)
                    loss_val_linprobe = criterion_linprobe(output_logits, target_labels)
            acc1_val, acc5_val = accuracy(output_logits.float(), target_labels, topk=(1, 5))
            batch_size_linprobe = images.size(0)
            val_loss_meter_linprobe.update(loss_val_linprobe.item(), batch_size_linprobe)
            val_top1_acc_meter_linprobe.update(acc1_val.item(), batch_size_linprobe)
            val_top5_acc_meter_linprobe.update(acc5_val.item(), batch_size_linprobe)
            if is_main_process():
                val_bar_linprobe.set_postfix({
                    "Loss": f"{val_loss_meter_linprobe.val:.4f} ({val_loss_meter_linprobe.avg:.4f})",
                    "Acc@1": f"{val_top1_acc_meter_linprobe.val:.2f}% ({val_top1_acc_meter_linprobe.avg:.2f}%)"
                })
        val_loss_meter_linprobe.synchronize_between_processes()
        val_top1_acc_meter_linprobe.synchronize_between_processes()
        val_top5_acc_meter_linprobe.synchronize_between_processes()
        avg_val_loss_linprobe = val_loss_meter_linprobe.avg
        avg_val_top1_acc_linprobe = val_top1_acc_meter_linprobe.avg
        avg_val_top5_acc_linprobe = val_top5_acc_meter_linprobe.avg
        if is_main_process():
            print(f"LinProbe Epoch {epoch_i + 1} Val Summary (Aggregated): Loss: {avg_val_loss_linprobe:.4f}, "
                  f"Acc@1: {avg_val_top1_acc_linprobe:.2f}%, Acc@5: {avg_val_top5_acc_linprobe:.2f}%")
            writer.add_scalar("LinProbe_Val/epoch_avg_loss", avg_val_loss_linprobe, epoch_i + 1)
            writer.add_scalar("LinProbe_Val/epoch_avg_acc1", avg_val_top1_acc_linprobe, epoch_i + 1)
            writer.add_scalar("LinProbe_Val/epoch_avg_acc5", avg_val_top5_acc_linprobe, epoch_i + 1)
            is_best_linprobe = avg_val_top1_acc_linprobe > best_val_accuracy
            if is_best_linprobe:
                best_val_accuracy = avg_val_top1_acc_linprobe
                print(f"*** Rank 0: New Best LinProbe Val Acc@1: {best_val_accuracy:.2f}% ***")
            save_dict_linprobe = {
                'epoch': epoch_i,
                'classifier_state_dict': model_without_ddp_linprobe[1].state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'optimizer_linprobe': optimizer_linprobe.state_dict(),
                'scheduler_linprobe': scheduler_linprobe.state_dict(),
                'scaler_linprobe': scaler_linprobe.state_dict(), 'args_linprobe': args
            }
            filename_linprobe = f'linprobe_checkpoint_epoch_{epoch_i + 1}.pth'
            save_checkpoint(save_dict_linprobe, is_best_linprobe, filename=filename_linprobe, save_dir=save_dir_linprobe)
        if args.distributed: dist.barrier()
    if is_main_process():
        print(f"Linear Probing finished. Best Validation Acc@1: {best_val_accuracy:.2f}%")
        final_results_file = os.path.join(save_dir_linprobe, "final_accuracy.txt")
        with open(final_results_file, "w") as f:
            f.write(f"Best Validation Acc@1: {best_val_accuracy:.2f}%\n")
            f.write(f"Achieved at epoch: (Find corresponding best checkpoint)\n") # User needs to check logs/best_model
            f.write(f"Pretrained Weights: {args.pretrained_weights_path}\n")
        print(f"Final accuracy saved to {final_results_file}")
        if writer: writer.close()
    cleanup()

def parse_args_linprobe():
    parser = argparse.ArgumentParser(description='MAE Linear Probing (Multi-GPU DDP + AMP)')
    parser.add_argument('--pretrained_weights_path', type=str, required=True, help='Path to the .pth MAE pretrained model weights.')
    parser.add_argument('--model', type=str, default='base', choices=['tiny', 'small', 'base', 'large', 'huge'],
                        help='MAE model architecture (must match pretrained weights).')
    parser.add_argument('--imagenet_path', type=str, required=True, help='Path to ImageNet root (e.g., IN-100 or IN-1K).')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for linear probing.')
    parser.add_argument('--batch_size_linprobe', type=int, default=512, help='Linprobe batch size PER GPU (e.g., 256, 512)')
    parser.add_argument('--epochs_linprobe', type=int, default=90, help='Total epochs for linear probing (e.g., 90 for IN1K, 50-90 for IN100).')
    parser.add_argument('--lr_linprobe', type=float, default=0.1, help='Learning rate for linear classifier (SGD).')
    parser.add_argument('--warmup_epochs_linprobe', type=int, default=10, help='Warmup epochs for linear probing scheduler.')
    parser.add_argument('--save_dir_linprobe', type=str, default='./mae_linprobe_results', help='Save directory for linear probe results.')
    parser.add_argument('--num_workers', type=int, default=8, help='Data loading workers per GPU.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.') # Changed default seed
    parser.add_argument('--sigma_for_tag', type=float, default=None, help='Sigma value used in pretraining (for naming save dir).')
    args = parser.parse_args()
    args.distributed = False; args.rank = 0; args.world_size = 1; args.gpu = 0; args.dist_url = 'env://'
    return args

if __name__ == "__main__":
    args_linprobe = parse_args_linprobe()
    seed = args_linprobe.seed
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) # Seed all GPUs
    np.random.seed(seed); random.seed(seed)
    # Deterministic settings (can slow down training, but good for reproducibility if needed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False # Set False for deterministic
    torch.backends.cudnn.benchmark = True # Usually True for performance if input sizes don't change much

    main_linear_probe(args_linprobe)