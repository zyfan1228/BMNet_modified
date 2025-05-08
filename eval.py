# from ast import Gt
import os
# from re import I
import time
import datetime

# import cv2
import einops
# from networkx import out_degree_centrality
import numpy as np
from PIL import Image
# from sympy import im # Unused import
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from argparse import ArgumentParser


# Assume these imports are correct and available
from train_mae_single import calculate_masked_psnr_original_scale # Needs definition or import
from utils import PSNR, SSIM, time2file_name, AverageMeter      # Needs definition or import
# Make sure custom dataset paths/imports are correct if used
from model import BMNet, mae_vit_base_patch16, mae_vit_tiny_mine # Needs definition or import
# Use the correct import path/name if it's different
from train_mae_amp import MAEImageNetDataset 

CUSTOM_DATASET_MEAN = [0.330694]
CUSTOM_DATASET_STD = [0.176989]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
ORIGINAL_DATA_RANGE = 1.0


def batch_PSNR(img, imclean, data_range):
    if not isinstance(img, torch.Tensor) or not isinstance(imclean, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensor")
    if img.shape != imclean.shape:
        raise ValueError(f"Input shapes must match: img {img.shape}, imclean {imclean.shape}")
    squared_error = (img - imclean) ** 2
    if img.ndim == 4: # NCHW
        mse_per_image = torch.mean(squared_error, dim=(1, 2, 3))
    elif img.ndim == 3: # NHW
        mse_per_image = torch.mean(squared_error, dim=(1, 2))
    else:
        raise ValueError(f"Unsupported input dimensions: {img.ndim}. Expected 3 or 4.")
    epsilon = 1e-10
    psnr_per_image = 10.0 * torch.log10((data_range**2) / (mse_per_image + epsilon))
    average_psnr = torch.mean(psnr_per_image)
    return average_psnr

# ---------------------------------------------------------------------------
# Modified save_array function to handle RGB
# ---------------------------------------------------------------------------
def save_array(data_tensor, file_path):
    """Saves a torch tensor (NCHW or N1HW) as an image file."""
    if not isinstance(data_tensor, torch.Tensor):
        print(f"Warning: Input to save_array is not a tensor (type: {type(data_tensor)}). Skipping save for {file_path}")
        return

    # Assuming batch size is 1 (N=1), remove batch dimension and move to CPU
    # Add error handling for unexpected shapes
    if data_tensor.shape[0] != 1:
         print(f"Warning: save_array expects batch size 1, but got shape {data_tensor.shape}. Saving only the first item.")
    img_tensor = data_tensor[0].cpu()

    # Clamp tensor values to [0, 1] before converting
    img_tensor = torch.clamp(img_tensor, 0., 1.)

    # Convert to numpy array
    img_np = img_tensor.numpy()

    # Determine number of channels (C) from shape (C, H, W)
    if img_np.ndim == 3:
        num_channels = img_np.shape[0]
        if num_channels == 1:
            # Grayscale: remove channel dim -> (H, W)
            img_processed = img_np.squeeze(0)
            mode = 'L'
        elif num_channels == 3:
            # RGB: transpose from (C, H, W) to (H, W, C)
            img_processed = img_np.transpose(1, 2, 0)
            mode = 'RGB'
        else:
            print(f"Warning: Unsupported number of channels ({num_channels}) in save_array for {file_path}. Skipping save.")
            return
    elif img_np.ndim == 2: # Already grayscale (H, W)
         img_processed = img_np
         mode = 'L'
    else:
        print(f"Warning: Unsupported array dimension ({img_np.ndim}) after CPU conversion in save_array for {file_path}. Skipping save.")
        return

    # Scale to [0, 255] and convert to uint8
    img_uint8 = (img_processed * 255).astype(np.uint8)

    # Save using PIL
    try:
        img_pil = Image.fromarray(img_uint8, mode=mode)
        img_pil.save(file_path)
    except Exception as e:
        print(f"Error saving image {file_path}: {e}")
# ---------------------------------------------------------------------------

def eval(args, results_dir):
    try:
        from data.dotadataset import DATADataset, MAEDatasetEval
    except ImportError:
    # Handle case where custom datasets might not be present if only ImageNet is used
        if 'args' in locals() and not args.imagenet: # Check if args exists and if custom dataset is needed
            print("Warning: Could not import custom datasets. Ensure they exist if not using ImageNet.")
        DATADataset, MAEDatasetEval = None, None # Define as None to avoid later errors、

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu") # Use specified GPU
    # device = torch.device("cpu")
    cr1, cr2 = args.cs_ratio # Assuming cs_ratio is always length 2

    time_avg_meter = AverageMeter()
    psnr_avg_meter = AverageMeter()
    ssim_avg_meter = AverageMeter()
    psnr_masked_avg_meter = AverageMeter()

    # Determine input channels based on dataset type for model loading
    num_input_channels = 3 if args.imagenet else 1

    if args.mae:
        # Load correct model based on args and channels
        if args.model == 'base':
             model = mae_vit_base_patch16()
        elif args.model == 'tiny':
             model = mae_vit_tiny_mine() # Assuming tiny also takes in_chans
        else:
             raise ValueError(f"Unknown MAE model type: {args.model}")

        if args.imagenet:
            print("Using ImageNet dataset for MAE evaluation.")
            if MAEImageNetDataset is None: raise ImportError("MAEImageNetDataset not loaded.")
            test_dataset = MAEImageNetDataset(args.data_path, 
                                              train=False,
                                              img_size=args.image_size[0],
                                              apply_blur_to_input=args.blur,
                                              kernel_size=args.kernel_size,
                                              sigma=args.sigma) # Assuming square images
            mean_tensor = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).view(1, 3, 1, 1)
            std_tensor = torch.tensor(IMAGENET_DEFAULT_STD, device=device).view(1, 3, 1, 1)
        else:
            print("Using custom dataset for MAE evaluation.")
            if MAEDatasetEval is None: raise ImportError("MAEDatasetEval not loaded.")
            test_dataset = MAEDatasetEval(args.data_path,
                                          args.blur,
                                          args.kernel_size,
                                          args.sigma,
                                          img_size=args.image_size[0]) # Assuming square images
            mean_tensor = torch.tensor(CUSTOM_DATASET_MEAN, device=device).view(1, 1, 1, 1)
            std_tensor = torch.tensor(CUSTOM_DATASET_STD, device=device).view(1, 1, 1, 1)
    else: # Not MAE (BMNet case)
        print("Using BMNet evaluation.")
        # Ensure BMNet always expects 1 channel input based on original code
        if num_input_channels != 1: print("Warning: BMNet path expects 1 channel, but dataset might be RGB?")
        model = BMNet(in_chans=1, # Hardcoded based on original
                  num_stage=10,
                  embed_dim=32,
                  cs_ratio=args.cs_ratio) # Assuming BMNet takes cs_ratio

        if DATADataset is None: raise ImportError("DATADataset not loaded.")
        test_dataset = DATADataset(args.data_path,
                               cr=max(args.cs_ratio),
                               defocus=args.defocus)
        print(f"Defocus data is {args.defocus}")
        # Assume BMNet deals with single channel mean/std if needed internally or not at all
        mean_tensor = torch.tensor(CUSTOM_DATASET_MEAN, device=device).view(1, 1, 1, 1) # Default assumption
        std_tensor = torch.tensor(CUSTOM_DATASET_STD, device=device).view(1, 1, 1, 1) # Default assumption

    ckpt_path = os.path.join(args.model_path, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu') # Load to CPU first

    model_state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt)) # Handle different checkpoint formats
    # Clean 'module.' prefix if saved from DDP
    model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

    # Load weights, allow missing keys for flexibility (e.g., if only encoder is saved)
    load_result = model.load_state_dict(model_state_dict, strict=False)
    print(f"Model weights loaded from {ckpt_path} with status: {load_result}")

    model.to(device)
    model.eval()

    # Load mask only for BMNet case if present
    mask = None
    if not args.mae and 'mask' in ckpt:
        mask = ckpt['mask'].to(device)
        print("Loaded mask for BMNet.")

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1, # Keep batch size 1 for saving individual images
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True)

    print('################# Testing ##################')

    # Define indices to save images for (adjust as needed)
    num_total_images = len(test_dataloader)
    # Example: save first 5, last 5, and some in between
    save_idxs = list(range(min(5, num_total_images))) + \
                list(range(max(0, num_total_images - 5), num_total_images)) + \
                [1000, 2000, 3000, 4000]
    save_idxs = sorted(list(set(save_idxs))) # Remove duplicates and sort
    print(f"Will save images for indices: {save_idxs}")

    # --- Evaluation Loop ---
    for idx, (data, gt) in enumerate(tqdm(test_dataloader, ncols=125, colour='green')):
        data = data.to(device)
        gt = gt.to(device)
        current_mask = None # To store mask relevant to this batch for PSNR calc

        with torch.no_grad():
            if args.mae:
                # MAE forward pass
                # Assuming model returns: loss (ignored here), prediction_patches, mask
                _, out_test_patches, current_mask = model(data, 
                                                          gt=gt, 
                                                          mask_ratio=args.eval_mask_ratio if hasattr(args, 'eval_mask_ratio') else 0.75) # Use eval mask ratio if provided
                # Unpatchify prediction
                out_test = model.unpatchify(out_test_patches, in_chan=num_input_channels)
                # Denormalize output and ground truth for saving/metrics
                out_test = (out_test * std_tensor) + mean_tensor
                gt_denorm = (gt * std_tensor) + mean_tensor
                input_denorm = (data * std_tensor) + mean_tensor # Also denormalize input for saving
            else: # BMNet forward pass
                # Resize logic (kept from original)
                if args.resize_size > 0:
                    t = transforms.Resize(args.resize_size)
                    input_img = t(data)
                else:
                    input_img = data

                if mask is None:
                    raise ValueError("Mask is required for BMNet evaluation but not loaded.")
                # Assume bs=1 due to dataloader setting
                input_mask = mask.unsqueeze(0) # * model.scaler # Assuming model has scaler if needed

                # Rearrange and measure (kept from original)
                input_img = einops.rearrange(input_img,
                                            "b c (cr1 h) (cr2 w) -> b (cr1 cr2) c h w",
                                            cr1=cr1, cr2=cr2)
                meas = torch.sum(input_img * input_mask, dim=1, keepdim=True)

                # BMNet inference
                torch.cuda.synchronize()
                st = time.time()
                out_test = model(meas, input_mask)
                torch.cuda.synchronize()
                ed = time.time()
                time_avg_meter.update(ed - st)

                # Resize output if needed (kept from original)
                if args.resize_size > 0:
                    t = transforms.Resize(args.image_size) # Resize back to original size
                    out_test = t(out_test)

                # Denormalize (assuming BMNet output is in normalized space)
                # NOTE: Check if BMNet output needs denormalization. If it outputs [0,1] directly, remove this.
                out_test = (out_test * std_tensor) + mean_tensor
                gt_denorm = (gt * std_tensor) + mean_tensor
                input_denorm = (data * std_tensor) + mean_tensor # Denormalize original input
                # BMNet doesn't naturally produce a mask like MAE, set to None for PSNR calc
                current_mask = None

        # Clamp final output to valid range [0, 1] AFTER denormalization
        out_test_clamped = torch.clamp(out_test, 0., 1.)
        gt_clamped = torch.clamp(gt_denorm, 0., 1.)
        input_clamped = torch.clamp(input_denorm, 0., 1.)

        # Calculate metrics using clamped, denormalized images
        psnr_test_full = batch_PSNR(out_test_clamped, gt_clamped, ORIGINAL_DATA_RANGE)
        psnr_avg_meter.update(psnr_test_full.item()) # Use .item()
        ssim_test = SSIM(out_test_clamped, gt_clamped, ORIGINAL_DATA_RANGE) # Assuming SSIM function exists
        ssim_avg_meter.update(ssim_test.item()) # Use .item()

        # Calculate masked PSNR only if MAE and mask is available
        if args.mae and current_mask is not None:
            try:
                # Ensure patchify method is available on the model (might be model.module if DDP was used for training)
                patchify_func = getattr(model, 'patchify', None)
                if patchify_func:
                     psnr_val_masked = calculate_masked_psnr_original_scale(
                         out_test_clamped,
                         gt_clamped,
                         current_mask,
                         patchify_func,
                         num_input_channels,
                         ORIGINAL_DATA_RANGE
                     )
                     if not torch.isnan(psnr_val_masked) and not torch.isinf(psnr_val_masked):
                          psnr_masked_avg_meter.update(psnr_val_masked.item()) # Use .item()
                else:
                     print("Warning: model.patchify method not found, cannot calculate masked PSNR.")
            except Exception as e:
                 print(f"Warning: Error calculating masked PSNR for idx {idx}: {e}")


        # Save images if index matches using the modified save_array
        if idx in save_idxs:
            save_array(out_test_clamped.cpu(), file_path=os.path.join(results_dir, f"output_idx_{idx}.png"))
            save_array(gt_clamped.cpu(), file_path=os.path.join(results_dir, f"gt_idx_{idx}.png"))
            # Save the (denormalized) input image as well
            save_array(input_clamped.cpu(), file_path=os.path.join(results_dir, f"input_idx_{idx}.png"))

    # Print final average metrics
    print(f"Average Test PSNR: {psnr_avg_meter.avg:.4f}")
    if args.mae:
        print(f"Average Masked PSNR: {psnr_masked_avg_meter.avg:.4f}")
        print(f"Average Test SSIM: {ssim_avg_meter.avg:.4f}")
    if not args.mae:
        print(f"Average Inference Time (GPU): {time_avg_meter.avg:.4f} s/image")

    # Mask comparison logic (kept from original, ensure it's only run if applicable)
    if not args.mae and args.lm:
        try:
            mask_origin_path = os.path.join(args.model_path, "mask_origin.npy")
            mask_best_path = os.path.join(args.model_path, "mask_best.npy")
            if os.path.exists(mask_origin_path) and os.path.exists(mask_best_path):
                 mask_origin = np.load(mask_origin_path)
                 mask_best = np.load(mask_best_path)
                 mask_origin = einops.rearrange(
                     mask_origin,
                     "(cr1 cr2) c h w -> c (cr1 h) (cr2 w)",
                     cr1=cr1, cr2=cr2
                 )
                 mask_best = einops.rearrange(
                     mask_best,
                     "(cr1 cr2) c h w -> c (cr1 h) (cr2 w)",
                     cr1=cr1, cr2=cr2
                 )
                 compare_diff(mask_origin, mask_best, results_dir)
            else:
                 print("Warning: mask_origin.npy or mask_best.npy not found, skipping mask comparison.")
        except Exception as e:
             print(f"Error during mask comparison: {e}")


def compare_diff(mask_origin, mask_best, results_dir):
    transm_origin = mask_origin.sum() / mask_origin.size
    transm_best = mask_best.sum() / mask_best.size
    mask_diff = mask_best - mask_origin
    if mask_diff.shape[0] == 1: # Only save heatmap for single channel masks
         save_heatmap_with_signs(mask_diff[0], results_dir)
    else:
         print("Skipping heatmap generation for multi-channel mask difference.")
    print(f"Transmittance: [origin]: {transm_origin:.6f}, [best]: {transm_best:.6f}")

def save_heatmap_with_signs(array, results_dir, file_name="mask_diff_heatmap.png"):
    save_path=os.path.join(results_dir, file_name)
    colors = [ (0.7, 0.0, 0.0), (1.0, 0.7, 0.7), (1.0, 1.0, 1.0), (0.6, 0.8, 1.0), (0.0, 0.2, 0.8) ]
    cmap = LinearSegmentedColormap.from_list("RdWhBu", colors, N=256) # Changed to Blue for positive
    plt.figure(figsize=(10, 8))
    min_value = array.min()
    max_value = array.max()
    # Normalize array to [-1, 1] for consistent coloring if desired, otherwise use original range
    # Center the colormap around 0
    ax = sns.heatmap( array, cmap=cmap, center=0, vmin=min_value, vmax=max_value,
                      square=True, annot=False, cbar=True, cbar_kws={'label': 'Difference Value'} )
    ax.set_title("Mask Difference Heatmap (Red: Negative, Blue: Positive)", pad=20)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Heatmap saved to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description='Model Evaluation')
    # --- Arguments copied from original ---
    parser.add_argument('--gpu', type=str, default='0', help='gpu index') # Default to 0
    parser.add_argument('--data_path', type=str, required=True, help='path to test set')
    parser.add_argument('--model_path', type=str, required=True, help='trained model directory')
    parser.add_argument('--results_path', type=str, default='./test_results', help='results directory')
    parser.add_argument('--ckpt_name', type=str, default='model_best.pth', help='checkpoint filename')
    parser.add_argument('--image_size', type=int, nargs='+', default=[224, 224], help='image size (used for MAE dataset)') # Changed default
    parser.add_argument('--resize_size', type=int, default=-1, help='resize size before processing (BMNet)') # Simplified resize
    parser.add_argument("--cs_ratio", type=int, nargs=2, default=[4, 4], help="compression ratio (BMNet)") # Ensure exactly 2 values
    parser.add_argument('--lm', action='store_true', help='whether learnable mask was used (BMNet)')
    parser.add_argument('--defocus', action='store_true', help='whether defocus sci was used (BMNet)')
    parser.add_argument('--seed', type=int, default=717, help='random seed') # Changed default
    # --- MAE specific arguments ---
    parser.add_argument('--mae', action='store_true', help='Run MAE evaluation instead of BMNet')
    parser.add_argument('--model', type=str, default='base', choices=['tiny', 'base'], help='MAE model size (if --mae)')
    parser.add_argument('--imagenet', action='store_true', help='Use ImageNet dataset (requires --mae)')
    parser.add_argument('--blur', action='store_true', help='Apply Gaussian blur to input (MAE custom dataset)')
    parser.add_argument('--kernel_size', type=int, default=3, help='Gaussian kernel size (if --blur)')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian sigma (if --blur)')
    parser.add_argument('--eval_mask_ratio', type=float, default=0.75, help='Masking ratio for MAE evaluation (if --mae)')

    args = parser.parse_args()

    # --- Basic argument validation ---
    if args.imagenet and not args.mae:
        parser.error("--imagenet flag requires --mae flag.")
    if not args.imagenet and args.mae and not args.data_path:
         parser.error("--mae with custom dataset requires --data_path.")
    if not args.mae and not args.data_path:
         parser.error("BMNet evaluation requires --data_path.")

    # --- Set Seed ---
    seed_value = args.seed
    torch.manual_seed(seed_value)
    np.random.seed(seed_value) # Seed numpy as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # for multi-GPU

    # --- Set CUDNN flags ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Deterministic requires benchmark=False

    # --- Set GPU device ---
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # No need to set CUDA_VISIBLE_DEVICES if device is selected via torch.device(f'cuda:{args.gpu}')
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # --- Create results directory ---
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time) # Assumes time2file_name exists
    model_name_tag = f"{'mae' if args.mae else 'bmnet'}"
    if args.mae: model_name_tag += f"_{args.model}"
    dataset_tag = "imagenet" if args.imagenet else os.path.basename(args.data_path)
    results_dir = os.path.join(args.results_path, f"{dataset_tag}_{model_name_tag}_{date_time}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

    # --- Run evaluation ---
    eval(args, results_dir)

    # --- Commented out heatmap comparison ---
    # ... (original heatmap code can be uncommented if needed) ...