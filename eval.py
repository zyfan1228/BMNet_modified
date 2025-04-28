from ast import Gt
import os
from re import I
import time
import datetime

import cv2
import einops
from networkx import out_degree_centrality
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from argparse import ArgumentParser

from train_mae_single import calculate_masked_psnr_original_scale
from utils import PSNR, SSIM, time2file_name, AverageMeter
from data.dotadataset import DATADataset, MAEDatasetEval
from model import BMNet, mae_vit_base_patch16, mae_vit_tiny_mine


# for sim_v2_gt
mean_gt = 0.330694
std_gt = 0.176989


def batch_PSNR(img, imclean, data_range):
    """
    Calculates the average PSNR across a batch of images using PyTorch.
    IMPORTANT: Ensure 'img' tensor is clamped to [0, data_range] BEFORE calling this function.
    """
    # 确保输入是 Tensor 且形状匹配
    if not isinstance(img, torch.Tensor) or not isinstance(imclean, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensor")
    if img.shape != imclean.shape:
        raise ValueError(f"Input shapes must match: img {img.shape}, imclean {imclean.shape}")

    # 计算每个像素的平方误差
    squared_error = (img - imclean) ** 2

    # 计算每张图片的 MSE (在 C, H, W 或 H, W 维度上求平均)
    if img.ndim == 4: # NCHW
        mse_per_image = torch.mean(squared_error, dim=(1, 2, 3))
    elif img.ndim == 3: # NHW
        mse_per_image = torch.mean(squared_error, dim=(1, 2))
    else:
        raise ValueError(f"Unsupported input dimensions: {img.ndim}. Expected 3 or 4.")

    # 避免 log10(0) - 添加一个小的 epsilon
    # 对于 MSE 接近 0 的情况，PSNR 会非常高
    epsilon = 1e-10
    psnr_per_image = 10.0 * torch.log10((data_range**2) / (mse_per_image + epsilon))

    # 计算批次的平均 PSNR
    average_psnr = torch.mean(psnr_per_image)

    return average_psnr

def eval(args, results_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cr1, cr2 = args.cs_ratio

    time_avg_meter = AverageMeter()
    psnr_avg_meter = AverageMeter()
    ssim_avg_meter = AverageMeter()
    psnr_masked_avg_meter = AverageMeter()
    # show_test = args.num_show

    if args.mae:
        # model = mae_vit_base_patch16().to(device)
        model = mae_vit_tiny_mine().to(device)
        test_dataset = MAEDatasetEval(args.data_path, 
                                      args.blur, 
                                      args.kernel_size, 
                                      args.sigma)

    else:
        model = BMNet(in_chans=1, 
                  num_stage=10, 
                  embed_dim=32, 
                  cs_ratio=args.cs_ratio).to(device)
        
        test_dataset = DATADataset(args.data_path, 
                               cr=max(args.cs_ratio),
                               defocus=args.defocus)
        print(f"Defocus data is {args.defocus}")

    ckpt_path = os.path.join(args.model_path, args.ckpt_name)
    ckpt = torch.load(ckpt_path)

    if args.mae == False:
        mask = ckpt['mask'].to(device)
    model_ckpt = ckpt['state_dict']
    model_ckpt = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
    if 'mask' in model_ckpt: 
        del model_ckpt['mask']
    model.load_state_dict(model_ckpt, strict=False)
    model.to(device)
    model.eval()
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1, 
                                 shuffle=False, 
                                 num_workers=4,
                                 pin_memory=True)

    print('################# Testing ##################')
    mean_train = torch.tensor([mean_gt]).view(1, 1, 1, 1).to(device)
    std_train = torch.tensor([std_gt]).view(1, 1, 1, 1).to(device)
    save_idxs = [7, 14, 21, 28, 35, 42, 49, 56, 63]
    for idx, (data, gt) in enumerate(tqdm(test_dataloader, ncols=125, colour='green')):
        # if idx not in save_idxs:
        #     continue

        if args.mae:
            data = data.to(device)
            gt = gt.to(device)

            with torch.no_grad():
                _, out_test, mask = model(data, gt, mask_ratio=0.75)
                out_test = model.unpatchify(out_test, in_chan=data.shape[1])
                out_test = (out_test * std_train) + mean_train
                gt = (gt * std_train) + mean_train
                data = (data * std_train) + mean_train
                # breakpoint()
        else:
            bs = data.shape[0]
            img_test = data.to(device)
            gt = gt.to(device)

            if args.resize_size > 0:
                t = transforms.Resize(args.resize_size)
                input_img = t(img_test)
            else:
                input_img = img_test

            input_mask = mask.unsqueeze(0).expand(bs, -1, -1, -1, -1).to(device) * model.scaler

            input_img = einops.rearrange(input_img, 
                                        "b c (cr1 h) (cr2 w) -> b (cr1 cr2) c h w", 
                                        cr1=cr1, cr2=cr2)

            meas = torch.sum(input_img * input_mask, dim=1, keepdim=True)

            with torch.no_grad():
                torch.cuda.synchronize()
                st = time.time()
                out_test = model(meas, input_mask)
                torch.cuda.synchronize()
                ed = time.time()
                time_avg_meter.update(ed - st)

                if args.resize_size > 0:
                    t = transforms.Resize(args.image_size)
                    out_test = t(out_test)

        out_test = torch.clamp(out_test, 0, 1)

        psnr_test_full = batch_PSNR(out_test, gt, 1.)
        psnr_avg_meter.update(psnr_test_full)
        ssim_test = SSIM(out_test, gt, 1.)
        ssim_avg_meter.update(ssim_test)

        psnr_val_masked = calculate_masked_psnr_original_scale(
            out_test, 
            gt, 
            mask, # Pass the validation mask
            model.patchify, 
            data.shape[1], 
            1.0
        )
        psnr_masked_avg_meter.update(psnr_val_masked)

        if idx in save_idxs:
            save_array(out_test.cpu(), file_path=os.path.join(results_dir, f"output_idx_{idx}.png"))
            save_array(gt.cpu(), file_path=os.path.join(results_dir, f"gt_idx_{idx}.png"))
            save_array(data.cpu(), file_path=os.path.join(results_dir, f"input_idx_{idx}.png"))

    print("test psnr: %.4f" % psnr_avg_meter.avg)
    print("masked psnr: %.4f" % psnr_masked_avg_meter.avg)
    print("test ssim: %.4f" % ssim_avg_meter.avg)
    # print("avg throughput: %.4f s/image" % time_avg_meter.avg)

    if args.lm: 
        mask_origin = np.load(os.path.join(args.model_path, "mask_origin.npy"))
        mask_best = np.load(os.path.join(args.model_path, "mask_best.npy"))
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
    

def compare_diff(mask_origin, mask_best, results_dir):
    transm_origin = mask_origin.sum() / mask_origin.size
    transm_best = mask_best.sum() / mask_best.size

    mask_diff = mask_best - mask_origin
    save_heatmap_with_signs(mask_diff[0], results_dir)
    # save_heatmap_with_signs(mask_origin[0], file_name="mask_origin_heatmap.png")

    print(f"Transmittance: [origin]: {transm_origin:.6f}, [best]: {transm_best:.6f}")

def save_array(data, file_path):
    if isinstance(data, torch.Tensor):
        data_array = data.numpy()
    else:
        data_array = data
    data_array = data_array.squeeze()
    Image.fromarray((data_array * 255).astype(np.uint8)).save(file_path)

def save_heatmap_with_signs(array, results_dir, file_name="mask_diff_heatmap.png"):
    """
    生成并保存带正负值区分的热力图
    :param array: 输入的ndarray，值范围[-1, 1]
    """
    save_path=os.path.join(results_dir, file_name)

    # 创建自定义颜色映射（红-白-绿）
    colors = [
        (0.7, 0.0, 0.0),   # 深红（负值最大，RGB: 70% 红）
        (1.0, 0.7, 0.7),   # 亮红（负值中等，RGB: 100% 红 + 70% 亮化）
        (1.0, 1.0, 1.0),   # 纯白（零值，RGB: 100% 白）
        (0.6, 0.8, 1.0),   # 浅蓝（正值中等，RGB: 60% 红 + 80% 绿 + 100% 蓝）
        (0.0, 0.2, 0.8)    # 深蓝（正值最大，RGB: 20% 绿 + 80% 蓝）
    ]
    cmap = LinearSegmentedColormap.from_list("RdWhGn", colors, N=256)

    # 设置图形大小
    plt.figure(figsize=(10, 8))

    # min-max normalization to [-1, 1]
    min_value = array.min()
    max_value = array.max()
    array = 2 * (array - min_value) / (max_value - min_value) - 1
    # array = np.where(array > 0, array, 0)
    
    # 绘制热力图
    ax = sns.heatmap(
        array,
        cmap=cmap,
        center=0,          # 以0为中心点
        vmin=min_value,           # 最小值-1
        vmax=max_value,            # 最大值1
        square=True,       # 单元格为方形
        annot=False,       # 不显示数值
        cbar=True,         # 显示颜色条
        cbar_kws={'label': 'Value'}
    )
    
    # 添加标题
    ax.set_title("Mask Difference Heatmap (Red: Negative, Green: Positive)", pad=20)
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Heatmap saved to {save_path}")


if __name__ == "__main__":
    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = ArgumentParser(description='BMI')
    parser.add_argument('--gpu', type=str, default='3', help='gpu index')
    parser.add_argument('--data_path', 
                        type=str, 
                        default="./samples", 
                        help='path to test set')
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="/data2/wangzhibin/_trainning_ckpt/_RSSCI_v2.0/baseline-16", 
        help='trained or pre-trained model directory'
    )
    parser.add_argument('--results_path', 
                        type=str, 
                        default='./test_results', 
                        help='results for reconstructed images')
    parser.add_argument('--ckpt_name', 
                        type=str, 
                        default='model_best.pth')

    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512], help='image size')
    parser.add_argument('--resize_size', type=int, nargs='+', default=-1, help='image size')
    parser.add_argument("--cs_ratio", type=int, nargs='+', default=[4, 4], help="compression ratio")

    parser.add_argument("--num_show", type=int, default=1, help="number of images to show")
    parser.add_argument('--lm', action='store_true', help='whether learnable mask')
    parser.add_argument('--defocus', action='store_true', help='whether do dedocus sci')

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    ## -- for mae exp --
    parser.add_argument('--mae', action='store_true', help='whether do mae exp')
    parser.add_argument('--blur', action='store_true', help='whether do blur images')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=1.0)

    args = parser.parse_args()

    # -- load model and make model saving/log dir --
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    results_dir = os.path.join(args.results_path, date_time)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    eval(args, results_dir)


    # -- compare outputs' mse heatmap --
    # output_path = './test_results/2025_04_21_13_42_29/output_idx_380.png'
    # gt_path = './test_results/2025_04_21_13_42_29/gt_idx_380.png'

    # output = np.array(Image.open(output_path))
    # gt = np.array(Image.open(gt_path))

    # # breakpoint()
    # mse_diff = (gt - output) ** 2
    # mse_diff_normalized = (mse_diff - mse_diff.min()) / (mse_diff.max() - mse_diff.min())

    # plt.figure(figsize=(8, 6))
    # heatmap = plt.imshow(
    #     mse_diff_normalized, 
    #     cmap='turbo',  
    #     vmin=0, 
    #     vmax=np.max(mse_diff_normalized)
    # )
    # plt.colorbar(heatmap, label='MSE Value')  # 添加颜色条
    # plt.title("Element-wise MSE Heatmap (Single Channel)")
    # plt.axis('off')  # 可选：关闭坐标轴

    # output_dir = "./"
    # os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    # output_path = os.path.join(output_dir, "mse_diff_heatmap.png")
    # plt.savefig(output_path, bbox_inches='tight', dpi=300)  # 保存为PNG，高分辨率
    # print(f"热力图已保存至: {output_path}")