import os
from re import I
import time
import datetime

import cv2
import einops
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

from utils import PSNR, SSIM, time2file_name, AverageMeter
from data.dotadataset import DATADataset
from network.BMNet import BMNet


def eval(args, results_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    time_avg_meter = AverageMeter()
    psnr_avg_meter = AverageMeter()
    ssim_avg_meter = AverageMeter()
    # show_test = args.num_show

    model = BMNet(in_chans=1, 
                  num_stage=10, 
                  embed_dim=32, 
                  cs_ratio=args.cs_ratio).to(device)
    ckpt_path = os.path.join(args.model_path, "model_best.pth")
    ckpt = torch.load(ckpt_path)

    mask = ckpt['mask'].to(device)
    model_ckpt = ckpt['state_dict']
    model_ckpt = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
    if 'mask' in model_ckpt: 
        del model_ckpt['mask']
    model.load_state_dict(model_ckpt, strict=False)
    model.to(device)
    model.eval()

    test_dataset = DATADataset(args.data_path, 
                               cr=max(args.cs_ratio),
                               defocus=args.defocus)
    print(f"Defocus data is {args.defocus}")
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1, 
                                 shuffle=False, 
                                 num_workers=4,
                                 pin_memory=True)

    print('################# Testing ##################')
    for idx, (data, gt) in enumerate(tqdm(test_dataloader, ncols=125, colour='green')):
        # break
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

        psnr_test = PSNR(out_test, gt, 1.)
        psnr_avg_meter.update(psnr_test)
        ssim_test = SSIM(out_test, gt, 1.)
        ssim_avg_meter.update(ssim_test)

        # if show_test:
        #     img_ycrcb = (img_ycrcb[0].cpu().data.numpy() * 255.).astype(np.uint8).transpose(1, 2, 0)

        #     show1 = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

        #     show2 = out_test[0, :]
        #     y = (show2.squeeze(0).cpu().data.numpy() * 255.).astype(np.uint8)
        #     show2 = img_ycrcb.copy()
        #     show2[:, :, 0] = y
        #     show2 = cv2.cvtColor(show2, cv2.COLOR_YCrCb2BGR)

        #     cv2.imwrite(results_dir + f'/orig_{name}.jpg', show1)
        #     cv2.imwrite(results_dir + f'/recon_{name}.jpg', show2)

        #     show_test = show_test - 1

    print("test psnr: %.4f" % psnr_avg_meter.avg)
    print("test ssim: %.4f" % ssim_avg_meter.avg)
    print("avg throughput: %.4f s/image" % time_avg_meter.avg)

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
        compare_mask(mask_origin, mask_best)
    

def compare_mask(mask_origin, mask_best):
    transm_origin = mask_origin.sum() / mask_origin.size
    transm_best = mask_best.sum() / mask_best.size

    mask_diff = mask_best - mask_origin
    # save_heatmap_with_signs(np.where(mask_diff[0] > 0, mask_diff[0], 0))
    save_heatmap_with_signs(mask_diff[0])

    print(f"Transmittance: [origin]: {transm_origin:.6f}, [best]: {transm_best:.6f}")

def save_heatmap_with_signs(array):
    """
    生成并保存带正负值区分的热力图
    :param array: 输入的ndarray，值范围[-1, 1]
    """
    save_path=os.path.join(args.results_path, "mask_diff_heatmap.png")

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
    # array = np.where(array < 0, array, 0)
    
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

    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512], help='image size')
    parser.add_argument('--resize_size', type=int, nargs='+', default=-1, help='image size')
    parser.add_argument("--cs_ratio", type=int, nargs='+', default=[4, 4], help="compression ratio")

    parser.add_argument("--num_show", type=int, default=1, help="number of images to show")
    parser.add_argument('--lm', action='store_true', help='whether learnable mask')
    parser.add_argument('--defocus', action='store_true', help='whether do dedocus sci')

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()

    # load model and make model saving/log dir
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    results_dir = os.path.join(args.results_path, date_time)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    cr1, cr2 = args.cs_ratio

    eval(args, results_dir)
