import os
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
from argparse import ArgumentParser

from utils import PSNR, SSIM, time2file_name, AverageMeter
from data.dotadataset import DATADataset
from network.BMNet import BMNet



def eval(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    time_avg_meter = AverageMeter()
    psnr_avg_meter = AverageMeter()
    ssim_avg_meter = AverageMeter()
    show_test = args.num_show

    model = BMNet(in_chans=1, num_stage=10, embed_dim=32, cs_ratio=args.cs_ratio).to(device)
    ckpt = torch.load(args.model_path, map_location='cpu')

    mask = ckpt['mask'].to(device)
    model_ckpt = ckpt['state_dict']
    model_ckpt = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
    if 'mask' in model_ckpt: del model_ckpt['mask']
    model.load_state_dict(model_ckpt, strict=False)
    model.to(device)
    model.eval()

    test_dataset = DATADataset(args.data_path, cr=max(args.cs_ratio))
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1, 
                                 shuffle=False, 
                                 num_workers=2,
                                 pin_memory=True)

    iter = 0
    print('#################Testing##################')
    for idx, (data, gt) in enumerate(tqdm(test_dataloader)):
        bs = data.shape[0]
        img_test = data.to(device)

        if args.resize_size > 0:
            t = transforms.Resize(args.resize_size)
            input_img = t(img_test)
        else:
            input_img = img_test

        input_mask = mask.unsqueeze(0).expand(bs, -1, -1, -1, -1).to(device) * model.scaler

        input_img = einops.rearrange(input_img, "b c (cr1 h) (cr2 w) -> b (cr1 cr2) c h w", cr1=cr1, cr2=cr2)

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

        psnr_test = PSNR(out_test, img_y, 1.)
        psnr_avg_meter.update(psnr_test)
        ssim_test = SSIM(out_test, img_y, 1.)
        ssim_avg_meter.update(ssim_test)

        if show_test:
            img_ycrcb = (img_ycrcb[0].cpu().data.numpy() * 255.).astype(np.uint8).transpose(1, 2, 0)

            show1 = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

            show2 = out_test[0, :]
            y = (show2.squeeze(0).cpu().data.numpy() * 255.).astype(np.uint8)
            show2 = img_ycrcb.copy()
            show2[:, :, 0] = y
            show2 = cv2.cvtColor(show2, cv2.COLOR_YCrCb2BGR)

            cv2.imwrite(results_dir + f'/orig_{name}.jpg', show1)
            cv2.imwrite(results_dir + f'/recon_{name}.jpg', show2)

            show_test = show_test - 1

    print("test psnr: %.4f" % psnr_avg_meter.avg)
    print("test ssim: %.4f" % ssim_avg_meter.avg)
    print("avg throughput: %.4f s/image" % time_avg_meter.avg)


if __name__ == "__main__":
    parser = ArgumentParser(description='BMI')

    parser.add_argument('--gpu', type=str, default='3', help='gpu index')
    parser.add_argument('--data_path', 
                        type=str, 
                        default="./samples", 
                        help='path to test set')
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="/data2/wangzhibin/_trainning_ckpt/_RSSCI_v2.0/baseline-16/model_best.pth", help='trained or pre-trained model directory'
    )
    parser.add_argument('--results_path', 
                        type=str, 
                        default='./results', 
                        help='results for reconstructed images')

    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512], help='image size')
    parser.add_argument('--resize_size', type=int, nargs='+', default=None, help='image size')
    parser.add_argument("--cs_ratio", type=int, nargs='+', default=[4, 4], help="compression ratio")

    parser.add_argument("--num_show", type=int, default=1, help="number of images to show")

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load model and make model saving/log dir
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    results_dir = args.results_path + '/' + date_time

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    cr1, cr2 = args.cs_ratio

    eval(args)
