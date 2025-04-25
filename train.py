import os
# from re import M
import time
import shutil
import datetime
from argparse import ArgumentParser
# from xmlrpc.client import TRANSPORT_ERROR

import torch
import numpy as np
import einops
# from apex import amp
# from PIL import Image
from tqdm import tqdm
import tensorboardX
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from timm.scheduler import CosineLRScheduler

from data.dotadataset import MAEDatasetEval, make_dataset, MAEDataset
from utils import batch_PSNR, time2file_name, AverageMeter
from model import BMNet, mae_vit_base_patch16, mae_vit_tiny_mine


def main(args):
    global save_dir
    # load model and make model saving/log dir
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    save_dir = os.path.join(args.save_dir, date_time)
    if not os.path.exists(save_dir) and args.local_rank in [-1, 0]:
        os.makedirs(save_dir)
    # tensorboardX
    writer = tensorboardX.SummaryWriter(log_dir=save_dir)

    train_dir = args.data_path
    test_dir = train_dir.replace('train', 'valid')
    if args.mae:
        train_dataset = MAEDataset(train_dir, 
                                   args.blur, 
                                   args.kernel_size, 
                                   args.sigma)
        test_dataset = MAEDatasetEval(test_dir, 
                                      args.blur, 
                                      args.kernel_size, 
                                      args.sigma)
    else:
        train_dataset, test_dataset = make_dataset(train_dir, 
                                                   test_dir, 
                                                   cr=max(args.cs_ratio), 
                                                   defocus=args.defocus)

    if args.local_rank != -1:
        sampler = DistributedSampler(train_dataset, 
                                     num_replicas=args.world_size, 
                                     rank=args.local_rank, 
                                     shuffle=True)
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=args.batch_size, 
                                      sampler=sampler, 
                                      pin_memory=True, 
                                      num_workers=6)
    else:
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=args.batch_size, 
                                      shuffle=True, 
                                      drop_last=True,
                                      pin_memory=True,
                                      num_workers=6,
                                      persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1, 
                                 shuffle=False, 
                                 num_workers=2,
                                 pin_memory=True)

    # model defination
    if args.mae:
        model = mae_vit_base_patch16()
        # model = mae_vit_tiny_mine()
        model.cuda()
    else:
        model = BMNet(
            in_chans=args.n_channels,
            embed_dim=32,
            num_stage=args.num_stage,
            cs_ratio=args.cs_ratio,
            scaler=args.scaler,
            use_checkpoint=args.use_checkpoint
        ).cuda()

    # args.lm: learnable mask or not
    if args.mae == False:
        model.mask = nn.Parameter(mask.cuda(), requires_grad=args.lm)
        print(f"Learnable mask is {model.mask.requires_grad}")
        save_mask(model.mask.detach().cpu(), os.path.join(save_dir, "mask_origin.npy"))

    if args.finetune is not None:
        checkpoint_filename = args.finetune
        if os.path.isfile(checkpoint_filename):
            print("=> loading weights '{}'".format(checkpoint_filename))
            checkpoint = torch.load(checkpoint_filename, map_location='cpu')
            checkpoint['state_dict'] = {
                k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()
            }
            del checkpoint['mask']
            if 'mask' in checkpoint['state_dict']:
                del checkpoint['state_dict']['mask']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded weights '{}' (epoch {})".format(checkpoint_filename, checkpoint['epoch']))
        else:
            print("=> no model weights found at '{}'".format(checkpoint_filename))

    if args.mae:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=5e-5, #* args.batch_size / 256, 
                                      weight_decay=0.1)
    else:
        # loss and optimizer
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.local_rank in [-1, 0]:
        tot_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params (M): %.2f' % (tot_grad_params / 1.e6))

    # lr scheduler
    warmup_epochs = 10
    num_training_steps = args.end_epoch * len(train_dataloader)
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=num_training_steps,
        lr_min=1e-5,
        warmup_t=warmup_epochs * len(train_dataloader),
        warmup_lr_init=1e-6,
        warmup_prefix=True,
        cycle_decay=1,
        t_in_epochs=False
    )
    # scheduler = MultiStepLR(
    #     optimizer=optimizer,
    #     milestones=[125, 150, 175],
    #     gamma=0.5,
    #     verbose=True
    # )

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = DDP(model, delay_allreduce=True)

    if args.resume is not None:
        # Use a local scope to avoid dangling referencesa
        def resume():
            checkpoint_filename = args.resume
            if os.path.isfile(checkpoint_filename):
                print("=> loading checkpoint '{}'".format(checkpoint_filename))
                checkpoint = torch.load(checkpoint_filename, map_location='cpu')
                if not isinstance(model, DDP):
                    checkpoint['state_dict'] = {
                        k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()
                    }
                global best_psnr
                best_psnr = checkpoint['best_psnr']
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                # amp.load_state_dict(checkpoint['amp'])
                print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_filename, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(checkpoint_filename))

        resume()

    # Training
    iter = 0
    best_psnr = 0

    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        epoch_avg_loss = 0
        model.train()
        # breakpoint()
        if args.local_rank in [-1, 0]:
            lr = optimizer.param_groups[0]['lr']
            print("current learning rate: %e" % lr)
            writer.add_scalar('lr', lr, epoch_i)
            torch.cuda.synchronize()
            time_start = time.time()
        if args.local_rank != -1:
            sampler.set_epoch(epoch_i)
        
        training_bar = tqdm(train_dataloader, 
                            desc=f"[Epoch {epoch_i}/{end_epoch}]", 
                            colour='yellow',
                            ncols=125)
        for idx, (data, gt) in enumerate(training_bar):
            iter += 1

            optimizer.zero_grad()

            if args.mae:
                data = data.cuda()
                gt = gt.cuda()

                loss, out_train, _ = model(data, gt, unpatch_pred=True, mask_ratio=0.1)
            else:
                bs = data.shape[0]
                img_train = data.cuda()
                gt = gt.cuda()

                if args.resize_size > 0:
                    t = transforms.Resize(args.resize_size)
                    input_img = t(img_train)
                else:
                    input_img = img_train

                input_img = einops.rearrange(input_img, 
                                            "b c (cr1 h) (cr2 w) -> b (cr1 cr2) c h w", 
                                            cr1=cr1, cr2=cr2)

                if isinstance(model, DDP):
                    input_mask = model.module.mask.unsqueeze(0).expand(bs, -1, -1, -1, -1) \
                        * model.module.scaler
                else:
                    input_mask = model.mask.unsqueeze(0).expand(bs, -1, -1, -1, -1) * model.scaler

                meas = torch.sum(input_img * input_mask, dim=1, keepdim=True)

                out_train = model(meas, input_mask)

                if args.resize_size > 0:
                    t = transforms.Resize(args.image_size)
                    out_train = t(out_train)

                loss_base = criterion(out_train, gt)
                # -- mask loss --
                # beta = 1e3
                # loss_mask = beta * torch.relu(0.5 - model.mask.mean())
                loss_mask = 0
                loss = loss_base + loss_mask

            loss.backward()
            optimizer.step()

            # -- scheduler --
            if args.mae:
                scheduler.step_update(epoch_i * len(train_dataloader) + idx)

            # results
            epoch_avg_loss += loss.detach().cpu()

            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, gt, 1.)
            psnr_train = torch.as_tensor(float(psnr_train)).cuda()
            if args.local_rank != -1:
                loss = reduce_tensor(loss)
                psnr_train = reduce_tensor(psnr_train)
            if args.local_rank in [-1, 0]:
                writer.add_scalar('psnr', psnr_train.item(), iter)
                writer.add_scalar('recon_loss', loss.item(), iter)
                # print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} \
                # [epoch {epoch_i}][{idx}/{len_train_dataloader}] \
                # recon_loss: {loss.item():.4f} \
                # PSNR_train: {psnr_train.item():.2f} dB")

            training_bar.set_postfix({"[Stage]": f"[{idx + 1}/{len(train_dataloader)}]", 
                                     "Recon_loss": f"{loss.item():.4f}",
                                     "PSNR_trian": f"{psnr_train.item():.4f}"})

        epoch_avg_loss = epoch_avg_loss / (idx + 1)
        writer.add_scalar('epoch_avg_loss', epoch_avg_loss.item(), epoch_i + 1)
        print(f"epoch_avg_loss: {epoch_avg_loss}")
        
        if args.local_rank in [-1, 0]:
            torch.cuda.synchronize()
            time_end = time.time()
        if args.local_rank in [-1, 0]:
            print('time cost', time_end - time_start)

        if args.local_rank in [-1, 0] and (epoch_i + 1) % 10 == 0:
            # evaluation
            psnr_avg_meter = AverageMeter()
            model.eval()
            show_test = 0
            print('################# Validing ##################')
            for _, (data, gt) in enumerate(tqdm(test_dataloader, 
                                                ncols=125, 
                                                colour='blue')):
                if args.mae:
                    data = data.cuda()
                    gt = gt.cuda()

                    with torch.no_grad():
                        _, model_out, _ = model(data, gt, unpatch_pred=True, mask_ratio=0.1)
                        model_out = torch.clamp(model_out, 0., 1.)
                else:
                    bs = data.shape[0]
                    img_test = data.cuda()
                    gt = gt.cuda()

                    if args.resize_size > 0:
                        t = transforms.Resize(args.resize_size)
                        input_img = t(img_test)
                    else:
                        input_img = img_test

                    if show_test:
                        test = data[0, :]
                        writer.add_image(f'img_{show_test}', test)

                    input_img = einops.rearrange(input_img, 
                                                "b c (cr1 h) (cr2 w) -> b (cr1 cr2) c h w", 
                                                cr1=cr1, cr2=cr2)
                    if isinstance(model, DDP):
                        input_mask = model.module.mask.unsqueeze(0).expand(bs, -1, -1, -1, -1) \
                            * model.module.scaler
                    else:
                        input_mask = model.mask.unsqueeze(0).expand(bs, -1, -1, -1, -1) * model.scaler
                    meas = torch.sum(input_img * input_mask, dim=1, keepdim=True)

                    if show_test:
                        test = meas[0, :]
                        writer.add_images(f'measurement_{show_test}', test)

                    with torch.no_grad():
                        out_test = model(meas, input_mask)
                        if args.resize_size:
                            t = transforms.Resize(args.image_size)
                            out_test = t(out_test)

                    model_out = torch.clamp(out_test, 0, 1)

                    if show_test:
                        test = model_out[0, :]
                        writer.add_image(f'reconstruction_{show_test}', test, epoch_i)
                        show_test = show_test - 1

                psnr_test = batch_PSNR(model_out, gt, 1.)
                psnr_avg_meter.update(psnr_test)

            writer.add_scalar("avg", psnr_avg_meter.avg.item(), epoch_i)
            print("test psnr: %.4f" % psnr_avg_meter.avg.item())

            is_best = psnr_avg_meter.avg > best_psnr
            if args.local_rank in [-1, 0] and is_best:
                best_psnr = psnr_avg_meter.avg
                if args.mae:
                    save_checkpoint(
                    {
                        # 'epoch': epoch_i,
                        'state_dict': model.state_dict(),
                        # 'best_psnr': best_psnr,
                        # 'optimizer': optimizer.state_dict(),
                        # 'amp': amp.state_dict(),
                        # 'mask': model.module.mask if isinstance(model, DDP) else model.mask
                    }, 
                    is_best, 
                    filename=os.path.join(save_dir, f'model_{epoch_i + 1}_psnr{best_psnr:.4f}.pth')
                    )
                else:
                    save_checkpoint(
                        {
                            'epoch': epoch_i,
                            'state_dict': model.state_dict(),
                            'best_psnr': best_psnr,
                            'optimizer': optimizer.state_dict(),
                            # 'amp': amp.state_dict(),
                            'mask': model.module.mask if isinstance(model, DDP) else model.mask
                        }, 
                        is_best, 
                        filename=os.path.join(save_dir, f'model_{epoch_i + 1}_psnr{best_psnr:.4f}.pth')
                    )
                print('best test psnr till now %.4f' % best_psnr)
                print('checkpoint with %d iterations has been saved. \n' % epoch_i)

                if args.lm:
                    mask_save_path = os.path.join(save_dir, f"mask_{epoch_i}.npy")
                    best_mask_path = os.path.join(save_dir, "mask_best.npy")
                    save_mask(model.mask.detach().cpu(), mask_save_path)
                    save_mask(model.mask.detach().cpu(), best_mask_path)
                    print(f'learnable mask (shape {model.mask.shape}) has been saved. \n')

            psnr_avg_meter.reset()

        # scheduler.step()

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, f'model_best.pth'))

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size

    return rt

def save_mask(mask_tensor, save_path):
    # mask_array = mask_tensor.numpy()
    np.save(save_path, mask_tensor.numpy())


if __name__ == "__main__":
    seed_value = 3407
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    parser = ArgumentParser(description='BMI')
    parser.add_argument('--warmup_steps', type=int, default=5, 
                        help='epoch number of learnig rate warmup')
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
    parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512], help='image size')
    parser.add_argument('--resize_size', type=int, nargs='+', default=-1, 
                        help='resize the image to fit the compression ratio')
    parser.add_argument('--n_channels', type=int, default=1, 
                        help='1 for gray image, currently only support 1')
    parser.add_argument("--cs_ratio", type=int, nargs='+', default=[4, 4], help="compression ratio")
    parser.add_argument('--num_stage', type=int, default=10, help='satge number of the DUN')
    parser.add_argument('--scaler', action='store_true', help='whether add scaler to mask')
    parser.add_argument('--lm', action='store_true', help='whether learnable mask')
    parser.add_argument('--defocus', action='store_true', help='whether do dedocus sci')

    parser.add_argument('--use_checkpoint', action='store_true', 
                        help='whether use torch checkpoint to save memory')

    parser.add_argument('--save_interval', type=int, default=5, 
                        help='save model on test set every x epochs')
    parser.add_argument("--test_interval", type=int, default=1, 
                        help='evaluate model on test set every x epochs')
    parser.add_argument("--data_path", type=str, 
                        default="/data2/wangzhibin/DOTA/trainsplit512_nogap/images/", 
                        help='path to dota dataset')
    parser.add_argument('--save_dir', type=str, default='./model_ckpt', help='output dir')
    parser.add_argument('--finetune', type=str, default=None, help='resume path')
    parser.add_argument('--resume', type=str, default=None, help='resume path')

    parser.add_argument('--opt-level', type=str, default='O0', help='use fp32 or fp16')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)

    ## -- for mae exp --
    parser.add_argument('--mae', action='store_true', help='whether do mae exp')
    parser.add_argument('--blur', action='store_true', help='whether do blur images')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=1.0)
    args = parser.parse_args()

    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    batch_size = args.batch_size
    best_psnr = 0.

    args.local_rank = int(os.getenv('LOCAL_RANK', -1))  
    # https://pytorch.org/docs/stable/elastic/run.html
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # scale learning_rate according to total_batch_size
    # total_batch_size = args.world_size * args.batch_size
    # args.learning_rate = args.learning_rate * float(total_batch_size)
    # args.init_learning_rate = args.learning_rate / 5.
    # args.min_learning_rate = args.learning_rate / 100.

    cr1, cr2 = args.cs_ratio

    g = torch.Generator()
    g.manual_seed(args.seed)

    # mask = torch.bernoulli(torch.empty(
    #     cr1 * cr2, 
    #     1, 
    #     args.image_size[0] // cr1, 
    #     args.image_size[1] // cr2
    #     ).uniform_(0, 1), generator=g
    # )

    # 根据seed随机初始化0-1连续值掩码
    mask = torch.rand(
        cr1 * cr2, 
        1, 
        args.image_size[0] // cr1, 
        args.image_size[1] // cr2,
        generator=g
    )

    if args.resize_size > 0:
        mask = torch.bernoulli(torch.empty(
            cr1 * cr2, 
            1, 
            args.resize_size[0] // cr1, 
            args.resize_size[1] // cr2
            ).uniform_(0, 1), generator=g
        )

    main(args)
