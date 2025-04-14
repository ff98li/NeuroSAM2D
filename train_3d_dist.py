# train.py
#!/usr/bin/env	python3

""" train network using pytorch, distributed training
    Original author: Yunli Qi
    Modified by: @ff98li
"""

import os
import time

import torch
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from torch.cuda import device_count
from func_3d.optimizer import Optimizer
from fvcore.common.param_scheduler import CosineParamScheduler, ConstantParamScheduler
from my_util_script.visualize import show_mask
from copy import deepcopy
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
import datetime
import wandb

def plot_loss_curve(loss_list, prompt_loss_list, non_prompt_loss_list, tol_list, eiou_list, edice_list, args):
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    colors = {
        'total_train': '#27ae60',      # green
        'validation': '#e84393',       # pink
        'iou': '#2980b9',              # blue for metrics
        'dice': '#f39c12'              # orange for metrics
    }

    ax1.plot(loss_list, 'o-', color=colors['total_train'], label='Total training Loss', linewidth=2, markersize=5)
    ax1.plot(tol_list, 'o-', color=colors['validation'], label='Total validation Loss', linewidth=2, markersize=5)

    ax1.set_title('Loss', pad=20, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(bottom=0)  # Start y-axis from 0

    ax2.plot(eiou_list, 'o-', color=colors['iou'], label='IOU', linewidth=2, markersize=5)
    ax2.plot(edice_list, 'o-', color=colors['dice'], label='DICE', linewidth=2, markersize=5)

    ax2.set_title('Validation Metrics', pad=20, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(args.path_helper['ckpt_path'], 'metrics.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()


def sanity_check(args, rng):
    
    video_length = args.video_length * args.batch_size
    prompt_freq = args.prompt_freq

    temp_args = deepcopy(args)
    temp_args.distributed = False
    nice_train_loader, nice_test_loader = get_dataloader(temp_args)
    for pack in nice_train_loader:
        mask_dict = pack['label']
        prompt_frame_id = list(range(0, video_length, prompt_freq))
        obj_list = []
        for id in prompt_frame_id:
            obj_list += list(mask_dict[id].keys())
        obj_list = list(set(obj_list))
        print(f"obj_list: {obj_list}")
        if len(obj_list) == 0:
            continue ## skip example without any object annotation
        else:
            break

    imgs_tensor = pack['image']
    prompt = rng.choice(['click', 'bbox'])
    if prompt == 'click':
        pt_dict = pack['pt']
        point_labels_dict = pack['p_label']
    elif prompt == 'bbox':
        bbox_dict = pack['bbox']
    imgs_tensor = imgs_tensor.squeeze(0)
    imgs = imgs_tensor.detach().cpu().numpy()
    imgs = (
        (imgs - np.min(imgs))
        / (np.max(imgs) - np.min(imgs))
        * 255.0
    )
    imgs[imgs == 0] = 0
    imgs = imgs.astype(np.uint8)
    name = pack['image_meta_dict']['filename_or_obj']
    
    case_id = ""
    if len(name) > 1:
        case_ids = []
        for name_i in name:
            case_ids.append(os.path.basename(name_i[0]).split(".nii")[0])
        case_id = "_".join(case_ids)
    else:
        case_id = os.path.basename(name[0]).split(".nii")[0]

    print(f"Running sanity check on {case_id}")

    for id in range(video_length):
        for ann_obj_id in obj_list:
            if type(ann_obj_id) != int:
                ann_obj_id = int(ann_obj_id)
            try:
                mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32)
            except KeyError as e:
                continue

            os.makedirs(f'./temp/sancheck/{case_id}', exist_ok=True)
            fig, ax = plt.subplots(1, 2)
            img_show = np.transpose(imgs[id, :, :, :], (1, 2, 0))
            mask_show = mask[0, 0, :, :].numpy().astype(np.uint8)
            overlay_show = show_mask(mask_show, img_show)

            ax[0].imshow(img_show)
            ax[0].axis('off')
            if prompt == 'bbox':
                try:
                    bboxes = bbox_dict[id][ann_obj_id].clone()
                    bboxes = bboxes.reshape(-1, 4)
                    for bbox in bboxes:
                        ax[0].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                except KeyError:
                    pass
            elif prompt == 'click':
                try:
                    pts = pt_dict[id][ann_obj_id].clone()
                    pts = pts.reshape(-1, 2).cpu().numpy()
                    for pt in pts:
                        ax[0].scatter(pt[0], pt[1], s=10, c='red')
                except KeyError:
                    pass
            #ax[1].imshow(mask[0, 0, :, :].numpy(), cmap='gray')
            ax[1].imshow(overlay_show)
            ax[1].axis('off')
            plt.savefig(f'./temp/sancheck/{case_id}/{id}_{ann_obj_id}.png', bbox_inches='tight', pad_inches=0)
            plt.close()



def main(args, rng):

    should_log = (not args.distributed) or (args.distributed and args.global_rank == 0)
    os.environ['WANDB_MODE'] = args.wandb_mode if hasattr(args, 'wandb_mode') else 'offline'
    
    if should_log:
        wandb.init(
            project="NeuroSAM2D",
            name=args.exp_name,
            config=vars(args),
            dir=args.path_helper['log_path'] if hasattr(args, 'path_helper') else None,
            settings=wandb.Settings(start_method="thread")
        )

    dist.init_process_group(
        backend="nccl",
        timeout = datetime.timedelta(seconds=1800)
    )
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    print(f"[Rank {args.global_rank}] Ready to train on {os.environ['WORLD_SIZE']} GPUs.")
    print(f"[Rank {args.global_rank}] GPU device is {args.gpu_device}")
    GPUdevice = torch.device(type = 'cuda', index = args.gpu_device)
    torch.cuda.set_device('cuda:'+str(args.local_rank))

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    net.to(dtype=torch.bfloat16)
    if args.pretrain:
        print(f"[Rank {args.global_rank}]", args.pretrain)
        weights = torch.load(args.pretrain, weights_only=False, map_location="cpu")['model']
        state_dict_info = net.load_state_dict(weights,strict=False)
        print(f"[Rank {args.local_rank}] {state_dict_info}.")
        dist.barrier()
    
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    args.distributed = dist.is_available() and dist.is_initialized()
    if args.distributed:  # implicitly it's clear that we use cuda in this case
            print(f"I am global rank {args.global_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {GPUdevice}")

    vit_layers = []
    if args.finetune_backbone:
        vit_layers += list(net.module.image_encoder.trunk.parameters())
    if args.finetune_neck:
        vit_layers += list(net.module.image_encoder.neck.parameters())
    sam_layers = (
                  []
                #   + list(net.image_encoder.parameters())
                #   + list(net.sam_prompt_encoder.parameters())
                #  + list(net.module.image_encoder.parameters())
                  # + list(net.module.image_encoder.neck.parameters())
                  + list(net.module.sam_prompt_encoder.parameters())
                  + list(net.module.sam_mask_decoder.parameters())
                  )
    mem_layers = (
                    []
                    + list(net.module.obj_ptr_proj.parameters())
                    + list(net.module.memory_encoder.parameters())
                    + list(net.module.memory_attention.parameters())
                    + list(net.module.mask_downsample.parameters())
                    )
    if len(sam_layers) == 0:
        optimizer1 = None
    else:
        optimizer1 = Optimizer(
            optimizer = optim.AdamW(sam_layers, lr=args.lr_sam, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False),
            schedulers = [
                {
                    "lr": CosineParamScheduler(start_value=args.lr_sam, end_value=args.lr_sam * 0.01),
                    "weight_decay": ConstantParamScheduler(0.1)
                }
            ]
        )

    if len(mem_layers) == 0:
        optimizer2 = None
    else:
        optimizer2 = Optimizer(
            optimizer = optim.AdamW(mem_layers, lr=args.lr_mem, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False),
            schedulers = [
                {
                    "lr": CosineParamScheduler(start_value=args.lr_mem, end_value=args.lr_mem * 0.01),
                    "weight_decay": ConstantParamScheduler(0.1)
                }
            ]
        )
    if len(vit_layers) == 0:
        optimizer3 = None
    else:
        optimizer3 = Optimizer(
            optimizer = optim.AdamW(vit_layers, lr=args.lr_vit, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False),
            schedulers = [
                {
                    "lr": CosineParamScheduler(start_value=args.lr_vit, end_value=args.lr_vit * 0.01),
                    "weight_decay": ConstantParamScheduler(0.1)
                }
            ]
        )
        
    if should_log:
        wandb.watch(net, log="all", log_freq=1)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.global_rank == 0:
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)

    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if args.global_rank == 0:
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, args.net, settings.TIME_NOW))

        #create checkpoint folder to save model
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f"checkpoint_path: {checkpoint_path}")
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0
    start_epoch = 0
    loss_list = []
    prompt_loss_list = []
    non_prompt_loss_list = []
    eiou_list = []
    edice_list = []
    tol_list = []

    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False, map_location="cpu")
        state_dict_info = net.module.load_state_dict(checkpoint['model'])
        print(f"[Rank {args.local_rank}] Model loaded with {state_dict_info}.")
        #optimizer1.load_state_dict(checkpoint['optimizer1'])
        #optimizer2.load_state_dict(checkpoint['optimizer2'])

        optimizer1.optimizer.load_state_dict(checkpoint['optimizer1'])
        optimizer2.optimizer.load_state_dict(checkpoint['optimizer2'])
        optimizer3.optimizer.load_state_dict(checkpoint['optimizer3'])
        best_dice = checkpoint['best_dice']
        start_epoch = checkpoint['epoch'] + 1
        dist.barrier()

        #loss_list = checkpoint['loss_list']
        #prompt_loss_list = checkpoint['prompt_loss_list']
        #non_prompt_loss_list = checkpoint['non_prompt_loss_list']

    for epoch in range(start_epoch, settings.EPOCH):

        net.train()
        time_start = time.time()
        loss, prompt_loss, non_prompt_loss = function.train_sam(args, net, optimizer1, optimizer2, optimizer3, nice_train_loader, epoch, rng)
        if args.distributed:
            losses, prompt_losses, non_prompt_losses = [[None for _ in range(dist.get_world_size())] for _ in range(3)]
            dist.all_gather_object(losses, loss)
            dist.all_gather_object(prompt_losses, prompt_loss)
            dist.all_gather_object(non_prompt_losses, non_prompt_loss)
            loss, prompt_loss, non_prompt_loss = np.vstack(losses).mean(), np.vstack(prompt_losses).mean(), np.vstack(non_prompt_losses).mean()
            print(f"[RANK {args.global_rank}] On all data - Train loss: {loss}, {prompt_loss}, {non_prompt_loss} || @ epoch {epoch}.")

        if args.global_rank == 0:
            logger.info(f'Train loss: {loss}, {prompt_loss}, {non_prompt_loss} || @ epoch {epoch}.')
        loss_list.append(loss.item())
        prompt_loss_list.append(prompt_loss.item())
        non_prompt_loss_list.append(non_prompt_loss.item())

        time_end = time.time()
        epoch_time = time_end - time_start

        print(f'[RANK {args.global_rank}] time_for_training {epoch_time}')
        if should_log:
            wandb.log({
                "epoch": epoch,
                "train/loss": loss,
                "train/prompt_loss": prompt_loss,
                "train/non_prompt_loss": non_prompt_loss,
                "train/epoch_time": epoch_time,
                #"train/lr_sam": optimizer1.param_groups[0]['lr'] if optimizer1 else 0,
                #"train/lr_mem": optimizer2.param_groups[0]['lr'] if optimizer2 else 0
                "train/lr_sam": optimizer1.optimizer.param_groups[0]['lr'] if optimizer1 else 0,
                "train/lr_mem": optimizer2.optimizer.param_groups[0]['lr'] if optimizer2 else 0,
                "train/lr_vit": optimizer3.optimizer.param_groups[0]['lr'] if optimizer3 else 0
            })

        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, rng)
            if args.distributed: ## Manual reduction
                if tol.dtype == torch.bfloat16:
                    tol = tol.float()
                tol, eiou, edice = tol.detach().cpu().numpy(), eiou, edice
                tols, eious, edices = [[None for _ in range(dist.get_world_size())] for _ in range(3)]
                dist.all_gather_object(tols, tol)
                dist.all_gather_object(eious, eiou)
                dist.all_gather_object(edices, edice)
                tol, eiou, edice = np.vstack(tols).mean(), np.vstack(eious).mean(), np.vstack(edices).mean()
                print(f"[RANK {args.global_rank}] On all data - Total loss: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.")
            
            if args.global_rank == 0:
                logger.info(f'Total loss: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            eiou_list.append(eiou.item())
            edice_list.append(edice.item())
            tol_list.append(tol.item())

            if args.global_rank == 0:
                torch.save(
                    {
                        'model': net.module.state_dict() if args.distributed else net.state_dict(),
                        'optimizer1': optimizer1.optimizer.state_dict() if optimizer1 else None,
                        'optimizer2': optimizer2.optimizer.state_dict() if optimizer2 else None,
                        'optimizer3': optimizer3.optimizer.state_dict() if optimizer3 else None,
                        'epoch': epoch,
                        'best_dice': best_dice,
                    },
                    os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth')
                )
                ## create multiple subplots recording the training loss and val metrics
                plot_loss_curve(loss_list, prompt_loss_list, non_prompt_loss_list, tol_list, eiou_list, edice_list, args)

                if edice > best_dice:
                    best_dice = edice
                    torch.save(
                        {
                            'model': net.module.state_dict() if args.distributed else net.state_dict(),
                            'optimizer1': optimizer1.optimizer.state_dict() if optimizer1 else None,
                            'optimizer2': optimizer2.optimizer.state_dict() if optimizer2 else None,
                            'optimizer3': optimizer3.optimizer.state_dict() if optimizer3 else None,
                            'epoch': epoch,
                            'best_dice': best_dice
                        },
                        os.path.join(args.path_helper['ckpt_path'], 'best_dice.pth')
                    )
            
            if should_log:
                wandb.log({
                    "val/loss": tol,
                    "val/iou": eiou,
                    "val/dice": edice
                })
            torch.distributed.barrier() ## wait until the main process writes to disk
                

    if args.global_rank == 0:
        writer.close()
    
    if should_log:
        wandb.finish()

def set_all_seeds(seed):
    print(f"Setting seed {seed} for deterministic training")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)

    return rng

if __name__ == '__main__':
    args = cfg.parse_args()
    args.distributed = True
    args.global_rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpu_device = int(os.environ["LOCAL_RANK"])
    rng = set_all_seeds(args.seed)
    if args.global_rank == 0:
        sanity_check(args, rng)
        torch.cuda.empty_cache()
    main(args, rng)