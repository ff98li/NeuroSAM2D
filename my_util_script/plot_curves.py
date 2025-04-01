import os
import time

import torch
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np

def plot_loss_curve(
    loss_list,
    tol_list,
    eiou_list,
    edice_list,
    save_path,
    title = None,
    ):
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))


    #    'dice': '#f39c12'              # orange for metrics
    #}

    colors = {
        'total_train': '#27ae60',      # green
        'validation': '#e84393',       # pink
        'iou': '#2980b9',              # blue for metrics
        'dice': '#f39c12'              # orange for metrics
    }

    ax1.plot(loss_list, 'o-', color=colors['total_train'], label='Total training Loss', linewidth=2, markersize=5)
    #ax1.plot(prompt_loss_list, 'o-', color=colors['prompt'], label='Training prompt Loss', linewidth=2, markersize=5)
    #ax1.plot(non_prompt_loss_list, 'o-', color=colors['non_prompt'], label='Training non-Prompt Loss', linewidth=2, markersize=5)
    ax1.plot(tol_list, 'o-', color=colors['validation'], label='Total validation Loss', linewidth=2, markersize=5)

    ax1.set_title(
        'Loss' if title is None else title,
    pad=20, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(bottom=0)  # Start y-axis from 0
    ax1.set_ylim(top=1)

    ax2.plot(eiou_list, 'o-', color=colors['iou'], label='IOU', linewidth=2, markersize=5)
    ax2.plot(edice_list, 'o-', color=colors['dice'], label='DICE', linewidth=2, markersize=5)

    ax2.set_title('Validation Metrics', pad=20, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True, linestyle='--', alpha=0.7)
    #ax2.set_ylim(bottom=0, top=1)
    ax2.set_ylim(bottom=0)
    ax2.legend(frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, 
                dpi=600, 
                bbox_inches='tight',
                )
    plt.close()

if __name__ == '__main__':
    ## 2-gpu-full model
    #checkpoint_path = "/home/lifeifei/scratch/data/NeuroSAM/Medical-SAM2/logs/medsam-slice-192img-16video-2gpu-2_2024_12_05_05_05_50/Model/latest_epoch.pth"
    #title = "Distributed fine-tuning Medical-SAM2 on 2 GPUs (image encoder, memory attention, mask decoder)"
    #save_path = "/home/lifeifei/scratch/data/NeuroSAM/Medical-SAM2/temp/full-2-gpu.png"
    ## 2-gpu mask and memory only
    #checkpoint_path = "/home/lifeifei/scratch/data/NeuroSAM/Medical-SAM2/logs/medsam-slice-192img-16video-2gpu-2_2024_12_04_20_22_41/Model/latest_epoch.pth"
    #title = "Distributed fine-tuning Medical-SAM2 on 2 GPUs (memory attention, mask decoder)"
    #save_path = "/home/lifeifei/scratch/data/NeuroSAM/Medical-SAM2/temp/mask-memory-2-gpu.png"
    ## 1-gpu mask and memory only

    #checkpoint_path = "/home/lifeifei/scratch/data/NeuroSAM/Medical-SAM2/logs/medsam-slice-192img-16video-1gpu_2024_12_04_03_22_00/Model/latest_epoch.pth"
    #title = "Fine-tuning Medical-SAM2 on 1 GPU (memory attention, mask decoder)"
    #save_path = "/home/lifeifei/scratch/data/NeuroSAM/Medical-SAM2/temp/mask-memory-1-gpu.png"

    checkpoint_path = "/home/lifeifei/scratch/data/NeuroSAM/Medical-SAM2/logs/medsam2-nifiti-192img-16video-2gpu-full-subset_2025_02_25_07_22_35/Model/latest_epoch.pth"
    title = "Fine-tuning Medical-SAM2 on 2 GPU (MONAI Pipeline, Nifti Data)"
    save_path = "/home/lifeifei/scratch/data/NeuroSAM/Medical-SAM2/temp/nifti-2-gpu.png"

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    loss_list = checkpoint['loss_list']
    tol_list = checkpoint['tol_list']
    eiou_list = checkpoint['eiou_list']
    edice_list = checkpoint['edice_list']
    plot_loss_curve(loss_list, tol_list, eiou_list, edice_list, save_path, title)
