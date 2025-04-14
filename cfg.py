import argparse


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
    parser.add_argument('-exp_name', default='samba_train_test', type=str, help='experiment name')
    parser.add_argument('-vis', type=bool, default=False, help='Generate visualisation during validation')
    parser.add_argument('-train_vis', type=bool, default=False, help='Generate visualisation during training')
    #parser.add_argument('-prompt', type=str, default='bbox', help='type of prompt, bbox or click')
    parser.add_argument('-prompt_freq', type=int, default=2, help='frequency of giving prompt in 3D images')
    parser.add_argument('-pretrain', type=str, default=None, help='path of pretrain weights')
    parser.add_argument('-val_freq',type=int,default=5,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=1024, help='output_size')
    parser.add_argument('-distributed', default=None ,type=bool,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='btcv' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-sam_config', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-video_length', type=int, default=8, help='sam checkpoint address')
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=1,
        help='batch size for dataloader'
    )
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-lr_sam', type=float, default=1e-4, help='initial learning rate for SAM layers')
    #parser.add_argument('-lr_mem', type=float, default=1e-8, help='initial learning rate for memory layers')
    #parser.add_argument('-lr', type=float, default=5e-6, help='initial learning rate')
    #parser.add_argument('-lr_sam', type=float, default=5e-6, help='initial learning rate for SAM layers')
    parser.add_argument('-lr_mem', type=float, default=5e-6, help='initial learning rate for memory layers')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation')
    parser.add_argument('-memory_bank_size', type=int, default=16, help='sam 2d memory bank size')
    parser.add_argument('-max_targets', type=int, default=5, help='max number of targets for each image')
    parser.add_argument(
    '-data_path',
    type=str,
    default='./data/btcv',
    help='The path of segmentation data')
    parser.add_argument('-seed', type=int, default=42, help='seed for reproducible results.')
    parser.add_argument('-local_rank', type=int, default=-1, help='Local rank for distributed training (-1: not using distributed training)')
    parser.add_argument('-global_rank', type=int, default=-1, help='Global rank for distributed training (-1: not using distributed training)')
    parser.add_argument('-world_size', type=int, default=1, help='World size for distributed training')
    parser.add_argument('-resume', type=str, default=None, help='A checkpoint file to resume training from')
    parser.add_argument('-num_workers', type=int, default=8, help='Number of workers for dataloader')
    ## Wandb logging parameters
    parser.add_argument('--wandb_mode', type=str, default='offline', help='Wandb mode')
    parser.add_argument('--wandb_project', type=str, default='NeuroSAM2D', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')
    ## finetune image encoder
    parser.add_argument('--finetune_backbone', action='store_true', help='Finetune the image encoder')
    parser.add_argument('--finetune_neck', action='store_true', help='Finetune the neck')
    parser.add_argument('-lr_vit', type=float, default=5e-5, help='learning rate for the image encoder')
    opt = parser.parse_args()

    return opt
