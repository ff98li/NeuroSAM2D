""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

import cfg
from conf import settings
from func_3d.utils import eval_seg
from my_util_script.visualize import show_mask
from func_3d.loss_fns import dice_loss, sigmoid_focal_loss, iou_loss
from conf import settings
from typing import List

args = cfg.parse_args()

weight_dict = {
    "loss_mask": 20,
    "loss_dice": 1,
    "loss_iou": 1,
    "loss_class": 1
}
iou_use_l1_loss = True
pred_obj_scores = True
focal_gamma_obj_score = 0.0
focal_alpha_obj_score = -1.0
loss_on_multimask = False

class CombinedLoss(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        iou_use_l1_loss=True,
        pred_obj_scores=True,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        Compute segmentation loss and auxiliary losses (iou, object score)
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, pred_masks, target_masks, pred_ious, pred_scores):
        """
        Args:
            pred_masks: List[Tensor] of shape [1, H, W] (selected best mask)
            target_masks: List[Tensor] of shape [H, W]
            pred_ious: List[Tensor] of shape [1] (IoU of selected mask)
            pred_scores: List[Tensor] of shape [1] (object score of selected mask)
        """
        losses = {
            "loss_mask": 0.0,
            "loss_dice": 0.0,
            "loss_iou": 0.0,
            "loss_class": 0.0,
        }

        loss_mask = sigmoid_focal_loss(
            pred_masks,
            target_masks,
            num_objects = 1,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask = True,
        )

        loss_dice = dice_loss(
            pred_masks,
            target_masks,
            num_objects = 1,
            loss_on_multimask = True,
        )

        pred_ious = pred_ious.unsqueeze(0) ## [1] -> [1, 1]
        loss_iou = iou_loss(
            pred_masks,
            target_masks,
            pred_ious,
            num_objects = 1,
            loss_on_multimask = True,
            use_l1_loss = self.iou_use_l1_loss,
        )

        pred_scores = pred_scores.unsqueeze(0) ## [1] -> [1, 1]
        target_obj = torch.ones_like(pred_scores)
        loss_class = sigmoid_focal_loss(
            pred_scores,
            target_obj,
            num_objects = 1,
            alpha = self.focal_alpha_obj_score,
            gamma = self.focal_gamma_obj_score,
            loss_on_multimask = False,
        )

        losses["loss_mask"] += loss_mask.sum() * target_obj
        losses["loss_dice"] += loss_dice.sum() * target_obj
        losses["loss_iou"] += loss_iou.sum() * target_obj
        losses["loss_class"] += loss_class.sum() * target_obj
        
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        
        return reduced_loss
                
combined_loss = CombinedLoss(
    weight_dict,
    focal_alpha=0.25,
    focal_gamma=2,
    iou_use_l1_loss=iou_use_l1_loss,
    pred_obj_scores=pred_obj_scores,
    focal_gamma_obj_score=focal_gamma_obj_score,
    focal_alpha_obj_score=focal_alpha_obj_score
)

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
#criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#paper_loss = CombinedLoss(dice_weight=1 / 21, focal_weight=20 / 21)
#seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
#scaler = torch.cuda.amp.GradScaler() ## bfloat16 doesn't need grad scaler
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

combined_loss = CombinedLoss(
    weight_dict,
    focal_alpha=0.25,
    focal_gamma=2,
    iou_use_l1_loss=iou_use_l1_loss,
    pred_obj_scores=pred_obj_scores,
    focal_gamma_obj_score=focal_gamma_obj_score,
    focal_alpha_obj_score=focal_alpha_obj_score
)

def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader,
          epoch, rng):
    hard = 0
    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    ind = 0
    # train mode
    seed = args.seed
    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    video_length = args.video_length * args.batch_size

    if args.distributed:
        GPUdevice = torch.device('cuda:' + str(args.local_rank))
        pbar_desc = f'[RANK {args.global_rank}] Epoch {epoch}'
    else:
        GPUdevice = torch.device('cuda:' + str(args.gpu_device))
        pbar_desc = f'[GPU {args.gpu_device}] Epoch {epoch}'
    prompt_freq = args.prompt_freq

    lossfunc = combined_loss.to(device=GPUdevice, dtype=torch.bfloat16)
    total_steps = len(train_loader) * settings.EPOCH
    current_step = epoch * len(train_loader)
    where = current_step / total_steps

    with tqdm(total=len(train_loader), desc=pbar_desc, unit='img') as pbar:
        for pack in train_loader:
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            name = pack['image_meta_dict']['filename_or_obj']
            prompt = rng.choice(['click', 'bbox'])
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
            
            if args.distributed:
                train_state = net.module.train_init_state(imgs_tensor=imgs_tensor)
            else:
                train_state = net.train_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, video_length, prompt_freq))
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            with torch.cuda.amp.autocast():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                points = points.reshape(-1, 1, 2)
                                labels = labels.reshape(-1, 1)
                                for prompt_idx in range(len(points)):
                                    if args.distributed:
                                        _, _, _ = net.module.train_add_new_points(
                                            inference_state=train_state,
                                            frame_idx=id,
                                            obj_id=ann_obj_id,
                                            points=points[prompt_idx],
                                            labels=labels[prompt_idx],
                                            clear_old_points=False,
                                        )
                                    else:
                                        _, _, _ = net.train_add_new_points(
                                            inference_state=train_state,
                                            frame_idx=id,
                                            obj_id=ann_obj_id,
                                            points=points[prompt_idx],
                                            labels=labels[prompt_idx],
                                            clear_old_points=False,
                                        )

                                #if args.distributed:
                                #    _, _, _ = net.module.train_add_new_points(
                                #        inference_state=train_state,
                                #        frame_idx=id,
                                #        obj_id=ann_obj_id,
                                #        points=points,
                                #        labels=labels,
                                #        clear_old_points=False,
                                #    )
                                #else:
                                #    _, _, _ = net.train_add_new_points(
                                #        inference_state=train_state,
                                #        frame_idx=id,
                                #        obj_id=ann_obj_id,
                                #        points=points,
                                #        labels=labels,
                                #        clear_old_points=False,
                                #    )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                bbox = bbox.reshape(-1, 1, 4)
                                for prompt_idx in range(len(bbox)):
                                    if args.distributed:
                                        _, _, _ = net.module.train_add_new_bbox(
                                            inference_state=train_state,
                                            frame_idx=id,
                                            obj_id=ann_obj_id,
                                            bbox=bbox[prompt_idx].to(device=GPUdevice),
                                            clear_old_points=False,
                                        )
                                    else:
                                        _, _, _ = net.train_add_new_bbox(
                                            inference_state=train_state,
                                            frame_idx=id,
                                            obj_id=ann_obj_id,
                                            bbox=bbox[prompt_idx].to(device=GPUdevice),
                                            clear_old_points=False,
                                        )

                                #if args.distributed:
                                #    _, _, _ = net.module.train_add_new_bbox(
                                #        inference_state=train_state,
                                #        frame_idx=id,
                                #        obj_id=ann_obj_id,
                                #        bbox=bbox.to(device=GPUdevice),
                                #        clear_old_points=False,
                                #    )
                                #else:
                                #    _, _, _ = net.train_add_new_bbox(
                                #        inference_state=train_state,
                                #        frame_idx=id,
                                #        obj_id=ann_obj_id,
                                #        bbox=bbox.to(device=GPUdevice),
                                #        clear_old_points=False,
                                #    )
                        except KeyError:
                            if args.distributed:
                                _, _, _ = net.module.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                                )
                            else:
                                _, _, _ = net.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                                )
                video_segments = {}  # video_segments contains the per-frame segmentation results
            
                if args.distributed:
                    for out_frame_idx, out_obj_ids, out_mask_logits, out_pred_ious, out_object_score_logits in net.module.train_propagate_in_video(train_state, start_frame_idx=0):
                        video_segments[out_frame_idx] = {
                            out_obj_id: {
                                "mask": out_mask_logits[i], 
                                "iou": out_pred_ious[i][torch.argmax(out_pred_ious[i])].unsqueeze(0),
                                "score": out_object_score_logits[i]
                            }
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                else:
                    for out_frame_idx, out_obj_ids, out_mask_logits, out_pred_ious, out_object_score_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
                        video_segments[out_frame_idx] = {
                            out_obj_id: {
                                "mask": out_mask_logits[i], 
                                "iou": out_pred_ious[i][torch.argmax(out_pred_ious[i])].unsqueeze(0),
                                "score": out_object_score_logits[i]
                            }
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }

                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred_data = video_segments[id][ann_obj_id]
                        pred_mask = pred_data["mask"]
                        pred_mask = pred_mask.unsqueeze(0)
                        pred_iou = pred_data["iou"]
                        pred_score = pred_data["score"]
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros(pred_mask.shape).to(device=GPUdevice)
                        if args.train_vis and (not args.distributed or args.global_rank == 0):
                            os.makedirs(f'./temp/train/{os.path.basename(name[0]).split(".")[0]}/{id}', exist_ok=True)
                            img_show = imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
                            pred_show = torch.sigmoid(pred_mask)[0, 0, :, :].detach().cpu().numpy() > 0.5
                            mask_show = mask[0, 0, :, :].detach().cpu().numpy()

                            img_show = (img_show * 255).astype('uint8')
                            pred_show = pred_show.astype('uint8')
                            mask_show = mask_show.astype('uint8')

                            pred_overlay = show_mask(pred_show, img_show)
                            mask_overlay = show_mask(mask_show, img_show)

                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(img_show)
                            ax[0].axis('off')
                            ax[1].imshow(pred_overlay)
                            ax[1].axis('off')
                            try:
                                bbox = bbox_dict[id][ann_obj_id]
                                ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            except KeyError:
                                pass
                            ax[2].imshow(mask_overlay)
                            ax[2].axis('off')
                            plt.savefig(f'./temp/train/{os.path.basename(name[0]).split(".")[0]}/{id}/{obj_list.index(ann_obj_id)}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        #obj_loss = lossfunc(pred, mask)
                        obj_loss = lossfunc(pred_mask, mask, pred_iou, pred_score)
                        loss += obj_loss.item()
                        if id in prompt_frame_id:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss
                loss = loss / video_length / len(obj_list)
                if prompt_freq > 1:
                    non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                pbar.set_postfix(**{f'{args.gpu_device} loss (batch)': loss})
                epoch_loss += loss
                epoch_prompt_loss += prompt_loss.item()
                if prompt_freq > 1:
                    epoch_non_prompt_loss += non_prompt_loss.item()

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                if non_prompt_loss is not int and optimizer2 is not None and prompt_freq > 1:
                    non_prompt_loss.backward(retain_graph=True)
                    optimizer2.step(where = where, step = current_step)
                    #optimizer2.step()
                if optimizer1 is not None:
                    prompt_loss.backward()
                    optimizer1.step(where = where, step = current_step)
                    #optimizer1.step()
                    optimizer1.zero_grad()
                if optimizer2 is not None:
                    optimizer2.zero_grad()
                if args.distributed:
                    net.module.reset_state(train_state)
                else:
                    net.reset_state(train_state)
                current_step += 1
                where = current_step / total_steps

            pbar.update()
            torch.cuda.empty_cache()

    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(train_loader)

def validation_sam(args, val_loader, epoch, net: nn.Module, rng):
     # eval mode

    if args.distributed:
        GPUdevice = torch.device('cuda:' + str(args.local_rank))
        pbar_desc = f'[RANK {args.global_rank}] Epoch {epoch}'
    else:
        GPUdevice = torch.device('cuda:' + str(args.gpu_device))
        pbar_desc = f'[GPU {args.gpu_device}] Epoch {epoch}'
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq
    lossfunc = combined_loss.to(device=GPUdevice, dtype=torch.bfloat16)

    prompt = rng.choice(['click', 'bbox'])
    print(f"Running validation with [{prompt}] prompt")

    with tqdm(total=n_val, desc=pbar_desc, unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            name = pack['image_meta_dict']['filename_or_obj']
            if args.batch_size > 1:
                name_list = [
                    os.path.basename(fname[0]).split(".")[0] for fname in name
                ]
                name_join = "_".join(name_list)
                name = name_join
            else:
                name = os.path.basename(name[0][0]).split(".")[0]
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))
            
            if args.distributed:
                train_state = net.module.val_init_state(imgs_tensor=imgs_tensor)
            else:
                train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            #train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                points = points.reshape(-1, 1, 2) # [B, N, 2] -> [B*N, 1, 2]
                                labels = labels.reshape(-1, 1) # [B, N] -> [B*N]
                                for prompt_idx in range(len(points)):
                                    if args.distributed:
                                        _, _, _ = net.module.train_add_new_points(
                                            inference_state=train_state,
                                            frame_idx=id,
                                            obj_id=ann_obj_id,
                                            points=points[prompt_idx],
                                            labels=labels[prompt_idx],
                                            clear_old_points=False,
                                        )
                                    else:
                                        _, _, _ = net.train_add_new_points(
                                            inference_state=train_state,
                                            frame_idx=id,
                                            obj_id=ann_obj_id,
                                            points=points[prompt_idx],
                                            labels=labels[prompt_idx],
                                            clear_old_points=False,
                                        )

                                #if args.distributed:
                                #    _, _, _ = net.module.train_add_new_points(
                                #        inference_state=train_state,
                                #        frame_idx=id,
                                #        obj_id=ann_obj_id,
                                #        points=points,
                                #        labels=labels,
                                #        clear_old_points=False,
                                #    )
                                #else:
                                #    _, _, _ = net.train_add_new_points(
                                #        inference_state=train_state,
                                #        frame_idx=id,
                                #        obj_id=ann_obj_id,
                                #        points=points,
                                #        labels=labels,
                                #        clear_old_points=False,
                                #    )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                bbox = bbox.reshape(-1, 1, 4) # [B, N, 4] -> [B*N, 4]
                                for prompt_idx in range(len(bbox)):
                                    if args.distributed:
                                        _, _, _ = net.module.train_add_new_bbox(
                                            inference_state=train_state,
                                            frame_idx=id,
                                            obj_id=ann_obj_id,
                                            bbox=bbox[prompt_idx].to(device=GPUdevice),
                                            clear_old_points=False,
                                        )
                                    else:
                                        _, _, _ = net.train_add_new_bbox(
                                            inference_state=train_state,
                                            frame_idx=id,
                                            obj_id=ann_obj_id,
                                            bbox=bbox[prompt_idx].to(device=GPUdevice),
                                            clear_old_points=False,
                                        )

                                #if args.distributed:
                                #    _, _, _ = net.module.train_add_new_bbox(
                                #        inference_state=train_state,
                                #        frame_idx=id,
                                #        obj_id=ann_obj_id,
                                #        bbox=bbox.to(device=GPUdevice),
                                #        clear_old_points=False,
                                #    )
                                #else:
                                #    _, _, _ = net.train_add_new_bbox(
                                #        inference_state=train_state,
                                #        frame_idx=id,
                                #        obj_id=ann_obj_id,
                                #        bbox=bbox.to(device=GPUdevice),
                                #        clear_old_points=False,
                                #    )
                        except KeyError:
                            if args.distributed:
                                _, _, _ = net.module.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                                )
                            else:
                                _, _, _ = net.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                                )
                video_segments = {}  # video_segments contains the per-frame segmentation results
            
                if args.distributed:
                    #for out_frame_idx, out_obj_ids, out_mask_logits in net.module.propagate_in_video(train_state, start_frame_idx=0):
                    #    video_segments[out_frame_idx] = {
                    #        out_obj_id: out_mask_logits[i]
                    #        for i, out_obj_id in enumerate(out_obj_ids)
                    #    }
                    for out_frame_idx, out_obj_ids, out_mask_logits, out_pred_ious, out_object_score_logits in net.module.propagate_in_video(train_state, start_frame_idx=0):
                        video_segments[out_frame_idx] = {
                            out_obj_id: {
                                "mask": out_mask_logits[i], 
                                "iou": out_pred_ious[i][torch.argmax(out_pred_ious[i])].unsqueeze(0),
                                "score": out_object_score_logits[i]
                            }
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                else:
                    #for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    #    video_segments[out_frame_idx] = {
                    #        out_obj_id: out_mask_logits[i]
                    #        for i, out_obj_id in enumerate(out_obj_ids)
                    #    }
                    for out_frame_idx, out_obj_ids, out_mask_logits, out_pred_ious, out_object_score_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                        video_segments[out_frame_idx] = {
                            out_obj_id: {
                                "mask": out_mask_logits[i], 
                                "iou": out_pred_ious[i][torch.argmax(out_pred_ious[i])].unsqueeze(0),
                                "score": out_object_score_logits[i]
                            }
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }

                loss = 0
                #pred_iou = 0
                #pred_dice = 0
                val_iou = 0
                val_dice = 0
                for id in frame_id:
                    for ann_obj_id in obj_list:
                        #pred = video_segments[id][ann_obj_id]
                        #pred = pred.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        pred_data = video_segments[id][ann_obj_id]
                        pred_mask = pred_data["mask"]
                        pred_mask = pred_mask.unsqueeze(0)
                        pred_iou = pred_data["iou"]
                        pred_score = pred_data["score"]
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            #mask = torch.zeros_like(pred).to(device=GPUdevice)
                            mask = torch.zeros(pred_mask.shape).to(device=GPUdevice)
                        if args.vis and (not args.distributed or args.global_rank == 0):
                            os.makedirs(f'./temp/val/{name}/{id}', exist_ok=True)
                            img_show = imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
                            #pred_show = pred[0, 0, :, :].detach().cpu().numpy() > 0.5
                            #pred_show = pred_mask[0, 0, :, :].detach().cpu().numpy() > 0.5
                            pred_show = torch.sigmoid(pred_mask)[0, 0, :, :].detach().cpu().numpy() > 0.5
                            mask_show = mask[0, 0, :, :].detach().cpu().numpy()

                            img_show = (img_show * 255).astype('uint8')
                            pred_show = pred_show.astype('uint8')
                            mask_show = mask_show.astype('uint8')

                            pred_overlay = show_mask(pred_show, img_show)
                            mask_overlay = show_mask(mask_show, img_show)

                            fig, ax = plt.subplots(1, 3)
                            #ax[0].imshow(imgs_tensor[id, :, :, :].cpu().permute(1, 2, 0).numpy())
                            ax[0].imshow(img_show)
                            ax[0].axis('off')
                            #ax[1].imshow(pred[0, 0, :, :].cpu().numpy() > 0.5, cmap='gray')
                            ax[1].imshow(pred_overlay)
                            ax[1].axis('off')
                            #ax[2].imshow(mask[0, 0, :, :].cpu().numpy(), cmap='gray')
                            ax[2].imshow(mask_overlay)
                            ax[2].axis('off')
                            plt.savefig(f'./temp/val/{name}/{id}/{ann_obj_id}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        #loss += lossfunc(pred, mask)
                        loss += lossfunc(pred_mask, mask, pred_iou, pred_score)
                        #temp = eval_seg(pred, mask, threshold)
                        temp = eval_seg(pred_mask, mask, threshold)
                        #pred_iou += temp[0]
                        #pred_dice += temp[1]
                        val_iou += temp[0]
                        val_dice += temp[1]

                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                #temp = (pred_iou / total_num, pred_dice / total_num)
                temp = (val_iou / total_num, val_dice / total_num)
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            if args.distributed:
                net.module.reset_state(train_state)
            else:
                net.reset_state(train_state)
            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])
