""" Dataloader for the NEUROSAM dataset
    Anthony Rinaldi
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from func_3d.utils import generate_bbox, random_click
from glob import glob
import json
import os
import monai
from torch.nn import functional as F

class NEUROSAM(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        data_path: str,
        transform: Optional[Compose]=None,
        #transform_msk: Optional[Compose]=None,
        mode: str="Training",
        #prompt: str="click",
        seed: Optional[int]=None,
        variation: int=0,
    ):

        dataset_dir_names = [
            "AbdomenCT_1K",  # "AbdomenCT"
            "AMOS",  # "AMOS"
            "BraTs2020",  # "BRATS"
            "COVID_CT_Lung",  # "CovidCT"
            "CTStroke",  # "CTStroke"
            "Healthy-Total-Body CTs NIfTI Segmentations and Segmentation Organ Values spreadsheet",  # "HealthyTotalBody"
            "ISLES-2022",  # "ISLES"
            "kits23",  # "Kits"
            "KneeMRI",  # "StanfordKnee"
            "LITS",  # "LiTS"
            "LUNA16",  # "LUNA"
            "MM-WHS 2017 Dataset",  # "MultiModalWholeHeart"
            "MSD",  # "MedSamDecathlon"
            "PKG - CT-ORG",  # "CTOrgan"
            "PKG - UPENN-GBM-NIfTI",  # "MRIGlioblastoma"
            "Prostate MR Image Segmentation",  # "ProstateMRI"
            "SegTHOR",  # "SegThoracicOrgans"
            "TCIA_pancreas_labels-02-05-2017",  # "PancreasCt"
            "Totalsegmentator_dataset_v201",  # "TotalSegmentator"
            "wmh_hab",  # "ONDRI"
            "WORD-V0.1",  # "WORD"

            ## ----- New datasets ----- ##
            "adni",
            "oasis",
            "isles",
            "HCP",
            "ATLAS",
            "ondri"
        ]
        if mode == "Training":
            selected_datasets = [
                #"AbdomenCT_1K",
                #"BraTs2020",
                #"COVID_CT_Lung",
                #"CTStroke",
                #"ISLES-2022",
                #"LITS",
                #"LUNA16",
                #"PKG - UPENN-GBM-NIfTI",

                #### ----- Above used as subset for debugging ----- ##
                "CTStroke",   #"CTStroke",
                "PKG - UPENN-GBM-NIfTI",  # "MRIGlioblastoma"
                "ISLES-2022",  # "ISLES"
                "wmh_hab", #"ONDRIWMH",
                "ATLAS", #"ATLAS",
                "HCP", #"HCP",
                "ondri",   #'ONDRI_FS'
                #### ----- Above are brain datasets only ----- ##
            ]
            print(f"selected_datasets for training: {selected_datasets}")
        elif mode == "Validation":
            selected_datasets = [
                #"WORD-V0.1.0",

                ### ----- Above used as subset for debugging ----- ##
                'CTStroke',
                "PKG - UPENN-GBM-NIfTI",  # "MRIGlioblastoma"
                "ISLES-2022",  # "ISLES"
                "wmh_hab", #"ONDRIWMH",
                'ATLAS',
                'HCP',
                "ondri",   #'ONDRI_FS'
                ### ----- Above are brain datasets only ----- ##
            ]
            print(f"selected_datasets for validation: {selected_datasets}")
        elif mode == "Testing":
            selected_datasets = [
                "BraTs2020",  # "BRATS"
                "adni",
                "isles",#'ISLES-2018'
                ## ----- Above are brain datasets only ----- ##
            ]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # Set the data list for training
        self.name_list = []
        meta_info = []
        for ds in selected_datasets:
            if ds == "Totalsegmentator_dataset_v201":
                json_path = os.path.join(data_path, "Totalsegmentator_dataset_v201", "dataset_medsam2.json")
            else:
                json_path = os.path.join(data_path, ds, "dataset_splits_neurosam2d.json")
            try:
                with open(json_path, "r") as f:
                    meta_data = json.load(f)
                    if mode == "Training":
                        meta_info.extend(meta_data["training"])
                    elif mode == "Validation":
                        meta_info.extend(meta_data["validation"])
            except Exception as e:
                print(f"Error: {e}")
                print(f"json_path: {json_path}")
                continue
        name_list = []
        print(f"len(meta_info): {len(meta_info)}")
        for item in meta_info:
            try:
                name_list.append((item["image"], item["seg"]))
            except Exception as e:
                print(item)
                print(f"Error: {e}")
                continue

        name_list = list(set(name_list))
        name_list.sort(key=lambda x: x[0])
        self.name_list = name_list

        # Set the basic information of the dataset
        self.data_path = Path(data_path)
        self.mode = mode
        #self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        #self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == "Training":
            self.video_length = args.video_length
        else:
            self.video_length = None ## TODO: should we constrain the length in validation? this is taking so long
        self.max_targets = args.max_targets
        self.rng = np.random.RandomState(self.seed)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = name[0]
        if "/home/arinaldi/projects/" in img_path:
            img_path = img_path.replace(
                "/home/arinaldi/projects/rrg-mgoubran/arinaldi/data",
                "/home/lifeifei/projects/rrg-mgoubran/NeuroSAM_data/data"
            )
        elif "/project/aiconsgrp/bwood/NeuroSAM/data" in img_path:
            img_path = img_path.replace(
                "/project/aiconsgrp/bwood/NeuroSAM/data",
                "/home/lifeifei/projects/rrg-mgoubran/NeuroSAM_data/data"
            )
        mask_path = name[1]
        if "/home/arinaldi/projects/" in mask_path:
            mask_path = mask_path.replace(
                "/home/arinaldi/projects/rrg-mgoubran/arinaldi/data",
                "/home/lifeifei/projects/rrg-mgoubran/NeuroSAM_data/data"
            )
        elif "/project/aiconsgrp/bwood/NeuroSAM/data" in mask_path:
            mask_path = mask_path.replace(
                "/project/aiconsgrp/bwood/NeuroSAM/data",
                "/home/lifeifei/projects/rrg-mgoubran/NeuroSAM_data/data"
            )

        data_tensors = self.transform({"image": img_path, "label": mask_path})
        img_tensor_3d = data_tensors["image"] # [B, H, W, D]
        mask_tensor_3d = data_tensors["label"] # [B, H, W, D]


        if torch.sum(mask_tensor_3d) == 0:
            print(f"Skipping {name} due to no GT")
            return self.__getitem__(np.random.randint(0, len(self.name_list)))
        
        # only include slices with GT
        for i in range(mask_tensor_3d.shape[-1]):
            if torch.sum(mask_tensor_3d[..., i]) > 0:
                mask_tensor_3d = mask_tensor_3d[..., i:]
                img_tensor_3d = img_tensor_3d[..., i:]
                break
        
        starting_frame_nonzero = i

        for j in reversed(range(mask_tensor_3d.shape[-1])):
            if torch.sum(mask_tensor_3d[..., j]) > 0:
                mask_tensor_3d = mask_tensor_3d[..., : j + 1]
                img_tensor_3d = img_tensor_3d[..., : j + 1]
                break
        
        num_frame = mask_tensor_3d.shape[-1]

        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length
        
        if num_frame < video_length:
            ## Do padding instead of skipping, some datasets have far fewer frames than others
            img_tensor_3d = F.pad(
                input = img_tensor_3d,
                pad = (0, video_length - img_tensor_3d.size(-1)),
                mode = "constant",
                value = 0
            )
            mask_tensor_3d = F.pad(
                input = mask_tensor_3d,
                pad = (0, video_length - mask_tensor_3d.size(-1)),
                mode = "constant",
                value = 0
            )
            assert img_tensor_3d.shape == mask_tensor_3d.shape
        
        if num_frame > video_length and self.mode == "Training":

            starting_frame = np.random.randint(0, num_frame - video_length)
        else:
            starting_frame = 0
        
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            try:
                img = img_tensor_3d[..., frame_index]
                img = img.squeeze(0)
                img = img.repeat(3, 1, 1)
                mask = mask_tensor_3d[..., frame_index]
                mask = mask.int()
                
            except Exception as e:
                print(f"Error: {e}")
                print(f"name: {name}")
                print(f"video length: {video_length}")
                print(f"num frame: {num_frame}")
                print(f"starting frame: {starting_frame}")
                print(f"starting frame nonzero: {starting_frame_nonzero}")
                exit()

            mask = mask.numpy()
            obj_list = np.unique(mask[mask > 0])
            if self.max_targets is not None and len(obj_list) > self.max_targets:
                choose_targets = self.rng.choice(obj_list, size=self.max_targets, replace=False)
                obj_list = choose_targets
                mask_new = np.zeros_like(mask)
                for obj in obj_list:
                    mask_new[mask == obj] = obj
                mask = mask_new
            
            ## TODO: Add dynamic prompt suggestion here
            
            diff_obj_mask_dict = {}

            diff_obj_bbox_dict = {}
            diff_obj_pt_dict = {}
            diff_obj_point_label_dict = {}
            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask = obj_mask.astype(int)

                diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = (
                    random_click(
                        np.squeeze(obj_mask, 0),
                        point_label,
                        seed=self.seed
                    )
                )
                bbox = generate_bbox(
                    np.squeeze(obj_mask, 0),
                    variation=self.variation,
                    seed=self.seed,
                )
                if np.isnan(bbox).any():
                    print(f"bbox is nan for {obj}")
                    print(f"bbox: {bbox}")
                    continue
                diff_obj_bbox_dict[obj] = bbox

                diff_obj_mask_dict[obj] = torch.tensor(obj_mask)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
            point_label_dict[frame_index - starting_frame] = (
                diff_obj_point_label_dict
            )

        image_meta_dict = {"filename_or_obj": name}

        pack = {
            "image": img_tensor,
            "label": mask_dict,
            "bbox": bbox_dict,
            "p_label": point_label_dict,
            "pt": pt_dict,
            "image_meta_dict": image_meta_dict,
        }
        return pack
