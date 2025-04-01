"""
Modified from NeuroSAM3D's prepare_json_data.py
This script is used to prepare the data splits for the NeuroSAM2 dataset.
It will create a new json file in the dataset directory with the same name as the dataset, but with the suffix "_splits.json".
The final data split scheme is pending.
"""
import os
import json
import shutil
import os.path as osp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Union

import grp
import subprocess

# Ratios
TRAIN = 0.80
VAL = 0.20

# Dataset Splits
TRAIN_SETS = ['CTStroke', ## Has dataset.json, has validation split; No all_image_label_pairs.json
              'MRIGlioblastoma', ## Has dataset.json, no validation split; No all_image_label_pairs.json
              'ISLES', ## Has dataset.json, no validation split; No all_image_label_pairs.json
              'ONDRIWMH', ## Has dataset.json, has validation split; No all_image_label_pairs.json
              'ATLAS', ## No dataset.json, has all_image_label_pairs.json
              'HCP', ## Has dataset.json, no validation split; Has all_image_label_pairs.json
              'ONDRI_FS'] ## Has dataset.json, no validation split; Has all_image_label_pairs.json

VAL_SETS = ['CTStroke', ## Has dataset.json, has validation split; No all_image_label_pairs.json
            'MRIGlioblastoma', ## Has dataset.json, no validation split; No all_image_label_pairs.json
            'ISLES', ## Has dataset.json, no validation split; No all_image_label_pairs.json
            'ONDRIWMH', ## Has dataset.json, has validation split; No all_image_label_pairs.json
            'ATLAS', ## No dataset.json, has all_image_label_pairs.json
            'HCP', ## Has dataset.json, no validation split; Has all_image_label_pairs.json
            'ONDRI_FS'] ## Has dataset.json, no validation split; Has all_image_label_pairs.json

TEST_SETS = ['BRATS', ## Has dataset.json, no validation split; No all_image_label_pairs.json
             'adni', ## Has dataset.json, no validation split; Has all_image_label_pairs.json
             'ISLES-2018'] ## Has dataset.json, no validation split; Has all_image_label_pairs.json

DATA_ROOT = "/home/lifeifei/projects/rrg-mgoubran/NeuroSAM_data/data"
# Just brain datasets (less OASIS)
ADNI = {'name':'adni', 
        'path': osp.join(DATA_ROOT, 'adni')}
ATLAS = {'name':'ATLAS', 
         'path': osp.join(DATA_ROOT, 'ATLAS')}
BRATS2020 = {'name':'BRATS', 
             'path': osp.join(DATA_ROOT, 'BraTs2020')}
CT_STROKE = {'name':'CTStroke', 
             'path': osp.join(DATA_ROOT, 'CTStroke')}
HCP = {'name':'HCP', 
       'path': osp.join(DATA_ROOT, 'HCP')}
ISLES_2018 = {'name':'ISLES-2018', 
              'path': osp.join(DATA_ROOT, 'isles')}
ISLES_2022 = {'name':'ISLES', 
              'path': osp.join(DATA_ROOT, 'ISLES-2022')}
ONDRIFS = {'name':'ONDRI_FS', 
           'path': osp.join(DATA_ROOT, 'ondri')}
ONDRIWMH = {'name':'ONDRIWMH', 
            'path': osp.join(DATA_ROOT, 'wmh_hab')}
UPENN = {'name':'MRIGlioblastoma', 
         'path': osp.join(DATA_ROOT, 'PKG - UPENN-GBM-NIfTI')}

def split_data(path: Union[Path, str], name: str):
    """Splits data in to train, validation and test data sets for use in prepare_json_data.py

    Args:
        path (Path): Path to parent dataset
        name (str): Name of dataset

    Raises:
        RuntimeError: If the same dataset is in the training and testing set
    """
    #dest_dir = TARGET_DATASET_DIR.joinpath(name)
    print(f'########## {name} ##########')
    
    #all_data = json.load(open(path.joinpath('all_image_label_pairs.json'), 'r'))
    all_data_json_path = osp.join(path, 'all_image_label_pairs.json')
    dataset_json_path = osp.join(path, 'dataset.json')
    
    #########################################################
    save_json_path = osp.join(path, 'dataset_splits_neurosam2d.json')
    #save_json_path = osp.join("/home/lifeifei/scratch/data/NeuroSAM/Medical-SAM2/temp/new_dataset_json", f'{name}.json')
    #########################################################
    if osp.isfile(save_json_path):
        os.remove(save_json_path)

    #if osp.isfile(all_data_json_path):
    #    all_data = json.load(open(all_data_json_path, 'r'))
    #else:
    #    raise FileNotFoundError(f'{all_data_json_path} not found')
    
    has_dataset_json = osp.isfile(dataset_json_path)
    has_validation_split = False

    if has_dataset_json:
        dataset_json_dict = json.load(open(dataset_json_path, 'r'))
        if 'validation' in dataset_json_dict and dataset_json_dict['validation']:
            has_validation_split = True
    else:
        has_validation_split = False

    has_all_data_json = osp.isfile(all_data_json_path)

    new_dataset_json_dict = {}
    if has_dataset_json and has_validation_split: ## directly use dataset.json without any changes
        new_dataset_json_dict["name"] = dataset_json_dict["name"]
        new_dataset_json_dict["modality"] = dataset_json_dict["modality"]
        new_dataset_json_dict["labels"] = dataset_json_dict["labels"]

        train_data_dups = dataset_json_dict["training"]
        val_data_dups = dataset_json_dict["validation"]


        train_data_unique = list(
            set(
                [
                    (pair['image'], pair['seg'])
                    for pair in train_data_dups
                ]
            )
        )
        new_dataset_json_dict["training"] = [
            {
                "image": pair[0],
                "seg": pair[1],
            }
            for pair in train_data_unique
        ]
        
        val_data_unique = list(
            set(
                [
                    (pair['image'], pair['seg'])
                    for pair in val_data_dups
                ]
            )
        )
        new_dataset_json_dict["validation"] = [
            {
                "image": pair[0],
                "seg": pair[1],
            }
            for pair in val_data_unique
        ]

        #json.dump(new_dataset_json_dict, open(save_json_path, 'w'), indent=4)

    elif has_all_data_json: ## either dataset.json doesn't exist or doesn't have validation split
        all_data_dict = json.load(open(all_data_json_path, 'r'))

        train_data_all = []
        val_data_all = []
        test_data_all = []

        for modality, labels in all_data_dict.items():
            for label, pairs in labels.items():
                train_data, val_data, test_data = None, None, None

                if name in TRAIN_SETS and name in TEST_SETS:
                    raise RuntimeError(f'{name} in both Train and Test sets!')

                elif name in TRAIN_SETS and name in VAL_SETS:
                    train_data, val_data = _get_random_split(pairs)
                    train_data_all.extend(train_data)
                    val_data_all.extend(val_data)

                elif name in TRAIN_SETS:
                    train_data = pairs
                    train_data_all.extend(train_data)

                elif name in TEST_SETS:
                    test_data = pairs
                    test_data_all.extend(test_data)
        
        new_dataset_json_dict["name"] = name
        new_dataset_json_dict["training"] = train_data_all
        new_dataset_json_dict["validation"] = val_data_all
        if len(test_data_all) > 0:
            new_dataset_json_dict["testing"] = test_data_all

        #json.dump(new_dataset_json_dict, open(save_json_path, 'w'), indent=4)
    
    else: ## has dataset.json but no validation split and no all_image_label_pairs.json
        assert has_dataset_json, f'{name} has no dataset.json'
        new_dataset_json_dict["name"] = dataset_json_dict["name"]
        new_dataset_json_dict["modality"] = dataset_json_dict["modality"]
        new_dataset_json_dict["labels"] = dataset_json_dict["labels"]

        all_data_dups = dataset_json_dict["training"]
        all_data_unique = list(
            set(
                [
                    (pair['image'], pair['seg'])
                    for pair in all_data_dups
                ]
            )
        )
        all_data_unique = sorted(all_data_unique, key=lambda x: x[1]) ## Gurantee reproducible split
        train_data, val_data = _get_random_split(all_data_unique)
        new_dataset_json_dict["training"] = [
            {
                "image": pair[0],
                "seg": pair[1],
            }
            for pair in train_data
        ]
        new_dataset_json_dict["validation"] = [
            {
                "image": pair[0],
                "seg": pair[1],
            }
            for pair in val_data
        ]

    json.dump(new_dataset_json_dict, open(save_json_path, 'w'), indent=4)

    # Option 1: Change group using os.chown
    try:
        # Get the group ID
        group_name='rrg-mgoubran'
        group_id = grp.getgrnam(group_name).gr_gid
        
        # Get current owner to keep it unchanged
        file_stat = os.stat(save_json_path)
        owner_id = file_stat.st_uid
        
        # Change the group ownership
        os.chown(save_json_path, owner_id, group_id)
        print(f"Group permission changed to {group_name} using os.chown")
        return True
    except Exception as e:
        print(f"Error with os.chown method: {e}")
        
        # Option 2: Fall back to chgrp command
        try:
            subprocess.run(['chgrp', group_name, save_json_path], check=True)
            print(f"Group permission changed to {group_name} using chgrp command")
            return True
        except Exception as sub_e:
            print(f"Error with chgrp method: {sub_e}")
            return False



                
#def _copy_data(data: List[Dict], image_path: Path, label_path: Path):
#    """Copies data form source to image_path and label_path.
#
#    Args:
#        data (List[dict]): List of dictionaries containing image segmentation pairs.
#        image_path (Path): Path to where images are saved.
#        label_path (Path): Path to where labels are saved
#    """
#
#    return
#
#    if not image_path.is_dir():
#        image_path.mkdir(parents=True)
#        
#    if not label_path.is_dir():
#        label_path.mkdir(parents=True)
#    
#    for pair in tqdm(data):
#        img_name = osp.basename(pair['image'])
#        lbl_name = osp.basename(pair['seg'])
#        
#        if not image_path.joinpath(img_name).is_file():
#            shutil.copy(pair['image'],
#                        image_path.joinpath(img_name))
#
#        if not label_path.joinpath(lbl_name).is_file():
#            shutil.copy(pair['seg'],
#                        label_path.joinpath(lbl_name))
               
        
def _get_random_split(pairs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Generates training and validation splits based on the fractions defined in TRAIN and VAL.

    Args:
        pairs (List[Dict]): List of image-segmentation pairs.

    Returns:
        Tuple[List[Dict], List[Dict]]: Tuple of training and validation image-segmentaiton pairs.
    """

    rng = np.random.default_rng(42)
    idxs = range(len(pairs))
    val_idxs = list(rng.choice(idxs, size=np.floor(len(idxs) * VAL).astype(int), replace=False, shuffle=False))
    train_idxs = list(set(idxs) - set(val_idxs))
    
    return [pairs[idx] for idx in train_idxs], [pairs[idx] for idx in val_idxs]
    
    
if __name__ == '__main__':
    
    datasets = [
                ATLAS,
                CT_STROKE,
                HCP,
                ISLES_2022,
                ONDRIFS,
                ONDRIWMH,
                UPENN,
                ## Test sets below
                #ADNI,
                #ISLES_2018,
                #BRATS2020,
               ]
    
    for dataset in tqdm(datasets):
        split_data(dataset['path'], dataset['name'])
        