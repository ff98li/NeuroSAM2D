from collections import defaultdict
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from .btcv import BTCV
from .amos import AMOS
from .neurosam import NEUROSAM
from .transforms import get_train_transforms, get_val_transforms
import torch


def multi_video_collate(batch):
    """
    Collate function for multi-video batch
    """
    # Calculate the maximum object ID for each sample
    max_ids = []
    video_length = batch[0]["image"].shape[0]
    for sample in batch:
        current_max = 0
        for d in [sample["label"], sample["bbox"], sample["pt"]]:
            for frame_dict in d.values():
                try:
                    current_max = max(current_max, *(frame_dict.keys()))
                except Exception as e:
                    continue ## there could be empty frame_dict
        max_ids.append(current_max)
    
    # Calculate ID offsets
    offsets, offset = [], 0
    for mid in max_ids:
        offsets.append(offset)
        offset += mid
    
    # Merge images (concatenate along the time dimension)
    images = [s["image"] for s in batch]
    merged_image = torch.cat(images, dim=0) # [(T*B), C, H, W]
    n_frames = merged_image.shape[0]

    # Merge labels and bounding boxes
    #merged_prompt = dict()
    merged_bbox = dict()
    merged_pt = dict()
    merged_label = dict()
    merged_p_label = dict()
    for frame_id in range(n_frames):
        #merged_prompt[frame_id] = dict()
        merged_bbox[frame_id] = dict()
        merged_pt[frame_id] = dict()
        merged_label[frame_id] = dict()
        merged_p_label[frame_id] = dict()
    
    for sample_idx, (sample, offset) in enumerate(zip(batch, offsets)):
        # Process labels
        for frame_id, objs in sample["label"].items():
            frame_id_offset = sample_idx * video_length + frame_id
            for obj_id, tensor in objs.items():
                new_id = obj_id + offset
                merged_label[frame_id_offset][new_id] = tensor.unsqueeze(0)

        for frame_id, objs in sample["bbox"].items():
            frame_id_offset = sample_idx * video_length + frame_id
            for obj_id, array in objs.items():
                new_id = obj_id + offset
                merged_bbox[frame_id_offset][new_id] = torch.from_numpy(array).unsqueeze(0)
        
        for frame_id, objs in sample["pt"].items():
            frame_id_offset = sample_idx * video_length + frame_id
            for obj_id, array in objs.items():
                new_id = obj_id + offset
                merged_pt[frame_id_offset][new_id] = torch.from_numpy(array).unsqueeze(0)
        
        for frame_id, objs in sample["p_label"].items():
            frame_id_offset = sample_idx * video_length + frame_id
            for obj_id, label_int in objs.items():
                new_id = obj_id + offset
                merged_p_label[frame_id_offset][new_id] = torch.tensor(label_int, dtype=torch.int64).unsqueeze(0)
    
    collated_data = {
        "image": merged_image,
        "label": merged_label,
        "bbox": merged_bbox,
        "pt": merged_pt,
        "p_label": merged_p_label,
        "image_meta_dict": {
            'filename_or_obj': [
                sample["image_meta_dict"]["filename_or_obj"]
                for sample in batch
            ]
        }
    }

    return collated_data

def get_dataloader(args):
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''

    elif args.dataset == 'neurosam':
        neurosam_train_dataset = NEUROSAM(
            args, 
            args.data_path, 
            transform = get_train_transforms(args.image_size), 
            mode = 'Training', 
            #prompt=args.prompt
        )
        neurosam_test_dataset = NEUROSAM(
            args, 
            args.data_path, 
            transform = get_val_transforms(args.image_size), 
            mode = 'Validation', 
            #prompt=args.prompt
        )

        if args.batch_size > 1:
            collate_fn = multi_video_collate
        else:
            collate_fn = default_collate

        if args.distributed:
            train_sampler = DistributedSampler(
                neurosam_train_dataset,
                rank = args.global_rank
            )
            test_sampler = DistributedSampler(
                neurosam_test_dataset,
                rank = args.global_rank
            )
            nice_train_loader = DataLoader(
                neurosam_train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                sampler=train_sampler,
                collate_fn=collate_fn
            )
            nice_test_loader = DataLoader(
                neurosam_test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers // 2, pin_memory=True,
                sampler=test_sampler,
                collate_fn=collate_fn
            )
        else:
            nice_train_loader = DataLoader(
                neurosam_train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_workers, 
                pin_memory=True,
                collate_fn=collate_fn
            )
            nice_test_loader = DataLoader(
                neurosam_test_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=args.num_workers // 2,
                pin_memory=True,
                collate_fn=collate_fn
            )
        '''end'''

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader


#def get_dataloader_ddp(args):
#    
#    if args.dataset == 'btcv':
#        '''btcv data'''
#        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
#        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)
#        train_sampler = DistributedSampler(
#            btcv_train_dataset,
#            rank = args.local_rank
#        )
#
#        nice_train_loader = DataLoader(
#            btcv_train_dataset,
#            batch_size=args.batch_size,
#            shuffle=False,
#            num_workers=8,
#            pin_memory=True,
#            sampler=train_sampler
#        )
#        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
#        '''end'''
#    elif args.dataset == 'amos':
#        '''amos data'''
#        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
#        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)
#        train_sampler = DistributedSampler(
#            amos_train_dataset,
#            rank = args.local_rank
#        )
#
#        nice_train_loader = DataLoader(
#            amos_train_dataset,
#            batch_size=args.batch_size,
#            shuffle=False,
#            num_workers=8,
#            pin_memory=True,
#            sampler=train_sampler
#        )
#        nice_test_loader = DataLoader(
#            amos_test_dataset,
#            batch_size=1,
#            shuffle=False,
#            num_workers=1,
#            pin_memory=True
#        )
#        '''end'''
#
#    elif args.dataset == 'neurosam':
#        '''neurosam data'''
#        neurosam_train_dataset = NEUROSAM(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
#        neurosam_test_dataset = NEUROSAM(args, args.data_path, transform = None, transform_msk= None, mode = 'Validation', prompt=args.prompt)
#        train_sampler = DistributedSampler(
#            neurosam_train_dataset,
#            rank = args.global_rank
#        )
#        test_sampler = DistributedSampler(
#            neurosam_test_dataset,
#            rank = args.local_rank
#        )
#
#        nice_train_loader = DataLoader(
#            neurosam_train_dataset,
#            batch_size=1,
#            shuffle=False,
#            num_workers=4,
#            pin_memory=True,
#            sampler=train_sampler
#        )
#        nice_test_loader = DataLoader(
#            neurosam_test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
#            sampler=test_sampler
#        )
#        '''end'''
#
#    elif args.dataset == 'neurosam_npy':
#        '''neurosam data'''
#        from .neurosam_npy import NEUROSAM_NPY
#        neurosam_train_dataset = NEUROSAM_NPY(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
#        neurosam_test_dataset = NEUROSAM_NPY(args, args.data_path, transform = None, transform_msk= None, mode = 'Validation', prompt=args.prompt)
#        train_sampler = DistributedSampler(
#            neurosam_train_dataset,
#            rank = args.global_rank
#        )
#        test_sampler = DistributedSampler(
#            neurosam_test_dataset,
#            rank = args.local_rank
#        )
#
#        nice_train_loader = DataLoader(
#            neurosam_train_dataset,
#            batch_size=1,
#            shuffle=False,
#            num_workers=4,
#            pin_memory=True,
#            sampler=train_sampler
#        )
#        nice_test_loader = DataLoader(
#            neurosam_test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
#            sampler=test_sampler
#        )
#        '''end'''
#
#    elif args.dataset == 'neurosam_nifti':
#        neurosam_train_dataset = NEUROSAM_NIfTI(args, args.data_path, transform = get_train_transforms(args.image_size), mode = 'Training', prompt=args.prompt)
#        neurosam_test_dataset = NEUROSAM_NIfTI(args, args.data_path, transform = get_val_transforms(args.image_size), mode = 'Validation', prompt=args.prompt)
#
#        if args.batch_size > 1:
#            collate_fn = multi_video_collate
#        else:
#            collate_fn = default_collate
#
#        if args.distributed:
#            train_sampler = DistributedSampler(
#                neurosam_train_dataset,
#                rank = args.global_rank
#            )
#            test_sampler = DistributedSampler(
#                neurosam_test_dataset,
#                rank = args.local_rank
#            )
#            nice_train_loader = DataLoader(
#                neurosam_train_dataset,
#                batch_size=args.batch_size,
#                shuffle=False,
#                num_workers=args.num_workers,
#                pin_memory=True,
#                sampler=train_sampler,
#                collate_fn=collate_fn
#            )
#            nice_test_loader = DataLoader(
#                neurosam_test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers // 2, pin_memory=True,
#                sampler=test_sampler,
#                collate_fn=collate_fn
#            )
#        else:
#            nice_train_loader = DataLoader(
#                neurosam_train_dataset, 
#                batch_size=args.batch_size, 
#                shuffle=True, 
#                num_workers=args.num_workers, 
#                pin_memory=True,
#                collate_fn=collate_fn
#            )
#            nice_test_loader = DataLoader(
#                neurosam_test_dataset, 
#                batch_size=1, 
#                shuffle=False, 
#                num_workers=args.num_workers // 2,
#                pin_memory=True,
#                collate_fn=collate_fn
#            )
#    else:
#        print("the dataset is not supported now!!!")
#        
#    return nice_train_loader, nice_test_loader
