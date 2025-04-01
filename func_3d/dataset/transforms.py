import monai.transforms
import torch

def replace_nan_with_min(image):
    valid_values = image[~torch.isnan(image)]
    invalid_values = image[torch.isnan(image)]
    if len(invalid_values) > 0:
        if len(valid_values) > 0:
            fill_value = torch.min(valid_values)
        else:
            fill_value = 0.0
        image = torch.nan_to_num(image, nan=fill_value)

    return image

def get_train_transforms(img_size: int) -> monai.transforms.Compose:
#def get_train_transforms() -> monai.transforms.Compose:
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),

            monai.transforms.Lambdad(keys=["image"], func=replace_nan_with_min),

            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.05,
                upper=99.95,
                b_min=-1,
                b_max=1,
                clip=True
            ),

            ### monai.transforms.CropForegroundd( # TODO: do we need?
            ###     keys=["image", "label"],
            ###     source_key="image",
            ###     select_fn=lambda x: x > -1,
            ### ),

            monai.transforms.Compose([
                monai.transforms.RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
                monai.transforms.RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
                monai.transforms.RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
                monai.transforms.RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),
                monai.transforms.RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(1, 2)),
                monai.transforms.RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 2)),
            ]),
            monai.transforms.ToTensord(keys=["image", "label"]),
            monai.transforms.OneOf([
                monai.transforms.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
                monai.transforms.RandGaussianNoised(keys=["image"], prob=0.5),
                monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
                monai.transforms.RandHistogramShiftd(keys=["image"], prob=0.5, num_control_points=10),
                monai.transforms.RandGaussianSharpend(keys=["image"], prob=0.5),
            ]),

            ## For 2D models only
            monai.transforms.Resized(
                keys=["image", "label"],
                spatial_size=[img_size, img_size, -1], ## [H, W, D], keeps the video length as is
                mode=["trilinear", "nearest"],
            ),

            #### For NeuroSAM 3D models only
            ###monai.transforms.RandCropByPosNegLabeld(
            ###    keys=["image", "label"],
            ###    label_key="label",
            ###    spatial_size=[img_size, img_size, img_size],
            ###    pos=1,
            ###    neg=0,
            ###    num_samples=1,
            ###    allow_smaller=True,
            ###),
            #### ensure that we have the right shape in the end
            ###monai.transforms.ResizeWithPadOrCropd(
            ###    keys=["image", "label"],
            ###    spatial_size=[img_size, img_size, img_size],
            ###),

            monai.transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )

def get_val_transforms(img_size: int) -> monai.transforms.Compose:
#def get_val_transforms() -> monai.transforms.Compose:
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),

            monai.transforms.Lambdad(keys=["image"], func=replace_nan_with_min),

            monai.transforms.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.05,
                upper=99.95,
                b_min=-1,
                b_max=1,
                clip=True
            ),

            ## For 3D models only
            #monai.transforms.RandCropByPosNegLabeld(
            #    keys=["image", "label"],
            #    label_key="label",
            #    spatial_size=[img_size, img_size, img_size],
            #    pos=1,
            #    neg=0,
            #    num_samples=1,
            #    allow_smaller=True,
            #),

            monai.transforms.ToTensord(keys=["image", "label"]),

            ## For 3D models
            #monai.transforms.ResizeWithPadOrCropd(
            #    keys=["image", "label"],
            #    spatial_size=[img_size, img_size, img_size],
            #),

            ## For 2D models
            monai.transforms.Resized(
                keys=["image", "label"],
                spatial_size=[img_size, img_size, -1], ## [H, W, D], keeps the video length as is
                mode=["trilinear", "nearest"],
            ),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )