## Note and change log

This codebase is adapted from [Medical-SAM2](https://github.com/MedicineToken/Medical-SAM2). It extends the original implementation with additional features and optimizations for neuroimaging segmentation tasks.

* The original Medical-SAM2 codebase used only computed the segmentation loss with BCE loss only. Auxiliary losses for IoU and object predictions are added in this codebase to match Meta's released training script.
    * Auxiliary losses:
        * Segmentation loss: Sigmoid Focal Loss + Dice Loss
        * IoU loss: L1 loss
        * Object prediction loss (occlusion loss): Sigmoid Focal Loss
    * Model outputs 3 masks per object along with 3 IoU predictions, the final mask output is the one with the highest IoU prediction.
    * Object prediction handles target object's visibility (e.g. whether the object is occluded in the given frame/slice) 

* Added learning rate schedulers for the SAM2 and memory layers.
    * Used cosine lr scheduler to stay consistent with Meta's released training script (from Meta's `fvcore` library).
    * Pytorch's `CosineAnnealingLR` might be a better alternative as `fvcore`'s lr schedulers are stateless.

* Support direct loading of Nifti files with MONAI transforms preprocessing.
    * The original codebase requires the input images to be saved as 8-bit png images with preprocessing steps (it would eat up a lot of memory and time).
* Adam changed to AdamW to match SAM2's optimizer.
* Distributed training is supported.
* Multi-video batching is supported.

## What could be improved in this codebase

* Add an extra argument to control the number of target objects to use for training to avoid the GPU memory spiking issue (more constant VRAM usage).

* Dynamically adjust the probability of using click vs bbox prompts based on validation performance (this would require more work to implement).
    * So far the click prompt performs worse than the bbox prompt.

* Consider fine-tuning the image encoder's neck (last layer of the image encoder) even if leaving out the image encoder, since the image encoder is pre-trained on 8-bit input range (0-255) while this codebase directly apply MONAI transforms to the input Nifti images, normalizing the input images to the range of [-1, 1]. The shift in data distribution could be significant and affect feature extraction.
    * Improved validation performance observed when fine-tuning with the image encoder's neck.
    * The neck is a Feature Pyramid Network.
    * A more brute force approach: change `b_min` and `b_max` in `ScaleIntensityRangePercentilesd` from `[-1, 1]` to `[0, 1]` to match the image encoder's pre-trained input range. (haven't tried this yet)

* If fine-tuning the image encoder, consider adding another optimizer and lr scheduler for the image encoder.

* Gradient clipping has not been applied.
    * SAM2 used gradient clipping in their released training script with `max_norm=0.1, norm_type=2`.


## Installation (on CCDB cluster)

This codebase now requires torch version >= 2.5.0 rather than the original codebase's torch version 2.4.0. This is because the original codebase's torch version is too old to make MONAI transform work.

1. Create a python environment:

```bash
module load python/3.12
module load scipy-stack opencv cuda/12.2
virtualenv --no-download NeuroSAM2D
source NeuroSAM2D/bin/activate
pip install --no-index --upgrade pip
```

2. Install the latest torch version:

```bash
pip install torch torchvision --no-index --no-cache-dir
```
This installs torch version 2.6.0 with CUDA 12.2 as of the date of this writing (2025-03-31).

3. Install the other dependencies:

```bash
pip install -r requirements.txt
```

4. Install the current codebase:

4.1 Install the codebase:

```bash
pip install -e .
```

4.2 Build CUDA extension on a **GPU node**:

```bash
export MAX_JOBS=8
python setup.py build_ext --inplace
```

## Dataset format

The dataset json file needs to contain at least the following fields:
```json
{
    "training": [
        {
            "image": "path/to/data/DatasetName/imagesTr/image_name.nii.gz",
            "seg": "path/to/data/DatasetName/labelsTr/label_name.nii.gz",
        },
        ...
    ],
    "validation": [
        {
            "image": "path/to/data/DatasetName/imagesVal/image_name.nii.gz",
            "seg": "path/to/data/DatasetName/labelsVal/label_name.nii.gz"
        },
        ...
    ]
}
```
and placed under each dataset folder, e.g. `path/to/data/DatasetName/dataset.json`.



# Original README content
<h1 align="center">● Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2</h1>

<p align="center">
    <a href="https://discord.gg/DN4rvk95CC">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <img src="https://img.shields.io/static/v1?label=license&message=GPL&color=white&style=flat" alt="License"/>
</p>

python setup.py clean --all then python setup.py build_ext --inplace

Medical SAM 2, or say MedSAM-2, is an advanced segmentation model that utilizes the [SAM 2](https://github.com/facebookresearch/segment-anything-2) framework to address both 2D and 3D medical
image segmentation tasks. This method is elaborated on the paper [Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2](https://arxiv.org/abs/2408.00874).

## 🔥 A Quick Overview 
 <div align="center"><img width="880" height="350" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/framework.png"></div>
 
## 🩻 3D Abdomen Segmentation Visualisation
 <div align="center"><img width="420" height="420" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/example.gif"></div>

## 🧐 Requirement

 Install the environment:

 ``pip install -e ".[dev]"``

 You can download SAM2 checkpoint from checkpoints folder:
 
 ``bash download_ckpts.sh``

 Also download the pretrain weights [here.](https://huggingface.co/jiayuanz3/MedSAM2_pretrain/tree/main)


 Further Note: We tested on the following system environment and you may have to handle some issue due to system difference.
```
Operating System: Rocky Linux 8.10
Pip Version: 24.2
Python Version: 3.11.5
```