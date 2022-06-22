# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: tf
#     language: python
#     name: tf
# ---

# %%
from __future__ import print_function, unicode_literals, absolute_import, division
import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = "none"
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

import augmend

np.random.seed(42)
lbl_cmap = random_label_cmap()

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # Data
#
# We assume that data has already been split into disjoint training, validation and test sets.  
#
# <div class="alert alert-block alert-info">
# Training data (for input `X` with associated label masks `Y`) can be provided via lists of numpy arrays, where each image can have a different size. Alternatively, a single numpy array can also be used if all images have the same size.  
# Input images can either be two-dimensional (single-channel) or three-dimensional (multi-channel) arrays, where the channel axis comes last. Label images need to be integer-valued.
# </div>

# %%
num_imgs = 0
data = {}
for split in ["train", "val", "test"]:
    X = sorted(glob(f'{os.getenv("HOME")}/EPFL/data/yeast/split/{split}/images/*.tif'))
    X = [imread(x) for x in X]
    Y = sorted(glob(f'{os.getenv("HOME")}/EPFL/data/yeast/split/{split}/masks/*.tif'))
    Y = [imread(y) for y in Y]
    data[split] = (X.copy(), Y.copy())
    num_imgs += len(X)
X_trn, Y_trn = data["train"]
X_val, Y_val = data["val"]
X_test, Y_test = data["test"]
print('number of images: %3d' % num_imgs)
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))
print('- test:     %3d' % len(X_test))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

# %% [markdown]
# Normalize images using intensity percentiles and fill small label holes.

# %%
axis_norm = (0,1)   # normalize channels independently

if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X_trn = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_trn)]
Y_trn = [fill_label_holes(y) for y in tqdm(Y_trn)]

X_val = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_val)]
Y_val = [fill_label_holes(y) for y in tqdm(Y_val)]

X_test = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_test)]
Y_test = [fill_label_holes(y) for y in tqdm(Y_test)]


# %% [markdown]
# Training data consists of pairs of input image and label instances.

# %%
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()


# %%
i = min(9, len(X)-1)
img, lbl = X_trn[i], Y_trn[i]
assert img.ndim in (2,3)
img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
plot_img_label(img,lbl)
None;

# %% [markdown] tags=[]
# # General training setup

# %%
n_rays = 48

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = True and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel
)
print(conf)

# %%
if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(1, total_memory=16000)

# %%
img50_custom_mask = imread(f"{os.getenv('HOME')}/EPFL/data/yeast/split/train/custom_masks/clnF10BF_crop_1_10_ad.tif")

# %% [markdown] tags=[]
# # Training using few annotations

# %%
i = 165
img, lbl = X_trn[i], Y_trn[i]
assert img.ndim in (2,3)
img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
plot_img_label(img,lbl)
None;

# %%
X_trn_single, Y_trn_single = [X_trn[i]], [Y_trn[i]]

# %%
model = StarDist2D(conf, name='stardist_single_annotation_gt_cpu', basedir='models')

# %% [markdown]
# Check if the neural network has a large enough field of view to see up to the boundary of most objects.

# %%
median_size = calculate_extents(list([Y_trn_single]), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

# %% tags=[]
history = model.train(X_trn_single, Y_trn_single, validation_data=(X_val, Y_val), steps_per_epoch=10, seed=42, epochs=100)

# %%
plt.figure(figsize=(12,6))
plt.plot(range(100), history.history["loss"], label="Train")
plt.plot(range(100), history.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training on single image")
plt.legend()
plt.show();

# %%
model.optimize_thresholds(X_trn_single, Y_trn_single)

# %%
Y_trn_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_trn_single)]

# %%
plot_img_label(X_trn_single[0],Y_trn_single[0], lbl_title="label GT")
plot_img_label(X_trn_single[0],Y_trn_pred[0], lbl_title="label Pred")

# %%
# %%capture
Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]

# %%
idx = 22
plot_img_label(X_val[idx],Y_val[idx], lbl_title="label GT")
plot_img_label(X_val[idx],Y_val_pred[idx], lbl_title="label Pred")


# %% [markdown] tags=[]
# # Data Augmentation

# %% [markdown]
# You can define a function/callable that applies augmentation to each batch of the data generator.  
# We here use an `augmenter` that applies random rotations, flips, intensity, noise addition and elastic transformations, which are typically sensible for (2D) microscopy images.

# %%
def build_augmenter(use_gpu):
    augment_probability = 1
    aug = augmend.Augmend()
    axes = (0, 1)
    aug.add([augmend.AdditiveNoise(sigma=0.1), augmend.Identity()], probability=augment_probability)
    aug.add([augmend.Elastic(axis=axes, amount=4, grid=5, use_gpu=use_gpu), augmend.Elastic(axis=axes, amount=4, grid=5, use_gpu=use_gpu, order=0)], probability=augment_probability)
    aug.add([augmend.FlipRot90(axis=axes), augmend.FlipRot90(axis=axes)], probability=augment_probability)
    aug.add([augmend.Rotate(axis=axes), augmend.Rotate(axis=axes, order=0)], probability=augment_probability)
    aug.add([augmend.IntensityScaleShift(scale=(0.9, 1.1), shift=(-0.05, 0.05), axis=axes), augmend.Identity()], probability=augment_probability)
    return aug


# %%
augmenter = build_augmenter(use_gpu=True)
augmenter_fun = lambda x, y: augmenter((x, y))
# plot some augmented examples
img, lbl = X_trn[0], Y_trn[0]
plot_img_label(img, lbl)
for _ in range(3):
    img_aug, lbl_aug = augmenter_fun(img,lbl)
    plot_img_label(img_aug, lbl_aug, img_title="image augmented", lbl_title="label augmented")

# %%
model_aug = StarDist2D(conf, name='stardist_single_annotation_gt_augmented', basedir='models')

# %%
augmend_obj = build_augmenter(use_gpu=True)

# %% tags=[]
history_aug = model_aug.train(X_trn_single, Y_trn_single, validation_data=(X_val, Y_val), steps_per_epoch=10, seed=42, epochs=100, augmenter=augmenter_fun)

# %%
plt.figure(figsize=(12,6))
plt.plot(range(100), history_aug.history["loss"], label="Train")
plt.plot(range(100), history_aug.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Single-image w/ augmentation training pipeline")
plt.legend()
plt.show();

# %%
# %%capture
Y_trn_pred = [model_aug.predict_instances(x, n_tiles=model_aug._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_trn_single)]

# %%
plot_img_label(X_trn_single[0],Y_trn_single[0], lbl_title="label GT")
plot_img_label(X_trn_single[0],Y_trn_pred[0], lbl_title="label Pred")

# %%
# %%capture
Y_val_pred = [model_aug.predict_instances(x, n_tiles=model_aug._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]

# %%
idx = 22
plot_img_label(X_val[idx],Y_val[idx], lbl_title="label GT")
plot_img_label(X_val[idx],Y_val_pred[idx], lbl_title="label Pred")

# %% [markdown] tags=[]
# # Regular training pipeline

# %% tags=[]
model_full = StarDist2D(conf, name='stardist_regular_v2', basedir='models')
history_regular = model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter_fun, epochs=100, steps_per_epoch=np.ceil(len(X_trn)/conf.train_batch_size))

# %%
plt.figure(figsize=(12,6))
plt.plot(range(100), history_regular.history["loss"], label="Train")
plt.plot(range(100), history_regular.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Regular training pipeline")
plt.legend()
plt.show();

# %%
# %%capture
model_full.optimize_thresholds(X_val, Y_val)

# %%
# %%capture
Y_trn_pred = [model_full.predict_instances(x, n_tiles=model_full._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_trn_single)]

# %%
plot_img_label(X_trn_single[0],Y_trn_single[0], lbl_title="label GT")
plot_img_label(X_trn_single[0],Y_trn_pred[0], lbl_title="label Pred")

# %%
# %%capture
Y_val_pred = [model_full.predict_instances(x, n_tiles=model_full._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]

# %%
plot_img_label(X_val[0],Y_val[0], lbl_title="label GT")
plot_img_label(X_val[0],Y_val_pred[0], lbl_title="label Pred")

# %%
