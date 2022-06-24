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
#     display_name: Python [conda env:py39] *
#     language: python
#     name: conda-env-py39-py
# ---

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # 0. Install StarDist as described [here]( https://github.com/stardist/stardist#installation) (and some dependencies)

# %%
import stardist
# !pip install jupytext
# !pip install git+https://github.com/stardist/augmend.git
import augmend
from augmend import (
    Augmend,
    FlipRot90,
    AdditiveNoise,
    Elastic,
    Rotate,
    IntensityScaleShift,
    Scale,
    Identity,
)
# !pip install gputools pandas seaborn

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # 1. Load packages

# %%
from __future__ import print_function, unicode_literals, absolute_import, division
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = "none"
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
matplotlib.rcParams['figure.figsize'] = (12, 5)
import pandas as pd
import tensorflow as tf
from glob import glob
import seaborn as sns 
from tqdm.notebook import tqdm
from tifffile import imread
from csbdeep.utils import normalize
from csbdeep.utils import Path

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
from stardist.plot import render_label

from skimage.transform import resize
np.random.seed(42)
lbl_cmap = random_label_cmap()

# %% [markdown]
# ### Download data (images and masks, uncomment and only do this once!)

# %%
# # !wget https://drive.switch.ch/index.php/s/F8wpskqkkyVfnm0/download -O data.zip 
# # !unzip data.zip

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# # 2. Load and prepare annotated image

# %%
base_path = Path("yeastnet")


# %% [markdown]
# Utility functions

# %%
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    al.imshow(render_label(lbl, img=.3*img, normalize_img=False, cmap=lbl_cmap))
    al.set_title(lbl_title)
    plt.tight_layout()
    
def preprocess(X, Y, axis_norm=(0,1)):
    # normalize channels independently

    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()
    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X, leave=False, desc="Normalize images")]
    Y = [fill_label_holes(y) for y in tqdm(Y, leave=False, desc="Fill holes in labels")]
    return X, Y


# %%
img = imread(base_path / 'train/images/yeastnet_img_010.tif')
lbl = imread(base_path / 'train/masks/yeastnet_010_mask.tif')
n_channel = 1

# %% [markdown]
# Normalize images using intensity percentiles and fill small label holes.

# %%
img, lbl = preprocess([img], [lbl])
img, lbl = img[0], lbl[0]
plt.hist(img.flatten(), bins=100)
plt.title("Histogram of normalized image")
plt.show();

# %%
plot_img_label(img, lbl)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # 3. Load entire dataset
# We assume that data has already been split into disjoint training, validation and test sets.

# %%
# TODO more realistic here: Annotate a second image by hand, use that as validation?
num_imgs = 0
data = {}
for split in ["train", "val", "test"]:
    X = sorted((base_path / split / "images").glob("*.tif"))
    X = [imread(x) for x in X]
    Y = sorted((base_path / split / "masks").glob("*.tif"))
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
print(f"Number of channels: {n_channel}")

# %%
X_trn, Y_trn = preprocess(X_trn, Y_trn)
X_val, Y_val = preprocess(X_val, Y_val)
X_test, Y_test = preprocess(X_test, Y_test)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# # 4. Training with a single annotated image

# %% [markdown]
# We're going to annotate two images: one will be used to train the model, while the other will be used for validation purposes.

# %% [markdown] tags=[]
# ## Training setup

# %%
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = True and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays = n_rays,
    grid = grid,
    use_gpu = use_gpu,
    n_channel_in = n_channel,
    train_learning_rate = 0.0005,
    train_patch_size = (384,384),
    train_epochs=30, 
)

# Print config to see default values
vars(conf)

# %%
if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(.9, total_memory=16000)

# %%
model = StarDist2D(conf, name='single_image', basedir='models')

# %% [markdown]
# Check if the neural network has a large enough field of view to see up to the boundary of most objects.

# %%
median_size = calculate_extents(list([lbl]), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

# %%
df = pd.DataFrame()
def add_history(df0, hist, name):
    #convert history dict to dataframe and add/concatenate it to an existing dataframe 
    keys = tuple(c.split('val_')[1] for c in hist.keys() if c.startswith('val_'))
    res_train = dict(tuple((k,hist[k]) for k in keys) + (('phase', 'train'),))
    res_val = dict(tuple((k,hist[f'val_{k}']) for k in keys) + (('phase', 'val'),))
    for res in (res_train, res_val):
        res['name'] = name
        res['step'] = np.arange(len(hist['loss']))
    df = pd.concat((pd.DataFrame(res_train), pd.DataFrame(res_val)))
    df = pd.concat((df0, df)).reset_index(drop=True)
    return df    


# %% tags=[]
# # %%capture
X_val_single, Y_val_single = [X_val[5]], [Y_val[5]]
log = model.train([img], [lbl], 
                  validation_data=(X_val_single, Y_val_single), 
                  steps_per_epoch=20, 
                  seed=42)
df = add_history(df, log.history, 'single image')
model.optimize_thresholds(X_val_single, Y_val_single)

# %%
sns.lineplot(data=df, x='step', y='loss', hue = 'name', style='phase')

# %%
pred = model.predict_instances(img, n_tiles=model._guess_n_tiles(img), show_tile_progress=False)[0]
plot_img_label(img, lbl, img_title="Training image", lbl_title="Annotations")
plot_img_label(img, pred, img_title="Training image", lbl_title="Predicted segmentation")

# %% [markdown]
# ### Prediction on unseen test image (single image model)

# %%
idx_test = 3
Y_pred_single, _  = model.predict_instances(X_test[idx_test])

# %%
plot_img_label(X_test[idx_test],Y_test[idx_test], lbl_title="GT")
plot_img_label(X_test[idx_test], Y_pred_single, img_title=f"Test image", lbl_title="Predicted segmentation")

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# # 5. More training data, part 1: Augmentations

# %% [markdown]
# Original image

# %%
plot_img_label(img, lbl, img_title="Original", lbl_title="Original annotation")

# %% [markdown]
# Transformed image.
#
# Comment/Uncomment individual lines to apply different random transformations. Documentation at https://github.com/stardist/augmend.

# %%
transform = Augmend()
transform.add([IntensityScaleShift(scale=(0.75, 1.33), shift=(-0.33, 0.33)), Identity()])
transform.add([AdditiveNoise(), Identity()])
transform.add([Elastic(amount=3, grid=5, order=1, use_gpu=False), Elastic(amount=3, grid=5, order=0, use_gpu=False)])
# transform.add([Rotate(order=1, mode="constant"), Rotate(order=0, mode="constant")])
# transform.add([Elastic(amount=32, order=1), Elastic(amount=32, order=0)])
# transform.add([Scale(order=1), Scale(order=0)])


img_aug, lbl_aug = transform([X_trn[15], Y_trn[15]])
plot_img_label(img_aug, lbl_aug, img_title="Transformed image", lbl_title="Transformed label")


# %%
def build_augmenter(use_gpu):
    axis= (0, 1)
    aug = Augmend()
    aug.add([Elastic(amount=5, grid=11, use_gpu=False, order=1, axis=axis), Elastic(amount=5, grid=11, use_gpu=False, order=0, axis=axis)])
    aug.add([Rotate(order=1, mode="constant"), Rotate(order=0, mode="constant")])
    aug.add([IntensityScaleShift(scale=(0.75, 1.33), shift=(-0.1, 0.1)), Identity()])
    aug.add([Scale(order=1, amount=(.8,1.25), mode="constant"), Scale(order=0, amount=(.8,1.25), mode="constant")])
    aug.add([AdditiveNoise(sigma=0.1), Identity()])

    return aug

augmenter = build_augmenter(use_gpu=False)
augmenter_fun = lambda x, y: augmenter((x, y))

# %%
# plot some augmented examples
plot_img_label(img, lbl)
for _ in range(3):
    img_aug, lbl_aug = augmenter_fun(img,lbl)
    plot_img_label(img_aug, lbl_aug, img_title="Image augmented", lbl_title="Label augmented")

# %%
model_aug = StarDist2D(conf, name='single_image_augmented_test', basedir='models')

# %% tags=[]
# # %%capture
log_aug = model_aug.train(
    [img],
    [lbl],
    validation_data=(X_val_single, Y_val_single),
    augmenter=augmenter_fun,
    steps_per_epoch=20,
    seed=42
)
df = add_history(df, log_aug.history, 'single image w augmentation')
model_aug.optimize_thresholds(X_val_single, Y_val_single)

# %%
sns.lineplot(data=df, x='step', y='loss', hue = 'name', style='phase')

# %%
pred_aug = model_aug.predict_instances(img, n_tiles=model_aug._guess_n_tiles(img), show_tile_progress=False)[0]
plot_img_label(img, lbl, img_title="Training image", lbl_title="Annotations")
plot_img_label(img, pred_aug, img_title="Training image", lbl_title="Predicted segmentation")

# %% [markdown]
# ### Prediction on unseen test image (single image model with augmentation)

# %%
Y_pred_aug, _  = model_aug.predict_instances(X_test[idx_test])

plot_img_label(X_test[idx_test],Y_test[idx_test], lbl_title="GT")
plot_img_label(X_test[idx_test], Y_pred_aug, img_title=f"Test image", lbl_title="Predicted segmentation")

# %% [markdown] tags=[]
# # 6. More training data, part 2: Annotate more by hand!

# %% tags=[]
model_full = StarDist2D(conf, name='full_dataset', basedir='models')
history_regular = model_full.train(X_trn, Y_trn, 
                                   validation_data=(X_val,Y_val), 
                                   augmenter=augmenter_fun, 
                                   steps_per_epoch=50)

df = add_history(df, history_regular.history, 'full')
model_full.optimize_thresholds(X_val, Y_val)

# %%
sns.lineplot(data=df, x='step', y='loss', hue = 'name', style='phase')

# %%
# %%capture
Y_trn_pred = [model_full.predict_instances(x, n_tiles=model_full._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm([img])]

# %%
plot_img_label(img, lbl, lbl_title="label GT")
plot_img_label(img,Y_trn_pred[0], lbl_title="label Pred")

# %%
# %%capture
Y_val_pred = [model_full.predict_instances(x, n_tiles=model_full._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]

# %% [markdown]
# ### Prediction on unseen test image (Full model)

# %%
Y_pred_full, _  = model_full.predict_instances(X_test[idx_test])

plot_img_label(X_test[idx_test],Y_test[idx_test], lbl_title="GT")
plot_img_label(X_test[idx_test], Y_pred_full, img_title=f"Test image", lbl_title="Predicted segmentation")

# %%
idx = 3
plot_img_label(X_val[idx],Y_val[idx], lbl_title="label GT")
plot_img_label(X_val[idx],Y_val_pred[idx], lbl_title="label Pred")
