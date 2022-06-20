# biophysics_imaging_retreat_2022

## Agenda:
1. Deep-Learning-based image analysis from scratch in less than one hour
1. Image Analysis brainstorming:
    1. Manley group: TODO
    2. Martin group: TODO
    3. Aumeier group: TODO
1. (Outcome of time arrow predictions on participant's datasets)

## Material
- Dataset candidates:
    - Phase contrast microscopy of YEAST cells (TODO download link)
- ImageJ/Fiji https://imagej.net/software/fiji/downloads
- Napari (python-based image viewer): https://napari.org/tutorials/fundamentals/installation.html
- Labkit https://imagej.net/plugins/labkit/
- StarDist jupyter notebooks https://github.com/stardist/stardist/tree/master/examples/2D

## Part 1 curriculum
1. Option A: Load data into Labkit and annotate + export tif files
    - Downside: Not made for dense annotations, flood filling tool only works on single 2d images.
- Upside: Comparison with Random Forest classifier. Do we want to show that?
1. Option B: Annotate in Napari. Flood filling needed for dense ground truth annotations works in 2d even when working on images stacks. 
1. (Install StarDist)
1. Inspect data & annotations with jupyter notebook [1_data.ipynb](https://github.com/stardist/stardist/blob/master/examples/2D/1_data.ipynb).
1. Train a segmentation model with jupyter notebook [2_training.ipynb](https://github.com/stardist/stardist/blob/master/examples/2D/2_training.ipynb)
    1. Overfit
    1. etc TODO
1. Segment unannotated images with jupyter notebook [3_prediction.ipynb](https://github.com/stardist/stardist/blob/master/examples/2D/3_prediction.ipynb)

