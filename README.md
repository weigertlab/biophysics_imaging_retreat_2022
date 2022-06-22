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
- Napari (python-based image viewer): https://napari.org/tutorials/fundamentals/installation.html (Optional: Annotate via _Sidecar_ on iPad, Ben has this working)
- Labkit https://imagej.net/plugins/labkit/
- StarDist jupyter notebooks https://github.com/stardist/stardist/tree/master/examples/2D

## Part 1 curriculum
1. Intro + look at the dataset in napari (5 mins).
1. Use pretrained Fluo StarDist model for predictions in napari plugin, show failure (5 mins).
1. (Rough annotation of one image (ddF8BF_crop_1_10.tif) in napari by participant (30s) + train a model with that (5 mins))
1. Annotation of one image (ddF8BF_crop_1_10.tif) in napari + train a model (10 mins).
1. Data augmentations for one image + train a model (10 mins).
    - show different data augmetation transforms (and ask audience for input).
1. Start training on full data (10 mins), during that
    - StarDist method
    - explain train-val-test
    - overfitting
    - early-stopping

1. Segment unannotated images with jupyter notebook, using a model pretrained on the yeast dataset [3_prediction.ipynb](https://github.com/stardist/stardist/blob/master/examples/2D/3_prediction.ipynb) OR again in napari plugin. (5 mins)

Backup:
- Interactive watershed-based segmentation in Fiji (8 bit, denoise with median, (blur sigma 2), opening 3, invert, interactive watershed).
