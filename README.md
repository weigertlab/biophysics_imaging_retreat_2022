# biophysics_imaging_retreat_2022

## Agenda:
1. Image Analysis clinic (45 mins)
1. Deep-Learning-based image analysis from scratch in 45 mins
1. (TODO Outcome of time arrow predictions on participant's datasets)

## Resources
- Dataset: Phase contrast microscopy of YEAST cells (TODO download link)
- Napari (python-based image viewer): https://napari.org/tutorials/fundamentals/installation.html
- StarDist https://github.com/stardist/stardist#installation
- StarDist napari plugin https://github.com/stardist/stardist-napari

## Part 1 curriculum
1. Intro + look at the dataset in napari + intro to napari (10 mins, Ben).
1. Explain StarDist (5 mins, Ben).
1. Use pretrained Fluo StarDist model for predictions in napari plugin, show failure (5 mins, Ben).
1. Annotation of one image (Ben) in napari, train a model (Albert) (10 mins).
1. Data augmentations for one image + train a model (10 mins, Albert).
    - show different data augmetation transforms (and ask audience for input).
1. Start training on full data (10 mins, Albert), during that explain
    - train-val-test
    - overfitting
    - early-stopping
1. Segment unannotated images with trained yeast model in napari plugin (5 mins, Ben).

Backup:
- Interactive watershed-based segmentation in Fiji (8 bit, denoise with median, (blur sigma 2), opening 3, invert, interactive watershed).
