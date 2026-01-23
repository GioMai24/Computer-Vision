# <p align=center> Files Description </p>

## maincode folder
This folder contains the main code of the project. Inside there are four files:
- `__init__.py`: empty file to initialize the python package, so that the modules are callable from outside.
- `cvTools.py`: module containing a couple of functions to quickly show images with OpenCV.
- `extractor.py`: this is the core of the project. The script is used to process the images contained in custom folders (see `file_trimmer.py`). It contains both the segmentation and the feature extraction pipelines.
- `file_trimmer.py`: this script was used to remove unused images from the dataset. It also separates the images based on their known classification, even though it is not necessary for the feature extraction, nor for the classification algorithm.

## Notebooks
- `classification.ipynb`: since the classification algorithm is a simple implementation of the `RandomForestClassifier` class of scikit-learn, I have decided not to create a script for the task, but to have a notebook so as to run the cells more easily while changing hyperparameters values.
- `segmentation.ipynb`: this notebook contains testing code, and it is not meant to be shown. Nonetheless I have included it for completeness.

## data folder
This folder contains:
- `features.csv`: output of `extractor.py`. In contains a table of features for each image.
- `ISIC_2019_GroundTruth.csv`: the list of used images with their classification (whole Melanoma and Melanocytic Nevus classes from the original dataset).

## Conda environment
`cv.yml` contains the conda environment used throughout the project. Below are listed the main packages to install to let the code run.
- python=3.13
- numpy
- matplotlib
- opencv=4.x
- pandas
- tqdm
- scikit-learn
