#!/usr/bin/env python
"""Main processing script to segment images and extract features."""
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def thresholder(lab):
    """
    Apply threshold on H to remove colored artifacts, on grayscale to remove silhouette, and finally Otsu's thresholding.

    Parameters
    ----------
    lab : ndarray
        The image to be filtered, to be passed in LAB space.

    Returns
    -------
    otsu : array
        Binary output mask.
    lab : ndarray
        Original cropped image.
    """
    ## H thresh
    bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    temp = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    _, thresh = cv.threshold(temp[..., 0], 23, 255, cv.THRESH_BINARY_INV)
    _, thresh2 = cv.threshold(temp[..., 0], 110, 255, cv.THRESH_BINARY)
    thresh = cv.bitwise_or(thresh, thresh2)

    ## silhouette
    temp = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    lens = np.zeros_like(temp)
    cv.thresholdWithMask(temp, lens, thresh, 10, 255, cv.THRESH_BINARY)  # lens is the new mask of active pxs
    rows = [cv.hasNonZero(lens[i]) for i in range(lens.shape[0])]
    cols = [cv.hasNonZero(lens[:, i]) for i in range(lens.shape[1])]
    # additional cropping to break possible otsu rings (see report)
    lens = (lens[rows][:, cols])[40:-40, 40:-40].copy()
    lab = (lab[rows][:, cols])[40:-40, 40:-40, :].copy()
    
    ## otsu
    otsu = np.zeros_like(lab[..., 0])
    cv.thresholdWithMask(lab[..., 0], otsu, lens, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return otsu, lab



def counter(otsu, min_area):
    """
    Find and filter patches on a binary mask depending on minimum area, and centroid position.

    Parameters
    ----------
    otsu : array
        Binary mask to filter.
    min_area : int
        Minimum area of the patches to keep.

    Returns
    -------
    conts : list
        Full list of found contours.
    contSave : list
        Indices of contours to keep.
    """
    conts, _ = cv.findContours(otsu, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contSave = []
    for i, cont in enumerate(conts):
        mom = cv.moments(cont)
        if mom['m00'] < min_area: continue  # Area filter
        cx = int(mom['m10'] / mom['m00'])  # centroid x position
        lim = otsu.shape[1] * .1
        if cx < lim or cx > otsu.shape[1] - lim: continue
        cy = int(mom['m01'] / mom['m00'])  # centroid y position
        lim = otsu.shape[0] * .1
        if cy < lim or cy > otsu.shape[0] - lim: continue
        contSave.append(i)
    return conts, contSave



def main(img_path, clahe, sift):
    """
    Segmentation pipeline and feature extraction on a single image.

    Parameters
    ----------
    img_path : path-like object
        Path to the image to process.
    clahe : cv2.CLAHE object
        Initialized CLAHE to be applied.
    sift : cv2.SIFT object
        Initialized SIFT to be applied.

    Returns
    -------
    height : int
        Height of the final image.
    width : int
        Width of the final image.
    symms : list of floats
        Vertical, horizontal and central symmetry score.
    means : list of floats
        Means of the LAB channels.
    stds : list of floats
        Standard deviations of the LAB channels.
    feat_num : int
        Number of detected features by SIFT.
    """
    img = cv.imread(img_path)[1:-1, 1:-1, :]  # some photos have white borders
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    
    ### trimming hair
    lab = cv.morphologyEx(lab, cv.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8), iterations=2)
    
    ### CLAHE
    lab[..., 0] = clahe.apply(lab[..., 0])
    
    ### thresholding
    otsu, lab = thresholder(lab)

    ### kill small patches
    conts, contSave = counter(otsu, 1000)
    if len(contSave) == 0: contSave = counter(otsu, 100)
    if len(contSave) != 0:
        otsu = np.zeros_like(otsu)
        for cont in contSave: cv.drawContours(otsu, conts, cont, 255, -1)
        rows = [cv.hasNonZero(otsu[i]) for i in range(otsu.shape[0])]
        cols = [cv.hasNonZero(otsu[:,i]) for i in range(otsu.shape[1])]
        lab = (lab[rows][:, cols]).copy()
        otsu = (otsu[rows][:, cols]).copy()
    lab = cv.bitwise_and(lab, lab, mask=otsu)
    
    ### Filtering
    lab = cv.bilateralFilter(lab, d=30, sigmaColor=20, sigmaSpace=10)

    ### SIFT
    kp = sift.detect(lab[..., 0], otsu)
    
    ### symmetries
    active = cv.countNonZero(otsu)
    vert = cv.flip(otsu, 0)
    hor = cv.flip(otsu, 1)
    center = cv.flip(hor, 0)
    
    ### output features vector
    height = otsu.shape[0]
    width = otsu.shape[1]
    means, stds = cv.meanStdDev(lab, mask=otsu)
    symms = [cv.countNonZero(cv.bitwise_and(otsu, i)) / active for i in (vert, hor, center)]
    feat_num = len(kp)
    return height, width, symms, means, stds, feat_num 



if __name__ == '__main__':
    data_path = os.path.expanduser('~/uni/Computer-Vision/data/')
    clahe = cv.createCLAHE(clipLimit=1, tileGridSize=(10,10))
    sift = cv.SIFT_create()
    
    image = []
    heights = []
    widths = []
    ver_symm = []
    hor_symm = []
    cent_symm = []
    L_m = []
    L_s = []
    A_m = []
    A_s = []
    B_m = []
    B_s = []
    feat_nums = []
    
    for lesion in ('MEL/', 'NV/'):
        print(f'Processing {lesion}...')
        with os.scandir(data_path + lesion) as Dir:
            for entry in tqdm(Dir):
                height, width, symms, means, stds, feat_num = main(entry.path, clahe, sift)
                image.append(entry.name[:-4])
                heights.append(height)
                widths.append(width)
                for symm, res in zip(symms, (ver_symm, hor_symm, cent_symm)): res.append(symm)
                for i, channel in enumerate((L_m, A_m, B_m)): channel.append(means[i].item())
                for i, channel in enumerate((L_s, A_s, B_s)): channel.append(stds[i].item())
                feat_nums.append(feat_num)
                
    pd.DataFrame({
        'L_m':L_m,
        'L_s':L_s,
        'A_m':A_m,
        'A_s':A_s,
        'B_m':B_m,
        'B_s':B_s,
        'height':heights,
        'width':widths,
        'ver_symm':ver_symm,
        'hor_symm':hor_symm,
        'cent_symm':cent_symm,
        'feat_num':feat_nums
    }, index=image).to_csv(data_path + 'features.csv', index_label='image')
