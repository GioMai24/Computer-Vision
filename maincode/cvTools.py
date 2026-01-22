"""Functions to quickly interact with OpenCV images."""
import cv2 as cv
import numpy as np

def show(images, win_names=['window']):
    """
    Quick way to cv.imshow images.

    Parameters
    ----------
    images : list of ndarray
        The images to be shown, preferably read with cv.imread().
    win_name : list of str
        Name of the windows to create (one for each image).
    """
    if type(images) != list or type(win_names) != list: raise ValueError('Unfortunately images and win_names must be lists...')
    for name in win_names: cv.namedWindow(name, cv.WINDOW_NORMAL)    
    while True:
        for i, name in enumerate(win_names): cv.imshow(name, images[i])
        if cv.waitKey(20) & 0xFF == 27: break
    cv.destroyAllWindows()



def still(x):
    """Do you really need help with this? Pass."""
    pass



def threshing(img_path, c_space, low=None, high=None, thresh=None):
    """
    Show two windows to play with color space parameters' thresholding: the first one displays the original image, the second one the image masked by the trackbars' values.
    Press ESC to exit the windows, and the function.

    Parameters
    ----------
    img_path : str
        The path to the image to open.
    c_space : str
        The color space to explore. Must be 'LAB' or 'HSV'.
    low : tuple, optional
        3-elements tuple to set the intial trackbar minimum values.
    high : tuple, optional
        3-elements tuple to set the initial trackar maximum values.
    thresh : dict, optional
        Dictionary to save the output to.

    Returns
    -------
    low : tuple
        Last minimum values of the trackbars.
    high : tuple
        Last maximum values of the trackbars.
    thresh : dict, only if thresh is passed
        Updated dictionary.
    """
    if c_space not in ('LAB', 'HSV'): raise ValueError("c_space must be 'LAB' or 'HSV'!")
    img = cv.imread(img_path)
    win_name = img_path[5:-4]
    cv.namedWindow('Original - ' + win_name, cv.WINDOW_NORMAL)
    cv.namedWindow((touch := 'Masked - ' + win_name), cv.WINDOW_NORMAL)
    for i, c in enumerate(c_space):
        M = 179 if c == 'H' else 255
        cv.createTrackbar(c + ' - Low', touch, 0, M, still)
        cv.createTrackbar(c + ' - High', touch, 0, M, still)
        if low is not None: cv.setTrackbarPos(c + ' - Low', touch, low[i])
        if high is not None: cv.setTrackbarPos(c + ' - High', touch, high[i])
        
    while True:
        low = tuple(cv.getTrackbarPos(c + ' - Low', touch) for c in c_space)
        high = tuple(cv.getTrackbarPos(c + ' - High', touch) for c in c_space)
        res = cv.cvtColor(img, cv.COLOR_BGR2LAB if c_space == 'LAB' else cv.COLOR_BGR2HSV)
        mask = cv.inRange(res, low, high)
        res = cv.bitwise_and(res, res, mask=mask)
        res = cv.cvtColor(res, cv.COLOR_LAB2BGR if c_space == 'LAB' else cv.COLOR_HSV2BGR)
        cv.imshow('Original - ' + win_name, img)
        cv.imshow(touch, res)
        if cv.waitKey(20) & 0xFF == 27: break
    cv.destroyAllWindows()
    if thresh is not None:
        thresh[img_path] = {}
        thresh[img_path]['LOW'] = low
        thresh[img_path]['HIGH'] = high
        return low, high, thresh
    return low, high
    