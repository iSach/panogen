import numpy as np
import cv2
import glob
import os

def __calibrate_camera(imgs_folder, debug=False):
    """
    Calibrate the camera using the images in the given folder.
    Args:
        imgs_folder (str): path to the folder containing the images.
        debug (bool): whether to show the number of valid images or not.
    Returns:
        camera_matrix: The calibrated camera Matrix.
    """
    
    # Termination criteria: 30 iterations, or 0.001 accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Object points, with 25mm-sized squares on the checkerboard.
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2) * 25

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(imgs_folder + '/*.jpg')
    valid_count = 0
    for fname in images:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        ret, corners = cv2.findChessboardCorners(img, (6,9), None)
        if ret == True:
            valid_count += 1
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
    if debug:
        print("Calibrated camera with {} valid calibration images.".format(valid_count))
    
    return cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)[1]


def get_camera_matrix():
    """
    Get the camera matrix from the calibration folder.
    Returns:
        camera_matrix: The calibrated camera Matrix.
    """
    # if file exists, load from it, else calibrate and save.
    if os.path.isfile('data/calibration/camera_matrix.npy'):
        return np.load('data/calibration/camera_matrix.npy')
    else:
        camera_matrix = __calibrate_camera('data/calibration')
        np.save('data/calibration/camera_matrix', camera_matrix)
        return camera_matrix