# Then, use the calibration matrix to find the angle of rotation between two images.
def get_panning_angle(rot_matrix):
    """
    Get the panning angle from the rotation matrix.
    Args:
        rot_matrix (np.array): 3x3 rotation matrix.
    
    Returns:
        float: panning angle in degrees.
    """
    sy = np.linalg.norm(rot_matrix[:2,0])
    return np.arctan2(-rot_matrix[2,0], sy) * 180 / np.pi

def find_angle(img1, img2, camera_matrix):
    """
    Estimates the angle by which the camera panned between two given frames.
    Args:
        img1 (np.array): first frame.
        img2 (np.array): second frame.
        camera_matrix (np.array): Calibrated camera matrix.
    Returns:
        float: panning angle in degrees.
    """

    # Find the keypoints and descriptors.
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    des1, des2 = np.float32(des1), np.float32(des2)
    
    # Match features with FLANN.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    _, rotations, _, _ = cv.decomposeHomographyMat(M, K=camera_matrix)

    return get_panning_angle(rotations[0])