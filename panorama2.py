import cv2 as cv2
import numpy as np
import imutils

sampling_span = 5
condition_treshold = 0.5


def compute_homography(image1, image2):
    """
    Detect and match features between image1 and image2
    Using ORB as the detector
    and homography to find objects
    Returns: perspective transformation between two planes
    """
    # Initiate ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    des1, des2 = np.float32(des1), np.float32(des2)
    
    # Match features with FLANN.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    if len(src_pts) < 4 or len(dst_pts) < 4:
        return 0

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M


def warp_image(image, H):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w, _ = image.shape

    # Find min and max x, y of new image
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(H, p)

    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    ymin = min(yrow)
    xmin = min(xrow)
    ymax = max(yrow)
    xmax = max(xrow)

    # Create a new matrix that removes offset and multiply by homography
    new_mat = np.array([[1, 0, -1 * xmin], [0, 1, -1 * ymin], [0, 0, 1]])
    H = np.dot(new_mat, H)

    # height and width of new image frame
    height = int(round(ymax - ymin))
    width = int(round(xmax - xmin))
    size = (width, height)
    # Do the warp
    warped = cv2.warpPerspective(src=image, M=H, dsize=size)

    return warped, (int(xmin), int(ymin))

# https://pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
def crop_pano(output):
    stitched = cv2.copyMakeBorder(output, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    minRect = mask.copy()
    sub = mask.copy()
    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    stitched = stitched[y:y + h, x:x + w]
    stitched = cv2.resize(stitched, (1280, 720))

    return stitched


# https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b
def cylindrical_warp_image(img, H):
    """
    returns the cylindrical warp for a given image and intrinsics matrix H
    
    """
    h, w = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h, w))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h*w, 3) # to homog
    Hinv = np.linalg.inv(H) 
    X = Hinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w*h, 3)
    B = H.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w) | (B[:,1] < 0) | (B[:,1] >= h)] = -1
    B = B.reshape(h,w,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def create_mosaic(images, origins):
    
    # zip origins and images together
    zipped = list(zip(origins, images))
    
    # sort by x value
    x_sorted = sorted(zipped, key=lambda x: x[0][0])
    # sort by y value
    y_sorted = sorted(zipped, key=lambda x: x[0][1])

    # determine the coordinates in the new frame of the central image
    if x_sorted[0][0][0] > 0:
        cent_x = 0  # leftmost image is central image
    else:
        cent_x = abs(x_sorted[0][0][0])

    if y_sorted[0][0][1] > 0:
        cent_y = 0  # topmost image is central image
    else:
        cent_y = abs(y_sorted[0][0][1])

    # make a new list of the starting points in new frame of each image
    spots = []
    for origin in origins:
        spots.append((origin[0]+cent_x, origin[1] + cent_y))

    zipped = zip(spots, images)

    # get height and width of new frame
    total_height = 0
    total_width = 0

    for spot, image in zipped:
        total_width = max(total_width, spot[0]+image.shape[1])
        total_height = max(total_height, spot[1]+image.shape[0])

    # new frame of panorama
    stitch = np.zeros((total_height, total_width, 4), np.uint8)

    return stitch

def create_panorama(image,previous,direction):
    
    h,w,_ = image.shape
    f = 1000
    H = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    
    image = cylindrical_warp_image(image , H)
    if previous is None:
        return image

    if direction == 1:
        M = compute_homography(previous, image)
        if M is None:
            return previous
        image_warped, image_origin = warp_image(previous, M)
        
        return create_mosaic([image_warped, image], [image_origin, (0,0)])
    else : 
        M = compute_homography(image, previous)
        if M is None:
            return previous
        image_warped, image_origin = warp_image(image, M)
        
        return create_mosaic([image_warped, previous], [image_origin, (0,0)])


def pano(img_fg,previous,direction,pos):
    try:
        return create_panorama(img_fg,previous,direction,pos)
    except:
        print('Panorama failed.')
        return None