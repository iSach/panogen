import cv2 as cv2
import numpy as np
import math
import imutils
from anglemeter import compute_homography

sampling_span = 5
condition_treshold = 0.5

# https://github.com/tsherlock/panorama/blob/master/pano_stitcher.py
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

# https://github.com/tsherlock/panorama/blob/master/pano_stitcher.py
def create_mosaic(images, origins):
    """
    Combine multiple images into a mosaic.
    Arguments:
    images: a list of 4-channel images to combine in the mosaic.
    origins: a list of the locations upper-left corner of each image in
    a common frame, e.g. the frame of a central image.
    Returns: a new 4-channel mosaic combining all of the input images. pixels
    in the mosaic not covered by any input image should have their
    alpha channel set to zero.
    """
    # zip origins and images together
    zipped = list(zip(origins, images))
    
    # sort by distance from origin (highest to lowest)
    func = lambda x: math.sqrt(x[0][0] ** 2 + x[0][1] ** 2)
    dist_sorted = sorted(zipped, key=func, reverse=True)
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

    # stitch images into frame by order of distance
    for image in dist_sorted:
        offset_y = image[0][1] + cent_y
        offset_x = image[0][0] + cent_x
        end_y = offset_y + image[1].shape[0]
        end_x = offset_x + image[1].shape[1]

        stitch_cur = stitch[offset_y:end_y, offset_x:end_x, :4]
        stitch_cur[image[1]>0] = image[1][image[1]>0]

    return stitch

def create_panorama(image, previous, direction, cam_matrix):
    try:
        h,w,_ = image.shape
        f = cam_matrix[0,0]
        H = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
        
        image = cylindrical_warp_image(image , H)
        if previous is  None :
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
    except:
        print('Panorama failed.')
        return None