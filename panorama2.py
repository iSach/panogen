from statistics import mode     #some imports
from acquisition import VideoFeedReader
import time
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import glob

sampling_span = 5
condition_treshold = 0.5


# Parse arguments
parser = argparse.ArgumentParser()
# Mode: offline (parse images in a folder) or online (read camera stream)
parser.add_argument('-o', '--online', type=lambda x: (str(x).lower() == 'true'), default=True, help='offline or online')
# Path to the folder containing the images (offline mode only)
parser.add_argument('-p', '--path', type=str, default='images/imgs_', help='Path prefix of the sequence images')
# duration of the online video
parser.add_argument('-d', '--duration', type=int, default='20', help='Number of second of making online panorama')
# number of image to process
parser.add_argument('-nb', '--number', type=int, default='300', help='Number of images to process')



# Argument values
args = parser.parse_args()
online = args.online
path = args.path
duration = args.duration
nb = args.number



def compute_homography(image1, image2, bff_match=False):

    sift = cv2.SIFT_create(edgeThreshold=10, sigma=1.5, contrastThreshold=0.08)
    
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # Brute force matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

    # Lowes Ratio
    good_matches = []
    for m, n in matches:
        if m.distance < .75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])\
        .reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])\
        .reshape(-1, 1, 2)

    if len(src_pts) > 4:
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
    else:
        H = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    return H


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

def cylindrical_warp_image(img, H):
    
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
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

def create_mosaic(images, origins):
    # find central image
    for i in range(0, len(origins)):
        if origins[i] == (0, 0):
            central_index = i
            break

    central_image = images[central_index]
    central_origin = origins[central_index]
    
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

    # print "height ", total_height
    # print "width ", total_width

    # new frame of panorama
    stitch = np.zeros((total_height, total_width, 4), np.uint8)

    # stitch images into frame by order of distance
    for image in dist_sorted:
        
        offset_y = image[0][1] + cent_y
        offset_x = image[0][0] + cent_x
        end_y = offset_y + image[1].shape[0]
        end_x = offset_x + image[1].shape[1]
        
        ####
        stitch_cur = stitch[offset_y:end_y, offset_x:end_x, :4]
        stitch_cur[image[1]>0] = image[1][image[1]>0]
        ####
                
        #stitch[offset_y:end_y, offset_x:end_x, :4] = image[1]

    return stitch

def create_panorama(images, center):
    
    h,w,_ = images[0].shape
    f = 1000 # 800
    H = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    for i in range(len(images)):
        images[i] = cylindrical_warp_image(images[i], H)
    
    panorama = None
    for i in range(center):
        print('Stitching images {}, {}'.format(i+1, i+2))
        image_warped, image_origin = warp_image(images[i], compute_homography(images[i + 1], images[i]))
        panorama = create_mosaic([image_warped, images[i+1]], [image_origin, (0,0)])
        images[i + 1] = panorama

    #print('Done left part')

    for i in range(center, len(images)-1):
        print('Stitching images {}, {}'.format(i+1, i+2))
        image_warped, image_origin = warp_image(images[i+1], compute_homography(images[i], images[i + 1]))
        panorama = create_mosaic([images[i], image_warped], [(0,0), image_origin])
        images[i + 1] = panorama

    #print('Done right part')
    return panorama

images = [ cv2.cvtColor(VideoFeedReader(0, img, in_color=True).read(), cv2.COLOR_RGB2RGBA) for img in glob.glob(path + '000*.jpg')]

center = len(images) // 2
#print(len(images), center)

panorama = create_panorama(images, center)


cv2.imwrite('test.jpg', panorama)
