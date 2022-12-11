import os
import cv2
import imutils
import numpy as np
from feed_reader import VideoFeedReader

def crop_pano(output):
    stitched = cv2.copyMakeBorder(output, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT, (0, 0, 0))
    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # allocate memory for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    minRect = mask.copy()
    sub = mask.copy()
    # keep looping until there are no non-zero pixels left in the
    # subtracted image
    while cv2.countNonZero(sub) > 0:
        # erode the minimum rectangular mask and then subtract
        # the thresholded image from the minimum rectangular mask
        # so we can count if there are any non-zero pixels left
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    # find contours in the minimum rectangular mask and then
    # extract the bounding box (x, y)-coordinates
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    # use the bounding box coordinates to extract the our final
    # stitched image
    stitched = stitched[y:y + h, x:x + w]
    # Resize
    stitched = cv2.resize(stitched, (1280, 720))

    return stitched

count = 0

vfr = VideoFeedReader(online=False, path='data/img_5_1/img_5_1_', pad_count=True, file_format='jpg')
stitchy = cv2.Stitcher_create()
previous_pano = None
while True:
    # Read frame
    current_frame = vfr.read()

    # If the frame is empty, break immediately
    if current_frame is None:
        break

    if count % 25 == 0:
        count = 0
        if previous_pano is None:
            previous_pano = current_frame
        else:
            (dummy, output) = stitchy.stitch([previous_pano, current_frame])
            print('Stitching status: %d' % dummy)
            if dummy == cv2.Stitcher_OK:
                output = crop_pano(output)
                previous_pano = output
                
                cv2.imshow('final result', output)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    count += 1