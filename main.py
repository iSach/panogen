import cv2
from motion_detection import MotionDetector
from argparse import ArgumentParser
from feed_reader import VideoFeedReader
from json_writer import JsonWriter
import time
from tqdm import tqdm
from calibration import get_camera_matrix
from anglemeter import find_angle
from panorama import create_panorama
from panorama import crop_pano

# Parse arguments
parser = ArgumentParser()
# Mode: offline (parse images in a folder) or online (read camera stream)
parser.add_argument('-o', '--online', type=lambda x: (str(x).lower() == 'true'), default=True, help='offline or online')
# Path to the folder containing the images (offline mode only)
parser.add_argument('-p', '--path', type=str, default='data/img_5_1/img_5_1_', help='Path prefix of the sequence images')
# Produce a video with the angle overlayed on the frames
# Online mode shows a live feed, offline mode saves the video.
parser.add_argument('-sv', '--savevideo', type=lambda x: (str(x).lower() == 'true'), default=False, 
                    help='produce a video with the angle overlayed on the frames')
parser.add_argument('-dv', '--displayvideo', type=lambda x: (str(x).lower() == 'true'), default=True, 
                    help='display a video with the angle overlayed on the frames')
# Frames between two updates of the angle estimation
parser.add_argument('-s', '--skip', type=int, default=0, help='frames between two updates of the angle estimation')
# Print the angle in the console
parser.add_argument('-d', '--print', type=lambda x: (str(x).lower() == 'true'), default=False, help='print the angle in the console')
# Frame count format: 0037 or 37
parser.add_argument('-f', '--padcount', type=lambda x: (str(x).lower() == 'true'), default=True, help='Pad frame count format: 0037 or 37')
# File format
parser.add_argument('-ff', '--format', type=str, default='jpg', help='File format')
# YOLO model
parser.add_argument('-m', '--model', type=str, default='yolov5n', help='YOLO model')
# Small ball diameter
parser.add_argument('-sd', '--smalldiameter', type=float, default=10.5, help='Small ball diameter')
# Small ball diameter
parser.add_argument('-bd', '--bigdiameter', type=float, default=23.8, help='Big ball diameter')
# Save to JSON?
parser.add_argument('-sj', '--savejson', type=lambda x: (str(x).lower() == 'true'), default=False, help='Save to JSON?')
# Panorama
parser.add_argument('-pan', '--panorama', type=lambda x: (str(x).lower() == 'true'), default=False, help='Panorama?')
# Show bounding boxes and angle on video
parser.add_argument('-sb', '--showbbox', type=lambda x: (str(x).lower() == 'true'), default=True, help='Show bounding boxes and angle on video')
# Begin at frame:
parser.add_argument('-b', '--begin', type=int, default=0, help='Begin at frame')
# Max frames
parser.add_argument('-mf', '--maxframes', type=int, default=-1, help='Max frames')

# Argument values
args = parser.parse_args()
online = args.online
path = args.path
save_video = args.savevideo
display_video = args.displayvideo
frames_per_update = args.skip
print_angle = args.print
save_bbox = args.savejson
pad_count = args.padcount
file_format = args.format
yolo_model = args.model
small_ball_diam = args.smalldiameter
big_ball_diam = args.bigdiameter
save_bbox = args.savejson
show_panorama = args.panorama
show_bbox = args.showbbox
begin_frame = args.begin
max_frames = args.maxframes

# Calibrate camera, no matter the parameters.
cam_matrix = get_camera_matrix()

# set parameters for text drawn on the frames
font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 2
fontColor = (255, 0, 0)
lineType  = 3

# initialise text variables to draw on frames
angle = 'None'
translation = 'None'
motion = 'None'
motion_type = 'None'

vfr = VideoFeedReader(online=online, path=path, pad_count=pad_count, file_format=file_format, begin_frame=begin_frame)
md = MotionDetector(model=yolo_model, big_ball_diam=big_ball_diam, small_ball_diam=small_ball_diam)

if save_bbox:
    jw = JsonWriter('json/' + path.split('/')[-1])

if not display_video:
    pbar = tqdm(total=-1)

if save_video:
    # initialise video writer
    file_name = "results/" + path.split('/')[-1] + '.mp4'
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    outputStream = cv2.VideoWriter(file_name, codec, 25.0, (1280, 720), 1)

start_time = time.time()

previous_frame = vfr.read()
curr_angle = 0
frame_count = 0

previous_pano = None
previoush = 720

ANGLE_DIFF_THRESHOLD = 2
min_angle = -ANGLE_DIFF_THRESHOLD
max_angle = ANGLE_DIFF_THRESHOLD

while True:
    if max_frames != -1 and frame_count >= max_frames:
        break
    
    # Read frame
    current_frame = vfr.read()

    # If the frame is empty, break immediately
    if current_frame is None:
        break
    
    if show_bbox or save_bbox:
        res = md.detect(current_frame, size=640)
        boxes = res['boxes']

    if save_bbox:
        jw.write_frame(frame_count, boxes, md)

    if frame_count % (frames_per_update + 1) == 0:
        curr_angle += find_angle(previous_frame, current_frame, cam_matrix)
        if print_angle:
            print("Angle: {}".format(curr_angle))
    
        # Update the previous frame
        previous_frame = current_frame.copy()

    # 25 is too slow on board, changed to 50 (was 25 on poster)
    if frame_count % 50 == 0:
        # Position of current_frame in panorama
        if previous_pano is not None:
            pp_copy = previous_pano.copy()
            pp_copy = cv2.cvtColor(pp_copy, cv2.COLOR_BGRA2BGR)
            mt = cv2.matchTemplate(pp_copy, current_frame, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mt)
            top_left = max_loc
            h, w, _ = current_frame.shape
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(pp_copy, top_left, bottom_right, (0, 0, 255), 8)
            cv2.imshow('Panorama', pp_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if show_panorama:
        if curr_angle > max_angle + ANGLE_DIFF_THRESHOLD or curr_angle < min_angle - ANGLE_DIFF_THRESHOLD:
            if previous_pano is not None:
                previoush,previousw,_ = previous_pano.shape

            if not show_bbox:
                res = md.detect(current_frame, size=640)
            mask = res['mask'] * 255

            direction = 1 if curr_angle < min_angle - ANGLE_DIFF_THRESHOLD else 0
            mask_inv = cv2.bitwise_not(mask)   
            img_fg = cv2.bitwise_and(current_frame,current_frame,mask =mask_inv)
            panorama = create_panorama(img_fg,previous_pano, direction, cam_matrix)
            if panorama is not None:
                h,w,_ = panorama.shape
                if (h - previoush < 20 or w - previousw < 50):
                    previous_pano = panorama

        
        if curr_angle > max_angle + ANGLE_DIFF_THRESHOLD:
            max_angle = curr_angle

        if curr_angle < min_angle - ANGLE_DIFF_THRESHOLD:
            min_angle = curr_angle

    if display_video:
        if show_bbox:
            person_boxes = boxes[boxes[:, 5] == 0][:, :4]
            color = (0, 0, 255)
            for box in person_boxes:
                cv2.putText(current_frame, 'person', (box[0], box[1] + 16), font, 0.8, color, lineType)
                cv2.rectangle(current_frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            ball_boxes = boxes[boxes[:, 5] == 1][:, :4]
            balls_depths = md.compute_ball_depths(ball_boxes)
            color = (255, 0, 0)
            for i, box in enumerate(ball_boxes):
                is_big, depth = balls_depths[i]
                txt = 'BALL' if is_big else 'ball'
                txt_z = '{}m'.format(round(depth, 2))
                cv2.putText(current_frame, txt_z, (box[0], box[1] + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
                cv2.putText(current_frame, txt, (box[0], box[1] + 16), font, 0.8, color, lineType)
                cv2.rectangle(current_frame, (box[0], box[1]), (box[2], box[3]), color, 2)

        cv2.putText(current_frame, str(round(curr_angle, 3)), (50,90), font, 1.7, fontColor, lineType)
                    
        # Display the resulting frame
        cv2.imshow('Frame', current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        pbar.update(1)
    
    if save_video:
        outputStream.write(current_frame)

    frame_count += 1

if save_video:
    print('Video saved to {}'.format(file_name))
    outputStream.release()

if show_panorama:
    if not online:
        cv2.imwrite('results/panorama_' + path.split('/')[-1] + '.png', previous_pano)
        try:
            pano_cropped = crop_pano(previous_pano)
            cv2.imwrite('results/panorama_cropped_' + path.split('/')[-1] + '.png', pano_cropped)
        except:
            print("Failed to crop panorama")
    else:
        cv2.imwrite('results/panorama.png', previous_pano)
        try:
            pano_cropped = crop_pano(previous_pano)
            cv2.imwrite('results/panorama_cropped.png', pano_cropped)
        except:
            print("Failed to crop panorama")

if save_bbox:
    jw.close()

time_elapsed = time.time() - start_time
print("Finished in {} seconds (avg. FPS: {})".format(time_elapsed, frame_count / time_elapsed))

# After the loop release the cap object
vfr.release()

# Destroy all the windows
cv2.destroyAllWindows()
