import cv2
import torch
from motion_detection import MotionDetector
from argparse import ArgumentParser
from feed_reader import VideoFeedReader
from json_writer import JsonWriter
import time
from tqdm import tqdm

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
parser.add_argument('-s', '--skip', type=int, default=1, help='frames between two updates of the angle estimation')
# Print the angle in the console
parser.add_argument('-d', '--print', type=lambda x: (str(x).lower() == 'true'), default=False, help='print the angle in the console')

# Argument values
args = parser.parse_args()
online = args.online
path = args.path
save_video = args.savevideo
display_video = args.displayvideo
frames_per_update = args.skip
print_angle = args.print
save_bbox = False


# Calibrate camera, no matter the parameters.
#cam_matrix = calibrate_camera('calibration', debug=True)

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

curr_angle = 0
frame_count = 1
frame_id = 0

if save_video:
    # initialise video writer
    file_name = "results/" + path.split('/')[-1] + '.mp4'
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    outputStream = cv2.VideoWriter(file_name, codec, 25.0, (1280, 720), 0)

vfr = VideoFeedReader(online=online, path=path)
md = MotionDetector(model='yolov5s', big_ball_diam=14.8)
if save_bbox:
    jw = JsonWriter('json/' + path.split('/')[-1])

if not display_video:
    pbar = tqdm(total=-1)

start_time = time.time()
previous_frame = vfr.read()
while True:
    # Read frame
    current_frame = vfr.read()

    # If the frame is empty, break immediately
    if current_frame is None:
        break
    
    res = md.detect(current_frame, size=320)
    boxes = res['boxes']

    if save_bbox:
        jw.write_frame(frame_id, boxes, md)

    if frame_count == frames_per_update: # TODO fix
        #curr_angle += find_angle(previous_frame, current_frame, cam_matrix)
        curr_angle += 0
        if print_angle:
            print("Angle: {}".format(curr_angle))

        frame_count = 0
    
        # Update the previous frame
        previous_frame = current_frame.copy()
    
    if display_video:
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

        cv2.putText(current_frame, str(curr_angle), (50,90), 
                    font, fontScale, fontColor, lineType)
                    
        # Display the resulting frame
        cv2.imshow('Frame', current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        pbar.update(1)
    
    if save_video:
        outputStream.write(current_frame)

    frame_count += 1
    frame_id += 1

if save_video:
    print('Video saved to {}'.format(file_name))
    outputStream.release()

if save_bbox:
    jw.close()

time_elapsed = time.time() - start_time
print("Finished in {} seconds (avg. FPS: {})".format(time_elapsed, frame_id / time_elapsed))

# After the loop release the cap object
vfr.release()

# Destroy all the windows
cv2.destroyAllWindows()