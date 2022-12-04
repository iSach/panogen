import cv2
import torch
from motion_detection import MotionDetector
from argparse import ArgumentParser

# Parse arguments
parser = ArgumentParser()
# Mode: offline (parse images in a folder) or online (read camera stream)
parser.add_argument('-o', '--online', type=lambda x: (str(x).lower() == 'true'), default=True, help='offline or online')
# Path to the folder containing the images (offline mode only)
parser.add_argument('-p', '--path', type=str, default='images/ref_5_1_', help='Path prefix of the sequence images')
# Produce a video with the angle overlayed on the frames
# Online mode shows a live feed, offline mode saves the video.
parser.add_argument('-v', '--video', type=lambda x: (str(x).lower() == 'true'), default=True, help='produce a video with the angle overlayed on the frames')
# Frames between two updates of the angle estimation
parser.add_argument('-s', '--skip', type=int, default=15, help='frames between two updates of the angle estimation')
# Print the angle in the console
parser.add_argument('-d', '--print', type=lambda x: (str(x).lower() == 'true'), default=False, help='print the angle in the console')

# Argument values
args = parser.parse_args()
online = args.online
path = args.path
video = args.video
frames_per_update = args.skip
print_angle = args.print

# Calibrate camera, no matter the parameters.
cam_matrix = calibrate_camera('calibration', debug=True)

# set parameters for text drawn on the frames
font = cv.FONT_HERSHEY_COMPLEX
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

if video and not online:
    # initialise video writer
    file_name = "results/" + path.split('/')[-1] + '.mp4'
    codec = cv.VideoWriter_fourcc(*'mp4v')
    outputStream = cv.VideoWriter(file_name, codec, 25.0, (1280, 720), 0)


vfr = VideoFeedReader(online=online, path=path)
previous_frame = vfr.read()
while True:
    # Read frame
    current_frame = vfr.read()

    # If the frame is empty, break immediately
    if current_frame is None:
        break

    if frame_count == frames_per_update:
        curr_angle += find_angle(previous_frame, current_frame, cam_matrix)
        if print_angle:
            print("Angle: {}".format(curr_angle))

        frame_count = 0
    
        # Update the previous frame
        previous_frame = current_frame.copy()
    frame_count += 1
    
    if video:
        cv.putText(current_frame, str(curr_angle), (50,90), 
                    font, fontScale, fontColor, lineType)
        # Display the resulting frame
        cv.imshow('Frame', current_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if not online:
            outputStream.write(current_frame)

if video and not online:
    print('Video saved to {}'.format(file_name))
    outputStream.release()

# After the loop release the cap object
vfr.release()

# Destroy all the windows
cv.destroyAllWindows()