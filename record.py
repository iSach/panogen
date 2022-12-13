import cv2
from argparse import ArgumentParser
from feed_reader import VideoFeedReader
import os

# Parse arguments
parser = ArgumentParser()
# Path to the folder containing the images (offline mode only)
parser.add_argument('-n', '--name', type=str, help='Name for path')
# Max frames
parser.add_argument('-m', '--max', type=int, default=750, help='Max frames')

# Argument values
args = parser.parse_args()
name = args.name
max_frames = args.max

path = 'data/' + name
if not os.path.exists(path):
    os.mkdir(path)
path += '/' + name + '_'

vfr = VideoFeedReader(online=True)

frame_count = 0

while True:
    if max_frames != -1 and frame_count >= max_frames:
        break
    
    # Read frame
    current_frame = vfr.read()
    cv2.imwrite(path + str(frame_count).zfill(4) + '.jpg', current_frame)
    frame_count += 1

    # Display frame
    # Write progress on frame
    time = round(frame_count / 25, 1)
    cv2.putText(current_frame, '{}/{} ({}s)'.format(frame_count, max_frames, time), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
    cv2.imshow('frame', current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vfr.release()

# Destroy all the windows
cv2.destroyAllWindows()
