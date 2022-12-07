import cv2
from os import path

class VideoFeedReader:
	def __init__(self, online, path=None, in_color=True, begin_frame=0):
		self.online = online
		self.path = path
		self.frame_count = begin_frame
		self.prev_frame = None
		self.in_color = in_color


		if self.online:
			# TODO change
			gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM)," \
            "width=(int)1280, height=(int)720, format=(string)NV12, " \
            "framerate=(fraction)25/1 ! nvvidconv ! video/x-raw, " \
            "width=(int)1280, height=(int)720, format=(string)BGRx ! " \
            "videoconvert ! video/x-raw, format=(string)BGR !" \
            "appsink"
			self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

	def read(self):
		if self.online:
			ret, frame = self.cap.read()
			if self.in_color:
				return frame if ret else None
			return frame if ret else None
		else:
			img_path = self.path + str(self.frame_count).zfill(4) + '.jpg'
			self.frame_count += 1
			if path.exists(img_path):
				return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if not self.in_color else cv2.imread(img_path)
			else:
				return None

	def release(self):
		if self.online:
			self.cap.release()