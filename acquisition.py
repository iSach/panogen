import cv2 as cv
from os import path

class VideoFeedReader:
	def __init__(self, online, path=None, in_color=False, begin_frame=0):
		self.online = online
		self.path = path
		self.frame_count = begin_frame
		self.prev_frame = None
		self.in_color = in_color


		if self.online:
			self.cap = cv.VideoCapture(0)

	def read(self):
		if self.online:
			ret, frame = self.cap.read()
			if self.in_color:
				return frame if ret else None
			return cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if ret else None
		else:
			img_path = self.path + str(self.frame_count).zfill(4) + '.jpg'
			self.frame_count += 1
			if path.exists(img_path):
				return cv.imread(img_path, cv.IMREAD_GRAYSCALE) if not self.in_color else cv.imread(img_path)
			else:
				return None

	def release(self):
		if self.online:
			self.cap.release()