import torch
import numpy as np
import cv2

MODELS = ('yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
# Number of pixels per meter at 1 meter depth.
PPM = 9.52

class MotionDetector:
	def __init__(self, model='yolov5n', 
					   ball_mask='disk', 
					   person_mask='ellipse',
					   small_ball_diam=10.5,
					   big_ball_diam=23.8):
		"""
		model: The model to use. Can be 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'.
		ball_mask: The mask to use for the balls. Can be 'disk' or 'square'.
		person_mask: The mask to use for the persons. Can be 'ellipse' or 'rectangle'.
		"""

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if model.startswith('yolov5'):
			self.model = torch.hub.load('ultralytics/yolov5', model, pretrained=True).eval().to(self.device)
		elif model.startswith('yolov3'):
			self.model = torch.hub.load('ultralytics/yolov3', model, pretrained=True).eval().to(self.device)
		self.model_type = 'yolo'

		self.__PERSON_CLASS = 0
		self.__BALL_CLASS = 32

		self.ball_mask = ball_mask
		self.person_mask = person_mask

		# Diameter in pixels at 1m for the balls.
		self.small_ball_pxdiam = int(PPM * small_ball_diam)
		self.big_ball_pxdiam = int(PPM * big_ball_diam)

		closest_small_ball = 2
		self.px_threshold = self.small_ball_pxdiam / closest_small_ball
	
	def detect(self, frame):
		"""
		in: frame (OpenCV)

		out: A dictionary with the mask, and the bounding boxes of the detected objects
		"""
		res = self.model(frame).xyxy[0]
		res = res[(res[:, 5] == self.__BALL_CLASS) | (res[:, 5] == self.__PERSON_CLASS)]

		balls_center = []
		
		# Draw the mask
		mask = np.zeros((720, 1280), dtype=np.uint8)
		for box in res:
			c = int(box[5])
			if c == self.__BALL_CLASS:
				# Draw a disk
				# Center = middle of bounding box
				center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
				# Radius = half of the bounding box
				radius = int((box[2] - box[0]) / 2)
				balls_center.append(center)
				if self.ball_mask == 'disk':
					cv2.circle(mask, center, radius, 1, -1)
				elif self.ball_mask == 'square':
					mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
			else:
				# Person: draw a rectangle.
				if self.person_mask == 'rectangle':
					mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
				elif self.person_mask == 'ellipse':
					cv2.ellipse(mask, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), (int((box[2] - box[0]) / 2), int((box[3] - box[1]) / 2)), 0, 0, 360, 1, -1)

		res[res[:, 5] == self.__BALL_CLASS, 5] = 1
		
		return {'mask': mask, 'boxes': res}

	def compute_ball_depths(self, frame, ball_boxes):
		"""
		Compute the depth of the balls in the frame.
		"""

		depths = []
		for box in ball_boxes:
			depths.append(self.compute_ball_depth(frame, box))
		return depths

	def compute_ball_depth(self, frame, ball_box):
		"""
		Compute the depth of the ball in the frame.
		"""

		# Bounding box: x1, y1, x2, y2, score, class
		x1, y1, x2, y2 = ball_box[:4]
		x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
		px_diameter = int((x2 - x1))  # Diameter in pixels.
		# Compute the pixels per meter
		if px_diameter < self.px_threshold:
			return self.small_ball_pxdiam / px_diameter
		else:
			return self.big_ball_pxdiam / px_diameter


md = MotionDetector()