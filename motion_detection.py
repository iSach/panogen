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
			self.model = torch.hub.load('ultralytics/yolov5', model, pretrained=True, _verbose=False).eval().to(self.device)
		elif model.startswith('yolov3'):
			self.model = torch.hub.load('ultralytics/yolov3', model, pretrained=True, _verbose=False).eval().to(self.device)
		self.model_type = 'yolo'

		self.__PERSON_CLASS = 0
		self.__BALL_CLASS = 32

		self.ball_mask = ball_mask
		self.person_mask = person_mask

		# Diameter in pixels at 1m for the balls.
		self.small_ball_pxdiam = int(PPM * small_ball_diam)
		self.big_ball_pxdiam = int(PPM * big_ball_diam)

		closest_small_ball = 1.5
		self.px_threshold = self.small_ball_pxdiam / closest_small_ball
	
	def detect(self, frame, size=640):
		"""
		in: frame (OpenCV)

		out: A dictionary with the mask, and the bounding boxes of the detected objects
		"""
		frame = frame[..., ::-1]

		res = self.model(frame, size=size).xyxy[0]
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

	def compute_ball_depths(self, ball_boxes):
		"""
		Compute the depth of the balls in the frame.
		"""

		nb_boxes = len(ball_boxes)

		if nb_boxes == 0:
			return []
		elif nb_boxes == 2:
			box_sizes = ball_boxes[:, 2] - ball_boxes[:, 0]
			ball_types = box_sizes  == torch.max(box_sizes)

			# If the two balls sizes are too close, we cannot differentiate between them.
			if torch.abs(box_sizes[0] - box_sizes[1]) < 10:
				return [self.compute_ball_depth(box) for box in ball_boxes]
			depths = [(ball_types[i], self.__get_ball_depth(ball_boxes[i], ball_types[i])) for i in range(nb_boxes)]
		else:
			return [self.compute_ball_depth(box) for box in ball_boxes]

		return depths

	def compute_ball_depth(self, ball_box):
		"""
		Compute the depth of the ball in the frame.

		return: A tuple (is_big_ball, depth)
		"""

		# Criterion when seeing one ball to differentiate between big and small balls.
		px_diameter = int((ball_box[2] - ball_box[0]))
		if px_diameter < self.px_threshold:
			return False, self.__get_ball_depth(ball_box, False)
		else:
			return True, self.__get_ball_depth(ball_box, True)

	def __get_ball_depth(self, bbox, is_big):
		"""
		Compute the depth of the ball in the frame.
		"""
		px_diameter = int((bbox[2] - bbox[0]))
		if is_big:
			return self.big_ball_pxdiam / px_diameter
		else:
			return self.small_ball_pxdiam / px_diameter
