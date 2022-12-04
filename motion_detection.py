import torch
import numpy as np
import cv2

MODELS = ('yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')

class MotionDetector:
	def __init__(self, model='yolov5s'):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if model.startswith('yolov5'):
			self.model = torch.hub.load('ultralytics/yolov5', model, pretrained=True).eval().to(self.device)
		elif model.startswith('yolov3'):
			self.model = torch.hub.load('ultralytics/yolov3', model, pretrained=True).eval().to(self.device)
		self.model_type = 'yolo'

		self.__PERSON_CLASS = 0
		self.__BALL_CLASS = 32
	
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
				cv2.circle(mask, center, radius, 1, -1)
			else:
				# Person: draw a rectangle.
				#mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
				# Draw an ellipse in the box with value 1
				cv2.ellipse(mask, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), (int((box[2] - box[0]) / 2), int((box[3] - box[1]) / 2)), 0, 0, 360, 1, -1)

		res[res[:, 5] == self.__BALL_CLASS, 5] = 1
		
		return {'mask': mask, 'boxes': res}