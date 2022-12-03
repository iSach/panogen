import torch
import numpy as np

class MotionDetector:
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n').to(self.device)

		self.__PERSON_CLASS = 0
		self.__BALL_CLASS = 32
	
	def detect(self, frame):
		"""
		in: frame (OpenCV)

		out: A dictionary with the mask, and the bounding boxes of the detected objects
		"""
		res = self.model(frame).xyxy[0]
		res = res[(res[:, 5] == self.__BALL_CLASS) | (res[:, 5] == self.__PERSON_CLASS)]
		
		# Draw the mask
		mask = np.zeros((720, 1280), dtype=np.uint8)
		for box in res:
			mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1

		# Transform the bounding boxes to a tensor.
		# Remove 5th column in res tensor
		person_boxes = res[res[:, 5] == self.__PERSON_CLASS]
		ball_boxes = res[res[:, 5] == self.__BALL_CLASS]
		res = res[:, :5]
		
		return {'mask': mask, 'person_boxes': person_boxes, 'ball_boxes': ball_boxes}