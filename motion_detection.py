import torch
import numpy as np

MODELS = ('yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')

class MotionDetector:
	def __init__(self, model='yolov5s'):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = torch.hub.load('ultralytics/yolov5', model, verbose=False, skip_validation=True).to(self.device)
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
		
		# Draw the mask
		mask = np.zeros((720, 1280), dtype=np.uint8)
		for box in res:
			mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1

		res[res[:, 5] == self.__BALL_CLASS, 5] = 1
		
		return {'mask': mask, 'boxes': res}