import json

class JsonWriter(object):
    def __init__(self, path):
        self.path = path
        self.file = open(path + '.json', 'w')
        self.data = []

    def write_frame(self, frame_id, boxes, motion_detector):
        """
        Writes a frame to the json file

        :param frame_id: the id of the frame
        :param boxes: the bounding boxes of the frame
        :param motion_detector: the motion detector

        :return: None
        """
        person_boxes = boxes[boxes[:, 5] == 0]
        for box in person_boxes:
            x1 = int(box[0].item())
            y1 = int(box[1].item())
            x2 = int(box[2].item())
            y2 = int(box[3].item())
            score = box[4].item()
            self.write_person(frame_id, x1, y1, x2, y2, score)

        ball_boxes = boxes[boxes[:, 5] == 1]
        balls_depths = motion_detector.compute_ball_depths(ball_boxes)
        for i, box in enumerate(ball_boxes):
            _, depth = balls_depths[i]
            x1 = int(box[0].item())
            y1 = int(box[1].item())
            x2 = int(box[2].item())
            y2 = int(box[3].item())
            score = box[4].item()
            self.write_ball(frame_id, x1, y1, x2, y2, depth, score)


    def write_ball(self, id, x1, y1, x2, y2, z, score):
        """
        Writes a ball to the json file
        """
        data = {'image_id': id, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'category_id': 1, 'score': score, 'distance': z}
        self.data.append(data)

    def write_person(self, id, x1, y1, x2, y2, score):
        """
        Writes a person to the json file
        """
        data = {'image_id': id, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'category_id': 2, 'score': score}
        self.data.append(data)
        
    def close(self):
        """
        Saves the data to the json file
        """
        json.dump(self.data, self.file, indent=4)