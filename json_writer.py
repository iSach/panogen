import json

class JsonWriter(object):
    def __init__(self, path):
        self.path = path
        self.file = open(path + '.json', 'w')
        self.data = []

    def write_frame(self, frame_id, boxes, motion_detector):
        person_boxes = boxes[boxes[:, 5] == 0]
        for box in person_boxes:
            x1 = box[0].item()
            y1 = box[1].item()
            x2 = box[2].item()
            y2 = box[3].item()
            score = box[4].item()
            cat_id = 2 if box[5].item() == 0 else 1
            self.write_person(frame_id, x1, y1, x2, y2, score)

        ball_boxes = boxes[boxes[:, 5] == 1]
        balls_depths = motion_detector.compute_ball_depths(ball_boxes)
        for i, box in enumerate(ball_boxes):
            _, depth = balls_depths[i]
            x1 = box[0].item()
            y1 = box[1].item()
            x2 = box[2].item()
            y2 = box[3].item()
            score = box[4].item()
            cat_id = 2 if box[5].item() == 0 else 1
            self.write_ball(frame_id, x1, y1, x2, y2, depth, score)


    def write_ball(self, id, x1, y1, x2, y2, z, score):
        data = {'image_id': id, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'category_id': 1, 'score': score, 'distance': z}
        self.data.append(data)

    def write_person(self, id, x1, y1, x2, y2, score):
        data = {'image_id': id, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'category_id': 2, 'score': score}
        self.data.append(data)
        
    def close(self):
        json.dump(self.data, self.file, indent=4)