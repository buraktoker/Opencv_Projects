import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(f"currentdir is {currentdir}")
parentdir = os.path.dirname(currentdir)
print(f"parentdir is {parentdir}")
sys.path.insert(0, parentdir) 

import numpy as np
import Utils.utils as utils
import cv2
from ultralytics import YOLO
import datetime
from tracker import Tracker

# YOUTUBE LINK https://www.youtube.com/watch?v=jIRRuGN0j5E

def main():
    print("Within main")
    print(utils.get_cuda_info())
    # font info
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    
    video_cap = cv2.VideoCapture("D:/Python/OpenCV_YOLO/Opencv_Projects/Videos/test3.mp4")
    models_path = currentdir + "\..\..\Models"
    
    model = utils.get_model(models_path, "yolov8n.pt") # found on google
    # link https://colab.research.google.com/drive/1dEpI2k3m1i0vbvB4bNqPRQUO0gSBTz25?usp=sharing#scrollTo=PaXu7rNzNWFw
    tracker = Tracker()
    while True:
        # start time to compute the fps
        start = datetime.datetime.now()
        ret, frame = video_cap.read()
        # if there are no more frames to process, break out of the loop
        if not ret:
            break
        results = model(frame, classes=[2,3,4]) # if we have two frame we can have 2 element results
        result = results[0]
        # detections on each frame
        detections = []
        for r in result.boxes.data.tolist():
            print(r)
            x1, x2, y1, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            detections.append([x1, x2, y1, y2, score])
        tracker.update(frame, detections)
        
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1 , x2, y2 = bbox
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            track_id = track.track_id
            # print(f"x1 {x1} x2 {x2} y1 {y1} y2 {y2}")
            cv2.rectangle(frame , (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
            cv2.putText(frame, str(track_id), (x1, y1),
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()