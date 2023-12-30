import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(f"currentdir is {currentdir}")
parentdir = os.path.dirname(currentdir)
print(f"parentdir is {parentdir}")
sys.path.insert(0, parentdir) 

import Utils.utils as utils
import cv2
from ultralytics import YOLO
import datetime
from tracker import Tracker

# This purpose of this source code is traffic tracking with YOLOv8 and deepsort tracking algorithm

#  CONSTANT VALUES
CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)
CUSTOM_TRACKER = 1
WHITE = (255, 255, 255)

def main():
    print("Within main")
    print(utils.get_cuda_info())
    models_path = currentdir + "\..\..\Models"
    
    model = utils.get_model(models_path, "YOLOv8_Detection_custom_best.pt") # found on google
    # link https://colab.research.google.com/drive/1dEpI2k3m1i0vbvB4bNqPRQUO0gSBTz25?usp=sharing#scrollTo=PaXu7rNzNWFw
    tracker = utils.get_tracker(20)
    video_cap = cv2.VideoCapture("D:/Python/OpenCV_YOLO/Opencv_Projects/Videos/test3.mp4")
    while True:
        # start time to compute the fps
        start = datetime.datetime.now()
        ret, frame = video_cap.read()
        # if there are no more frames to process, break out of the loop
        if not ret:
            break
        # resize frame
        # run the YOLO model on the frame
        if CUSTOM_TRACKER is 1:
            results = []    
            detections = model.predict(frame, verbose= False, half = True)[0]
            for data in detections.boxes.data.tolist():
                # extract the confidence (i.e., probability) associated with the detection
                confidence = data[4]

                # filter out weak detections by ensuring the 
                # confidence is greater than the minimum confidence
                if float(confidence) < CONFIDENCE_THRESHOLD:
                    continue

                # if the confidence is greater than the minimum confidence,
                # draw the bounding box on the frame
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                class_id = int(data[5])
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
                tracks = tracker.update_tracks(results, frame=frame)
                for track in tracks:
                    # if the track is not confirmed, ignore it
                    if not track.is_confirmed():
                        continue

                    # get the track id and the bounding box
                    track_id = track.track_id
                    ltrb = track.to_ltrb()

                    xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                        ltrb[1]), int(ltrb[2]), int(ltrb[3])
                    # draw the bounding box and the track id
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                    cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
                    cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                    cv2.imshow("Deepsort Tracking", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break
        else: 
            frame = cv2.resize(frame, (512, 512))
            detections = model.track(frame, verbose= False, imgsz=512)[0]  
            annotated_frame = detections.plot()
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            if cv2.waitKey(1) == ord("q"):
                break
        end = datetime.datetime.now()
        # show the time it took to process 1 frame
        total = (end - start).total_seconds()
        print(f" 1 frame: {total * 1000:.0f} ms")
    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(__package__)
    print("Before Main")
    main()
    print("After Main")