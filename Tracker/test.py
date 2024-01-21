import cv2
from ultralytics import YOLO
import torch
import time

video_path = "D:/Python/OpenCV_YOLO/Opencv_Projects/Videos/saha/saha_1.mp4"
model_path = "D:/Python/OpenCV_YOLO/Opencv_Projects/Models/best_air_vehicles.pt"
vid = cv2.VideoCapture(video_path)
device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = YOLO(model_path).to(device)

# Example using StrongSORT tracker
tracker = cv2.TrackerKCF_create()  # Replace with your chosen tracker
tracker_is_using = False
xywh_tuple = (0, 0, 0, 0)
frame_counter = 0
start_time = time.time()
while True:
    ret, frame = vid.read()
    if not ret:
        break
    frame_counter += 1
    frame = cv2.resize(frame, None, fx=1/2, fy=1/2)
    if tracker_is_using is False:
        print(f"tracker_is_using is {tracker_is_using}")
        results = model(frame)  # Run YOLOv8 prediction
        result = results[0]
        for bbox in result.boxes.data.tolist():
            print(bbox)
            x1, y1, x2, y2, score, class_id = bbox
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            w = x2 - x1
            h = y2 - y1
            xywh_tuple = (int(x1), int(y1), int(w), int(h))
        # bboxes = results[0].boxes.xyxy[0].tolist()
        # print(bboxes)
        # Initialize tracker for new objects
        print(f"xywh_tuple is {xywh_tuple}")
        tracker = cv2.TrackerMIL_create()
        tracker.init(frame, xywh_tuple)
    ok, boxes = tracker.update(frame)
    if ok:
        tracker_is_using = True
        # Draw bounding boxes
        p1 = (int(boxes[0]), int(boxes[1]))
        p2 = (int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3]))
        # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        tracker_is_using = False
    # cv2.imshow("Tracking", frame)
    # if cv2.waitKey(1) == ord("q"):
    #    break
    end_time = time.time()
    print(f"FPS is {frame_counter/(end_time - start_time)}")

vid.release()
cv2.destroyAllWindows()
