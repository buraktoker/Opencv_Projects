import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics import NAS
import datetime
import os
import torch


def main():
    print(cv2.__version__)
    print(cv2.getBuildInformation())
    cwd = os.path.dirname(os.path.realpath(__file__))
    print(f"cwd is {cwd}")
    video_path = "D:/Python/OpenCV_YOLO/Opencv_Projects/Videos/saha/saha_1.mp4"
    model_path = "D:/Python/OpenCV_YOLO/Opencv_Projects/Models/best_air_vehicles.pt"
    cap = cv2.VideoCapture(video_path)
    width  = cap.get(3)  # float `width`
    height = cap.get(4)  # float `height`

    # it gives me 0.0 :/
    fps = cap.get(cv2.CAP_PROP_FPS)

    # cap.set(cv2.cudacodec.ColorFormat_BGR)
    print("before tracker create")
    tracker = cv2.TrackerMIL_create() 
    print("after tracker create")
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device is {device}")
    model = YOLO(model_path).to(device)
    print(f"width is {width} height {height} fps {fps}")
    # Display model information (optional)
    # model.info()
    frame_counter = 0
    FRAME_LIMIT = 4
    tracked_frame = 0
    non_tracked_frame = 0
    non_detected_frame = 0
    track_dict = dict()
    track_dict[-1] = 0
    while True:
        # ret, gpu_frame = cap.nextFrame()
        ret, frame = cap.read()
        # Break the loop if the video is over
        if not ret:
            break
        if frame_counter % FRAME_LIMIT == 0:
            frame = cv2.resize(frame, (640, 384))
            results = model.track(frame, persist = True, verbose = False, tracker="bytetrack.yaml")
            result = results[0]
            has_detection = False
            for res in result.boxes.data.tolist():
                has_detection = True
                if len(res) == 7:
                    x1, y1, x2, y2, track_id, score, class_id = res
                    track_id = int(track_id)
                    if track_id in track_dict:
                        track_dict[track_id] += 1
                    else:
                        track_dict[track_id] = 1
                    tracked_frame += 1
                else:
                    x1, y1, x2, y2, score, class_id = res
                    track_id = -1
                    non_tracked_frame += 1
                    track_dict[-1] += 1
            if not has_detection:
                non_detected_frame += 1
                
            cv2.imshow("Frame", result.plot())
            if cv2.waitKey(1) == ord("q"):
                break
        frame_counter += 1
    print(f"tracked_frame {tracked_frame} non_tracked_frame {non_tracked_frame} non_detected_frame {non_detected_frame} frame_counter {frame_counter}")
    print(f"track_dict is {track_dict}")

if __name__ == '__main__':
    print("Tracker Eval before main")
    main()
    print("Tracker Eval after main")