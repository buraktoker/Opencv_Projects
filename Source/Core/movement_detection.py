import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(f"currentdir is {currentdir}")
parentdir = os.path.dirname(currentdir)
print(f"parentdir is {parentdir}")
sys.path.insert(0, parentdir) 

import cv2
import numpy as np

def get_frame_blurry_and_gray(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (3,3), 0)
    return frame
    
USE_WEBCAM = 1
def main():
    """ Main Function"""
    if USE_WEBCAM == 1:
        video_cap = cv2.VideoCapture(0) # using webcam,
        video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        video_cap = cv2.VideoCapture("D:\Python\OpenCV_YOLO\Opencv_Projects\Videos\\test3.mp4")
    previous_frame = None
    while True:
        ret, frame = video_cap.read()
        if ret is False:
            break
        frame = cv2.resize(frame, (600,600))
        if previous_frame is None:
            previous_frame = get_frame_blurry_and_gray(frame)
            continue
        height, width = frame.shape[:2]
        black_image=np.zeros((height,width,3))

        frame_rgb = frame
        frame = get_frame_blurry_and_gray(frame)
        diff_frame = cv2.absdiff(src1=previous_frame, src2=frame)
        kernel = np.ones((3, 3))
        diff_frame = cv2.dilate(diff_frame, kernel, 15)
        diff_frame = cv2.erode(diff_frame, kernel, 10)
        thresh_frame = cv2.threshold(src=diff_frame, thresh=15, maxval=255, type=cv2.THRESH_BINARY)[1]
        print("MEAN OF ThreshFrame", np.mean(thresh_frame))
        print("Shape OF ThreshFrame", np.shape(thresh_frame))
        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        previous_frame = frame
        for contour in contours:
            if cv2.contourArea(contour) < 200:
                # too small: skip!
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img=black_image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.namedWindow("black_image")        # Create a named window
        cv2.namedWindow("diff_frame")        # Create a named window
        cv2.namedWindow("thresh_frame")        # Create a named window
        cv2.moveWindow("diff_frame", 0,0)  # Move it 
        cv2.moveWindow("thresh_frame", 600,0)  # Move it
        cv2.moveWindow("black_image", 1200,0)  # Move it 
        
        cv2.imshow("diff_frame", diff_frame)
        cv2.imshow("thresh_frame", thresh_frame)
        cv2.imshow("black_image", black_image)
        if cv2.waitKey(1) == ord("q"):
            break
    
    

if __name__ == '__main__':
    print("Before Main")
    main()
    print("After Main")