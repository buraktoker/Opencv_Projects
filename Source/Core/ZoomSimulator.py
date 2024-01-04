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
import Utils.utils as ut

IMSHOW_WIDTH = int(1920*3/4)
IMSHOW_HEIGHT = int(1080*3/4)
USE_IMAGE = 0

def zoom_in_image(image, center, zoom_constant):
    """ For zoom in of one frame"""
    # Center is tuple for zoom center x and y
    # Zoom constant is an index for how zoom to image
    # Zoom constant is how many parts we divide image
    # we all fit to image for imshow (1080, 720)
    zoom_constant_axis = int(zoom_constant/2) # find piece in one constant
    h_o, w_o, _ = image.shape
    x_o = int(w_o/2)
    y_o = int(h_o/2)
    # print(f"Center of original image => x {x_o} y {y_o}")
    h_new = int(h_o / zoom_constant_axis)
    w_new = int(w_o / zoom_constant_axis)
    x_center = center[0]
    y_center = center[1]
    y_start = y_center-int(h_new/2)
    x_start = x_center-int(w_new/2)
    # print(f"h_new {h_new} w_new {w_new}")
    # print(f"x_center {x_center} y_center {y_center}")
    # print(f"x_start {x_start} y_start {y_start}")
    cropped_image = image[y_start:y_start+h_new, x_start:x_start+w_new]
    img_final = cv2.resize(cropped_image, (IMSHOW_WIDTH, IMSHOW_HEIGHT))
    # print("Zoom Proceed")
    return img_final
    

def main():
    """ Main Function"""
    if USE_IMAGE == 1:
        img_path = currentdir + "/../../Images/colors.jpg"
        img = cv2.imread(img_path)
        h_o, w_o, _ = img.shape
        PIECE_NUMBER = 4
        one_height_piece = int(h_o/PIECE_NUMBER)
        one_width_piece = int(w_o/PIECE_NUMBER)
        for i in range(4):
            if i == 0:
                center = (one_width_piece, one_height_piece)
            elif i == 1:
                center = (one_width_piece*3, one_height_piece)
            elif i == 2:
                center = (one_width_piece, one_height_piece*3)
            elif i == 3:
                center = (one_width_piece*3, one_height_piece*3)
            cv2.imshow("img",zoom_in_image(img, center, PIECE_NUMBER))
            cv2.waitKey(0)
    else:
        video_path = currentdir + "/../../Videos/car_green.mp4"
        video_cap = cv2.VideoCapture(video_path)
        PIECE_NUMBER = 4
        while True:
            # start time to compute the fps
            ret, frame = video_cap.read()
            # if there are no more frames to process, break out of the loop
            if not ret:
                break
            h_o, w_o, _ = frame.shape
            one_height_piece = int(h_o/2)
            one_width_piece = int(w_o/2)
            center = (one_width_piece, one_height_piece)
            zoomed_frame = zoom_in_image(frame, center, PIECE_NUMBER)
            h_zoomed, w_zoomed, _ = zoomed_frame.shape
            print(f"Original shape {frame.shape} zoomed shape {zoomed_frame.shape}")
            print(f"center 0 {center[0]} center 1 {center[1]}")
            zoom_name = "zoomed frame"
            original_name = "original frame"
            cv2.namedWindow(zoom_name)        # Create a named window
            cv2.namedWindow(original_name)        # Create a named window
            zoom_window_x = int((w_o-w_zoomed)/2)
            zoom_window_y = int((h_o-h_zoomed)/2)
            print(f"zoom_window_x 0 {zoom_window_x} zoom_window_y {zoom_window_y}")
            #cv2.moveWindow(zoom_name, int(center[0]/PIECE_NUMBER), int(center[1]/PIECE_NUMBER))  # Move it to (40,30)
            cv2.moveWindow(zoom_name, zoom_window_x, zoom_window_y)  # Move it to (40,30)
            cv2.moveWindow(original_name, 0, 0)  # Move it to (40,30)
            cv2.imshow(zoom_name, zoomed_frame)
            cv2.imshow(original_name,frame)
            if cv2.waitKey(1) == ord("q"):
                break
    
    
    
if __name__ == '__main__':
    print("Before Main")
    main()
    print("End of Main")
