import cv2 
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

def get_cuda_info():
    return cv2.getBuildInformation()

def get_model(models_path, model_name):
    model_path_name = models_path + "/" + model_name
    print(f"models_path {models_path} model_name {model_name}")
    print(f"model_path_name {model_path_name}")

    # model = torch.hub.load('.', 'custom', path=model_path_name, source='local')  # or yolov8s, yolov8m, yolov8l, yolov8x, custom

    model = YOLO(model_path_name)
    model.cuda()
    print(model.info(detailed=False))
    model.metrics
    return model

def get_tracker(i_max_age):
    tracker = DeepSort(max_age=i_max_age)
    return tracker
