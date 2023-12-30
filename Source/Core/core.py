# from .Utils import utils
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

def main():
    print("Within main")
    print(utils.get_cuda_info())
    
if __name__ == '__main__':
    print(__package__)
    print("Before Main")
    main()
    print("After Main")