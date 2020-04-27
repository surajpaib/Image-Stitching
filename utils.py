import cv2
import numpy as np

def array2opencvkp(keypoints):
    keypoints_list = []
    for keypoint in keypoints:
        keypoints_list.append(cv2.KeyPoint(keypoint[1], keypoint[0], 1))

    return keypoints_list
