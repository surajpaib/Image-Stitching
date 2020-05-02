import cv2
import numpy as np

def array2opencvkp(keypoints):
    keypoints_list = []
    for keypoint in keypoints:
        keypoints_list.append(cv2.KeyPoint(keypoint[1], keypoint[0], 1))

    return keypoints_list

# def compute_euclidean_distance(matches, keypoints, H):
    
#     for i, match in enumerate(matches):
#         kp1 = keypoints[0][match.queryIdx].pt
#         kp2 = keypoints[1][match.trainIdx].pt

#         predicted_kp2 = 
