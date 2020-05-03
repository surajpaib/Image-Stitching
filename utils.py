import cv2
import numpy as np

def array2opencvkp(keypoints):
    keypoints_list = []
    for keypoint in keypoints:
        keypoints_list.append(cv2.KeyPoint(keypoint[1], keypoint[0], 1))

    return keypoints_list

def compute_euclidean_distance(p1, p2, H):
    p2 = np.concatenate((p2, np.ones((p2.shape[0], 1))), axis=1)
    p1 = np.concatenate((p1, np.ones((p1.shape[0], 1))), axis=1)
    pred_p1 = np.dot(H, p2.T).T
    sensitivity = np.linalg.norm(p1 - pred_p1)
    print("Score: ", sensitivity)     