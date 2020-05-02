import cv2
import numpy as np

def array2opencvkp(keypoints):
    keypoints_list = []
    for keypoint in keypoints:
        keypoints_list.append(cv2.KeyPoint(keypoint[1], keypoint[0], 1))

    return keypoints_list

def compute_euclidean_distance(matches, keypoints, H):
    sensitivity_score = 0
    for i, match in enumerate(matches):
        kp1 = keypoints[0][match.queryIdx].pt
        kp2 = keypoints[1][match.trainIdx].pt

        kp2 = np.concatenate((kp2, [1]))
        kp1 = np.concatenate((kp1, [1]))

        predicted_kp1 = np.dot(H, kp2)

        print("KP1", kp1)
        print("Pred KP1", predicted_kp1)

        sensitivity_score += np.linalg.norm(kp1 - predicted_kp1)

    sensitivity_score /= len(matches)
    print("Score:", sensitivity_score)