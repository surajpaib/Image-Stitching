import cv2


def array2opencvkp(keypoints):
    keypoints_list = []
    for keypoint in keypoints:
        keypoints_list.append(cv2.KeyPoint(keypoint[0], keypoint[1], 1))

    return keypoints_list