import cv2
import numpy as np
import pandas as pd
import os

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
    return sensitivity


def save_experiment(params):
    filename = params["results_file"]

    del params["no_gui"]
    del params["results_file"]


    df = pd.json_normalize(params)

    if not(os.path.isfile(filename)):
        df.to_csv(filename)
    else:
        df.to_csv(filename, mode='a', header=False)
    


def gui_display(log_dict):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    drawleft_image = cv2.drawKeypoints(log_dict["left_image"], log_dict["keypoint1"], None)

    cv2.imshow('Image', drawleft_image)
    cv2.waitKey(0)

    drawright_image = cv2.drawKeypoints(log_dict["right_image"], log_dict["keypoint2"], None)

    cv2.imshow('Image', drawright_image)
    cv2.waitKey(0)

    matched_image = cv2.drawMatches(log_dict["left_image"], log_dict["keypoint1"], log_dict["right_image"], log_dict["keypoint2"], log_dict["matches"], None, (255, 0, 0), (255, 0, 0), flags=2)
    cv2.imshow('Image', matched_image)
    cv2.waitKey(0)

    matched_image = cv2.drawMatches(log_dict["left_image"], log_dict["keypoint1"], log_dict["right_image"], log_dict["keypoint2"], log_dict["inlier_matches"], matched_image, (0, 255, 0), (0, 255, 0), flags=2)
    cv2.imshow('Image', matched_image)
    cv2.waitKey(0)

    cv2.imshow('Image', log_dict["stitchedImage"])
    cv2.waitKey(0)


def wandb_log(log_dict, params):
    import wandb
    wandb.init(entity="surajpai", project="image-stitching", config=params)

    drawleft_image = cv2.drawKeypoints(log_dict["left_image"], log_dict["keypoint1"], None)
    drawright_image = cv2.drawKeypoints(log_dict["right_image"], log_dict["keypoint2"], None)
    matched_image = cv2.drawMatches(log_dict["left_image"], log_dict["keypoint1"], log_dict["right_image"], log_dict["keypoint2"], log_dict["matches"], None, (0, 0, 255), (0, 0, 255), flags=2)

    inlier_matched_image = cv2.drawMatches(log_dict["left_image"], log_dict["keypoint1"], log_dict["right_image"], log_dict["keypoint2"], log_dict["inlier_matches"], None, (0, 255, 0), (0, 255, 0), flags=2)


    drawleft_image = cv2.cvtColor(drawleft_image, cv2.COLOR_BGR2RGB)
    drawright_image = cv2.cvtColor(drawright_image, cv2.COLOR_BGR2RGB)
    matched_image = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    inlier_matched_image = cv2.cvtColor(inlier_matched_image, cv2.COLOR_BGR2RGB)
    log_dict["stitchedImage"] = cv2.cvtColor(log_dict["stitchedImage"], cv2.COLOR_BGR2RGB)

    wandb.log({
        "left_image_keypoints": wandb.Image(drawleft_image),
        "right_image_keypoints": wandb.Image(drawright_image),
        "all_matches": wandb.Image(matched_image),
        "inlier_matches": wandb.Image(inlier_matched_image),
        "final_stitched_image": wandb.Image(log_dict["stitchedImage"]),
        "Sensitivity": log_dict["sensitivity"]
    })