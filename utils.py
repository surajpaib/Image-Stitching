import cv2
import numpy as np
import pandas as pd
import os

def array2opencvkp(keypoints):
    """
    keypoints: numpy array

    Convert keypoints to opencv keypoint format for easier drawing and visualization
    """
    keypoints_list = [cv2.KeyPoint(keypoint[1], keypoint[0], 1) for keypoint in keypoints]
    return keypoints_list

def compute_euclidean_distance(p1, p2, H):
    """
    Compute euclidean distance for sensitivity analysis
    """
    p2 = np.concatenate((p2, np.ones((p2.shape[0], 1))), axis=1)
    p1 = np.concatenate((p1, np.ones((p1.shape[0], 1))), axis=1)

    # Predicted p1 values using the best transformation
    pred_p1 = np.dot(H, p2.T).T
    # Euclidean distance between original and predicted gives sensitivity score
    sensitivity = np.linalg.norm(p1 - pred_p1)
    return sensitivity


def save_experiment(params):
    """
    Save experiment parameters and results to a CSV file
    """
    filename = params["results_file"] + ".csv"

    del params["no_gui"]
    del params["results_file"]


    df = pd.json_normalize(params)

    if not(os.path.isfile(filename)):
        df.to_csv(filename)
    else:
        df.to_csv(filename, mode='a', header=False)
    


def gui_display(log_dict):
    """
    Helpers for visualization of images through the stitching process
    """
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
    """
    Helper to upload data to weights and biases project.
    Citation:
    Weights and Biases,
    https://app.wandb.ai/surajpai/image-stitching/reports/Image-Stitching-Report--Vmlldzo5NzAyNg
    https://wandb.com
    """
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
        "Sensitivity": log_dict["sensitivity"],
        "left_image_keypoint_size": len(log_dict["keypoint1"]),
        "right_image_keypoint_size": len(log_dict["keypoint2"]),

    })


def post_process(image):
    """
    Ugly postprocessing for now!
    """
    del_row_idx = []
    del_col_idx = []

    for row in range(image.shape[0]):
        if np.all(image[row]==0):
            del_row_idx.append(row)

    for col in range(image.shape[1]):
        if np.all(image[:, col]==0):
            del_col_idx.append(col)

    image = np.delete(image, del_row_idx, 0)
    image = np.delete(image, del_col_idx, 1)

    return image