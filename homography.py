import numpy as np
import random
import logging
import time

logger = logging.getLogger(__name__)

def fit_model(p1, p2):
    p1 = np.concatenate((p1, np.ones((p1.shape[0], 1))), axis=1)
    p2 = np.concatenate((p2, np.ones((p2.shape[0], 1))), axis=1)
    model_params = np.linalg.lstsq(p2, p1, rcond=None)[0].T
    return model_params


def RANSAC(set1, set2, N=1000, init_points=5, inlier_threshold=50):
    logging.info('\n')

    # Score between inlier keypoints in image1 and transformed inlier keypoints in image 2.
    ransac_runs = []
    logger.info("Running RANSAC with {} iterations and {} points".format(N, init_points))
    start_time = time.time()
    for n in range(N):
        ransac_iteration = {}
        init_indices = random.sample(range(set1.shape[0]), init_points)
        set1_points, set2_points = set1[init_indices], set2[init_indices]

        ls_affinetransform = fit_model(set1_points, set2_points)
        n_inliers = 0
        inlier_indices = []
        residuals = []
        for idx, (pt1, pt2) in enumerate((zip(set1, set2))):
            if idx in init_indices:
                continue

            pt1 = np.concatenate((pt1, [1]))
            pt2 = np.concatenate((pt2, [1]))

            difference = pt1 - np.dot(ls_affinetransform, pt2)
            distance = np.sqrt(np.square(difference[0]) + np.square(difference[1]))
            if distance <= inlier_threshold:
                n_inliers += 1
                inlier_indices.append(idx)
                
                # Calculating residual for this transformation
                residuals.append([np.square(difference[0]), np.square(difference[1])])
                

        ransac_iteration["H"] = ls_affinetransform
        ransac_iteration["n_inliers"] = n_inliers
        ransac_iteration["n_outliers"] = len(set1) - (n_inliers + init_points)
        ransac_iteration["residuals"] = residuals
        ransac_iteration["inlier_indices"] = inlier_indices


        ransac_runs.append(ransac_iteration)

    end_time = time.time() - start_time

    ransac_runs.sort(key=lambda x: x["n_inliers"])
    best_run = ransac_runs[-1]

    best_points = set1[best_run["inlier_indices"]], set2[best_run["inlier_indices"]]
    best_model = fit_model(*best_points)
    
    # Get averaged residuals for x and y axis
    residuals = np.array(best_run["residuals"])
    average_residuals = np.average(residuals, axis=0)

    logger.info("Inlier Ratio: {}".format(float(best_run["n_inliers"])/best_run["n_outliers"]))
    logger.info("Average residuals for x axis: {}; y axis: {}".format(*average_residuals))
    logger.info("Time taken for RANSAC: {} seconds".format(end_time))
    best_run["H"] = best_model

    return best_run
