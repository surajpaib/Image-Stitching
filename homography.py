import numpy as np
import random
import logging
import time

logger = logging.getLogger(__name__)
eps = 0.00001


def fit_model(p1, p2):
    """
    Fit least squares on p1 and p2 arrays.
    The p1 and p2 arrays are of the shape - number_of_points x 2 
    The function returns the least squares fit transformation matrix - 3x3 matrix
    """

    # Append ones to both the arrays to have shape - number_of_points x 3
    p1 = np.concatenate((p1, np.ones((p1.shape[0], 1))), axis=1)
    p2 = np.concatenate((p2, np.ones((p2.shape[0], 1))), axis=1)

    # Least squares fit. model_params is a 3x3 matrix. 
    model_params = np.linalg.lstsq(p2, p1, rcond=None)[0].T

    return model_params


def RANSAC(set1, set2, N=1000, init_points=5, inlier_threshold=50):
    """
    Given two sets of points, calculate best estimated transformation between set1 and set2. 

    Parameters:
    N: Number of iterations to run RANSAC for
    init_points: Number of initial points to estimate transformation
    inlier_threshold: The distance to discriminate between an inlier and an outlier

    """
    logging.info('\n')

    # Score between inlier keypoints in image1 and transformed inlier keypoints in image 2.
    ransac_runs = []
    logger.info("Running RANSAC with {} iterations and {} points".format(N, init_points))
    start_time = time.time()
    
    
    # Iterate over N times 
    for n in range(N):
        ransac_iteration = {}

        # Sample n points based on init_points to start the estimation
        init_indices = random.sample(range(set1.shape[0]), init_points)

        # Get subset of points from these drawn samples
        set1_points, set2_points = set1[init_indices], set2[init_indices]

        # Get least squares fit for the subset of points
        ls_affinetransform = fit_model(set1_points, set2_points)

        n_inliers = 0
        inlier_indices = []
        residuals = []

        # Iterate over all other points to evaluate the least squares fit
        for idx, (pt1, pt2) in enumerate((zip(set1, set2))):

            # Ignore the points that were fit
            if idx in init_indices:
                continue

            # Get euclidean distance between the two points after transformation is applied
            pt1 = np.concatenate((pt1, [1]))
            pt2 = np.concatenate((pt2, [1]))
            difference = pt1 - np.dot(ls_affinetransform, pt2)
            distance = np.sqrt(np.square(difference[0]) + np.square(difference[1]))
            
            # Check with inlier threshold to see if the match is an inlier 
            if distance <= inlier_threshold:
                n_inliers += 1
                inlier_indices.append(idx)
                
                # Calculating residual for this transformation
                residuals.append([np.square(difference[0]), np.square(difference[1])])
                

        # Create dict for each iteration
        ransac_iteration["H"] = ls_affinetransform
        ransac_iteration["n_inliers"] = n_inliers
        ransac_iteration["n_outliers"] = len(set1) - (n_inliers + init_points)
        ransac_iteration["residuals"] = residuals
        ransac_iteration["inlier_indices"] = inlier_indices


        ransac_runs.append(ransac_iteration)

    end_time = time.time() - start_time

    # Sort RANSAC iteration results based on the number of inliers and pick the one with the highest inliers
    ransac_runs.sort(key=lambda x: x["n_inliers"])
    best_run = ransac_runs[-1]

    # Refit the transformation on all the inliers to obtained the best fit.
    best_points = set1[best_run["inlier_indices"]], set2[best_run["inlier_indices"]]
    best_model = fit_model(*best_points)
    
    # Get averaged residuals for x and y axis
    residuals = np.array(best_run["residuals"])
    average_residuals = np.average(residuals, axis=0)

    logger.info("Inlier Ratio: {}".format(float(best_run["n_inliers"])/(len(set1) - init_points) ))

    logger.info("Average residuals for x axis: {}; y axis: {}".format(*average_residuals))
    logger.info("Time taken for RANSAC: {} seconds".format(end_time))

    best_run["H"] = best_model

    return best_run

