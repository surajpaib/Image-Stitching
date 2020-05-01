import numpy as np
import random

def fit_model(p1, p2):
    p1 = np.concatenate((p1, np.ones((p1.shape[0], 1))), axis=1)
    p2 = np.concatenate((p2, np.ones((p2.shape[0], 1))), axis=1)
    model_params = np.linalg.lstsq(p1, p2)[0].T
    return model_params


def RANSAC(set1, set2, N=1000, init_points=3, inlier_threshold=20):
    # Score between inlier keypoints in image1 and transformed inlier keypoints in image 2.
    ransac_runs = []

    for n in range(N):
        ransac_iteration = {}
        init_indices = random.sample(range(set1.shape[0]), init_points)
        set1_points, set2_points = set1[init_indices], set2[init_indices]

        ls_affinetransform = fit_model(set1_points, set2_points)
        
        n_inliers = 0
        inlier_indices = []
        for idx, (pt1, pt2) in enumerate((zip(set1, set2))):
            if idx in init_indices:
                continue

            pt1 = np.concatenate((pt1, [1]))
            pt2 = np.concatenate((pt2, [1]))

            difference = pt2 - np.dot(ls_affinetransform, pt1)
            distance = np.sqrt(np.square(difference[0]) + np.square(difference[1]))
            if distance <= inlier_threshold:
                n_inliers += 1
                inlier_indices.append(idx)

        ransac_iteration["H"] = ls_affinetransform
        ransac_iteration["n_inliers"] = n_inliers
        ransac_iteration["n_outliers"] = len(set1) - (n_inliers + init_points)

        ransac_iteration["inlier_indices"] = inlier_indices


        ransac_runs.append(ransac_iteration)


    ransac_runs.sort(key=lambda x: x["n_inliers"])
    return ransac_runs

