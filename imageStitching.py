import cv2
import numpy as np
import logging

from homography import RANSAC
from keypoint_detector import KeypointDetector
from keypoint_matcher import Matcher

from utils import compute_euclidean_distance, save_experiment


logging.basicConfig(level=logging.INFO)


def main(args):

    # Initialize left and right image!
    left_image = cv2.imread(args.left_image_path)
    right_image = cv2.imread(args.right_image_path)


    harris_kd = KeypointDetector(block_size=args.harris_neighbourhood_size, descriptor_method=args.descriptor, \
                                keypoint_threshold=args.harris_keypoint_threshold, patch_size=args.patch_size)
    
    keypoint1, descriptor1 = harris_kd.detect_compute_descriptor(left_image)
    keypoint2, descriptor2 = harris_kd.detect_compute_descriptor(right_image)
    

    matcher = Matcher(matching_method=args.matching_method)
    matches = matcher.match(descriptor1, descriptor2)

    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:args.n_matches]
    
    set1 = np.zeros((len(matches), 2), dtype=np.float32)
    set2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        set1[i, :] = keypoint1[match.queryIdx].pt
        set2[i, :] = keypoint2[match.trainIdx].pt

    best_model = RANSAC(set1, set2, init_points=args.RANSAC_init_points, N=args.RANSAC_iterations, inlier_threshold=args.RANSAC_inlier_threshold)

    # Draw inlier matches!
    inlier_matches = [match for idx, match in enumerate(matches) if idx in best_model["inlier_indices"]]


    # Sensitivity Analysis score!
    sensitivity = compute_euclidean_distance(set1[best_model["inlier_indices"]], set2[best_model["inlier_indices"]], best_model["H"])
    # compute_euclidean_distance(set1, set2, best_model["H"])
    logging.info("Sensitivity score: {}".format(sensitivity))

    # Perspective warp to create the panorama
    stitchedImage = cv2.warpPerspective(right_image, best_model["H"], (left_image.shape[1] + right_image.shape[1], left_image.shape[0]))
    stitchedImage[0:left_image.shape[0], 0:left_image.shape[1]] = left_image

    # Draw only if GUI is enabled!
    if not args.no_gui:
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

        drawleft_image = cv2.drawKeypoints(left_image, keypoint1, None)
        cv2.imshow('Image', drawleft_image)
        cv2.waitKey(0)

        drawright_image = cv2.drawKeypoints(right_image, keypoint2, None)
        cv2.imshow('Image', drawright_image)
        cv2.waitKey(0)

        matched_image = cv2.drawMatches(left_image, keypoint1, right_image, keypoint2, inlier_matches, None, flags=2)
        cv2.imshow('Image', matched_image)
        cv2.waitKey(0)

        cv2.imshow('Image', stitchedImage)
        cv2.waitKey(0)

    # Save experiment parameters
    params = vars(args)
    params["Sensitivity"] = sensitivity
    save_experiment(params)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image_path", help="Path to first image in the pair")
    parser.add_argument("right_image_path", help="Path to second image in the pair")

    # Keypoint parameters!
    parser.add_argument("--harris_neighbourhood_size", help="Number of pixels in the harris neighbourhood", type=int, default=2)
    parser.add_argument("--harris_keypoint_threshold", help="Harris keypoint selection threshold", type=float, default=0.05)
    parser.add_argument("--descriptor", help="Type of descriptor to choose: sift | pixel_neighbourhood", type=str, default='pixel_neighbourhood')
    parser.add_argument("--patch_size", help="Patch size, ignore for sift since it does it by default", type=int, default=5)

    # Matcher parameters!
    parser.add_argument("--n_matches", help="Number of top matches to choose for RANSAC", type=int, default=500)
    parser.add_argument("--matching_method", help="Method to use for matching between the keypoints", type=str, default='euclidean')

    # RANSAC Parameters!
    parser.add_argument("--RANSAC_iterations", help="Number of iterations to run for RANSAC", type=int, default=1000)
    parser.add_argument("--RANSAC_init_points", help="Number of starting points to choose for RANSAC", type=int, default=5)
    parser.add_argument("--RANSAC_inlier_threshold", help="Threshold to choose inliers for RANSAC", type=float, default=50)
    
    # Application settings
    parser.add_argument("--no_gui", help="Set to false for no display", default=False, type=bool)
    parser.add_argument("--results_file", help="Path to sensitivity analysis results", type=str, default="results.csv")

    args = parser.parse_args()

    main(args)
