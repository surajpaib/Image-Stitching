import cv2
import numpy as np
import logging

from homography import RANSAC
from keypoint_detector import KeypointDetector
from keypoint_matcher import Matcher

from utils import compute_euclidean_distance, save_experiment, gui_display, wandb_log


logging.basicConfig(level=logging.INFO)


def main(args):

    # Initialize left and right image!
    left_image = cv2.imread(args.left_image_path)
    right_image = cv2.imread(args.right_image_path)

    # Create keypoint detector object by passing all relevant arguments that it needs. patch_size argument only for custom 
    # descriptors since SIFT automatically chooses 16x16 neighbourhood. descriptor_method decides which method to use for the 
    # descriptor generation
    harris_kd = KeypointDetector(block_size=args.harris_neighbourhood_size, descriptor_method=args.descriptor, \
                                keypoint_threshold=args.harris_keypoint_threshold, patch_size=args.patch_size)
    
    # Compute keypoints and descriptors for both the left and right images
    keypoint1, descriptor1 = harris_kd.detect_compute_descriptor(left_image)
    keypoint2, descriptor2 = harris_kd.detect_compute_descriptor(right_image)
    
    # Create matcher object. The matcher uses either normalized correlation or euclidean distance to generate matching keypoints.
    # The method is passed in while creating the object as matching_method
    matcher = Matcher(matching_method=args.matching_method)

    # Get matches for each descriptor in the left image. So the best matching descriptor is returned for each descriptor
    matches = matcher.match(descriptor1, descriptor2)

    # Sort the matches we have based on the distance between descriptors
    matches.sort(key = lambda x:x.distance)

    # Select top n matches as passed in the arguments
    matches = matches[:args.n_matches]
    
    # Create arrays to hold the coordinates of matches for both images
    set1 = np.zeros((len(matches), 2), dtype=np.float32)
    set2 = np.zeros((len(matches), 2), dtype=np.float32)

    # Fill in these arrays with coordinates from the top n matches. queryIdx refers to indices of keypoints in first image and trainIdx to the second image.
    # The coordinates of these keypoints are extracted from DMatch objects created during the matching process.
    for i, match in enumerate(matches):
        set1[i, :] = keypoint1[match.queryIdx].pt
        set2[i, :] = keypoint2[match.trainIdx].pt

    
    # RANSAC Affine transformation estimation. The two sets of keypoint coordinates are passed to the RANSAC function along with relevant arguments.
    # best_model is a dictionary with all metadata about the RANSAC process such as inlier count, inlier_indices, residuals and the best estimated affine transformation H
    best_model = RANSAC(set1, set2, init_points=args.RANSAC_init_points, N=args.RANSAC_iterations, inlier_threshold=args.RANSAC_inlier_threshold)

    # Collect all the inlier matches from the RANSAC estimation
    inlier_matches = [match for idx, match in enumerate(matches) if idx in best_model["inlier_indices"]]


    # Get the sensitivity score here for the keypoints and transformed keypoints for all the inliers.
    sensitivity = compute_euclidean_distance(set1[best_model["inlier_indices"]], set2[best_model["inlier_indices"]], best_model["H"])

    logging.info("Sensitivity score: {}".format(sensitivity))

    # Affine warp to create the final panorama
    stitchedImage = cv2.warpAffine(right_image, best_model["H"][:-1, :], (left_image.shape[1] + right_image.shape[1], left_image.shape[0]))
    stitchedImage[0:left_image.shape[0], 0:left_image.shape[1]] = left_image

    # Collect the arguments for experiment logging
    params = vars(args)
    params["Sensitivity"] = sensitivity
    
    # Keys of the best model are added to params to be saved
    params["n_inliers"] = best_model["n_inliers"]
    params["inlier_ratio"] = best_model["inlier_ratio"]
    params["average_residuals"] = best_model["average_residuals"]
   

    # Log dict for drawing GUI/ Uploading to dashboard

    if not args.no_gui or args.wandb:
        log_dict = {}
        log_dict["left_image"] = left_image
        log_dict["right_image"] = right_image
        log_dict["keypoint1"] = keypoint1
        log_dict["keypoint2"] = keypoint2
        log_dict["matches"] = matches
        log_dict["inlier_matches"] = inlier_matches
        log_dict["sensitivity"] = sensitivity
        log_dict["stitchedImage"] = stitchedImage

    # Draw only if GUI is enabled!
    if not args.no_gui:
        gui_display(log_dict)
    
    # Wandb dashboard interactive logging to visualize results.
    if args.wandb:
        wandb_log(log_dict, params)

    # Save experiment parameters to a csv file with sensitivity score.
    save_experiment(params)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image_path", help="Path to first image in the pair")
    parser.add_argument("right_image_path", help="Path to second image in the pair")

    # Keypoint parameters!
    parser.add_argument("--harris_neighbourhood_size", help="Number of pixels in the harris neighbourhood", type=int, default=2)
    parser.add_argument("--harris_keypoint_threshold", help="Harris keypoint selection threshold", type=float, default=0.01)
    parser.add_argument("--descriptor", help="Type of descriptor to choose: sift | pixel_neighbourhood", type=str, default='sift')
    parser.add_argument("--patch_size", help="Patch size, ignore for sift since it does it by default", type=int, default=5)

    # Matcher parameters!
    parser.add_argument("--n_matches", help="Number of top matches to choose for RANSAC", type=int, default=500)
    parser.add_argument("--matching_method", help="Method to use for matching between the keypoints", type=str, default='euclidean')

    # RANSAC Parameters!
    parser.add_argument("--RANSAC_iterations", help="Number of iterations to run for RANSAC", type=int, default=1000)
    parser.add_argument("--RANSAC_init_points", help="Number of starting points to choose for RANSAC", type=int, default=5)
    parser.add_argument("--RANSAC_inlier_threshold", help="Threshold to choose inliers for RANSAC", type=float, default=20)
    
    # Application settings
    parser.add_argument("--no_gui", help="Set to false for no display", default=False, type=bool)
    parser.add_argument("--wandb", help="Weights and Biases integration for experiment tracking", default=False, type=bool)

    parser.add_argument("--results_file", help="Path to sensitivity analysis results", type=str, default="results")

    args = parser.parse_args()
    main(args)
