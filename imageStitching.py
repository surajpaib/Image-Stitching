import cv2
import numpy as np
import logging

from homography import RANSAC
from keypoint_detector import KeypointDetector
from keypoint_matcher import Matcher

logging.basicConfig(level=logging.INFO)


def main(args):
    # Initialize left and right image!
    left_image = cv2.imread(args.left_image_path)
    right_image = cv2.imread(args.right_image_path)


    harris_kd = KeypointDetector(block_size=args.harris_neighbourhood_size, descriptor_method=args.descriptor, keypoint_threshold=args.harris_keypoint_threshold)
    
    keypoint1, descriptor1 = harris_kd.detect_compute_descriptor(left_image)
    
    drawImage = cv2.drawKeypoints(left_image, keypoint1, None)
    # cv2.imshow('Image', drawImage)
    # cv2.waitKey(0)

    keypoint2, descriptor2 = harris_kd.detect_compute_descriptor(right_image)
    
    drawImage = cv2.drawKeypoints(right_image, keypoint2, None)
    # cv2.imshow('Image', drawImage)
    # cv2.waitKey(0)

    logging.info("Descriptor shape for image1: {} \n Descriptor shape for image2: {}".format(descriptor1.shape, descriptor2.shape))
    matcher = cv2.BFMatcher()
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
    matched_image = cv2.drawMatches(left_image, keypoint1, right_image, keypoint2, inlier_matches, None, flags=2)
    cv2.imshow('Image', matched_image)
    cv2.waitKey(0)

    # Perspective warp to create the panorama
    stitchedImage = cv2.warpPerspective(right_image, best_model["H"], (left_image.shape[1] + right_image.shape[1], left_image.shape[0]))
    stitchedImage[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
  

    cv2.imshow('Image', stitchedImage)
    cv2.waitKey(0)
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image_path", help="Path to first image in the pair")
    parser.add_argument("right_image_path", help="Path to second image in the pair")

    # Keypoint parameters!
    parser.add_argument("--harris_neighbourhood_size", help="Number of pixels in the harris neighbourhood", type=int, default=2)
    parser.add_argument("--harris_keypoint_threshold", help="Harris keypoint selection threshold", type=float, default=0.05)
    parser.add_argument("--descriptor", help="Type of descriptor to choose", type=str, default='sift')
    parser.add_argument("--patch_size", help="Patch size, ignore for sift since it does it by default", type=int, default=5)

    # Matcher parameters!
    parser.add_argument("--n_matches", help="Number of top matches to choose for RANSAC", type=int, default=500)

    # RANSAC Parameters
    parser.add_argument("--RANSAC_iterations", help="Number of iterations to run for RANSAC", type=int, default=1000)
    parser.add_argument("--RANSAC_init_points", help="Number of starting points to choose for RANSAC", type=int, default=3)
    parser.add_argument("--RANSAC_inlier_threshold", help="Threshold to choose inliers for RANSAC", type=float, default=50)
    
    args = parser.parse_args()
    main(args)
