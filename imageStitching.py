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


    harris_kd = KeypointDetector(block_size=args.harris_neighbourhood_size, descriptor_method='pixel_neighbourhood', keypoint_threshold=0.05, max_keypoints=500)
    
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




    matched_image = cv2.drawMatches(left_image, keypoint1, right_image, keypoint2, matches, None, flags=2)
    # cv2.imshow('Image', matched_image)
    # cv2.waitKey(0)

    
    set1 = np.zeros((len(matches), 2), dtype=np.float32)
    set2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        set1[i, :] = keypoint1[match.queryIdx].pt
        set2[i, :] = keypoint2[match.trainIdx].pt

    best_candidate = RANSAC(set1, set2)[0]

    H = best_candidate["H"]

    print(H)

    # # Find homography
    # h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # # Use homography
    stitchedImage = cv2.warpPerspective(right_image, H, (left_image.shape[1] + right_image.shape[1], left_image.shape[0]))
    stitchedImage[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
  

    cv2.imshow('Image', stitchedImage)
    cv2.waitKey(0)
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image_path", help="Path to first image in the pair")
    parser.add_argument("right_image_path", help="Path to second image in the pair")
    parser.add_argument("--harris_neighbourhood_size", help="Number of pixels in the harris neighbourhood", type=int, default=2)
    parser.add_argument("--max_keypoints", help="Number of pixels in the harris neighbourhood", type=int, default=500)
    parser.add_argument("--n_matches", help="Number of pixels in the harris neighbourhood", type=int, default=500)
    args = parser.parse_args()
    main(args)
