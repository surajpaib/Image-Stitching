import cv2
import numpy as np
import logging

from keypoint_detector import KeypointDetector
from keypoint_matcher import Matcher

logging.basicConfig(level=logging.INFO)


def main(args):
    # Initialize left and right image!
    left_image = cv2.imread(args.left_image_path)
    right_image = cv2.imread(args.right_image_path)


    harris_kd = KeypointDetector(block_size=args.harris_neighbourhood_size, descriptor_method='sift', keypoint_threshold=0.1, max_keypoints=500)
    kp1, desc1 = harris_kd.detectAndCompute(left_image)
    drawImage = cv2.drawKeypoints(left_image, kp1, None)
    cv2.imshow('Image', drawImage)
    cv2.waitKey(0)

    kp2, desc2 = harris_kd.detectAndCompute(right_image)
    
    drawImage = cv2.drawKeypoints(right_image, kp2, None)
    cv2.imshow('Image', drawImage)
    cv2.waitKey(0)

    logging.info("Descriptor shape for image1: {} \n Descriptor shape for image2: {}".format(desc1.shape, desc2.shape))
    matcher = Matcher()
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:args.n_matches]

    matched_image = cv2.drawMatches(left_image, kp1, right_image, kp2, matches, None, flags=2)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)


    accuracy_score = 0
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
        print(match.distance)
        accuracy_score += match.distance

    logging.info("Accuracy Score: {}".format(accuracy_score/len(matches)))

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography
    stitchedImage = cv2.warpPerspective(right_image, h, (left_image.shape[1] + right_image.shape[1], left_image.shape[0]))
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
