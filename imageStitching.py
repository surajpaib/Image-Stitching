import cv2
import numpy as np
import logging

from keypoint_detector import KeypointDetector
from keypoint_matcher import Matcher

logging.basicConfig(level=logging.INFO)


def main(args):
    first_image = cv2.imread(args.first_image_path)
    second_image = cv2.imread(args.second_image_path)

    harris_kd = KeypointDetector(block_size=args.harris_neighbourhood_size, descriptor_method='sift', keypoint_threshold=0.05)
    kp1, desc1 = harris_kd.detectAndCompute(first_image)
    drawImage = cv2.drawKeypoints(first_image, kp1, None)
    cv2.imshow('First image', drawImage)
    cv2.waitKey(0)

    kp2, desc2 = harris_kd.detectAndCompute(second_image)
    
    drawImage = cv2.drawKeypoints(second_image, kp2, None)
    cv2.imshow('Second image', drawImage)
    cv2.waitKey(0)

    logging.info("Descriptor shape for image1: {} \n Descriptor shape for image2: {}".format(desc1.shape, desc2.shape))
    matcher = Matcher()
    matches = matcher.match(desc1, desc2, k=1)

    matches.sort(key= lambda x: x[0].distance)

    matches = matches[:args.n_matches]


    matched_image = cv2.drawMatchesKnn(first_image, kp1, second_image, kp2, matches, None, flags=2)

    matched_image = cv2.resize(matched_image, (640, 480))

    cv2.imshow('matched_image', matched_image)
    cv2.waitKey(0)
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("first_image_path", help="Path to first image in the pair")
    parser.add_argument("second_image_path", help="Path to second image in the pair")
    parser.add_argument("--harris_neighbourhood_size", help="Number of pixels in the harris neighbourhood", type=int, default=2)
    parser.add_argument("--n_matches", help="Number of pixels in the harris neighbourhood", type=int, default=100)
    args = parser.parse_args()
    main(args)
