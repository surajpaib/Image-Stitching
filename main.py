import cv2
import numpy as np

from keypoint_detector import KeypointDetector

def main(args):
    first_image = cv2.imread(args.first_image_path)
    second_image = cv2.imread(args.second_image_path)

    kd = KeypointDetector(block_size=args.harris_neighbourhood_size)
    kd.set_image(first_image)
    kd.get_keypoints()
    desc = kd.get_descriptors()

    print(desc.shape)
    # cv2.imshow('keypoints', first_image)
    # cv2.waitKey(0)
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("first_image_path", help="Path to first image in the pair")
    parser.add_argument("second_image_path", help="Path to second image in the pair")
    parser.add_argument("--harris_neighbourhood_size", help="Number of pixels in the harris neighbourhood", type=int, default=5)
    args = parser.parse_args()
    main(args)
