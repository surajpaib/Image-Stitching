import cv2
import numpy as np



def main(args):
    first_image = cv2.imread(args.first_image_path)
    second_image = cv2.imread(args.second_image_path)

    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("first_image_path", help="Path to first image in the pair")
    parser.add_argument("second_image_path", help="Path to second image in the pair")
    parser.add_argument("--harris_neighbourhood_size", help="Number of pixels in the harris neighbourhood", type=int, default=2)
