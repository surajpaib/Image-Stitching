import cv2
import numpy as np


class KeypointDetector:
    def __init__(self, block_size=2):
        self.block_size = block_size

    def set_image(self, image):
        self.image = image
        self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def get_keypoints(self):
        self.keypoints = cv2.cornerHarris(self.grayscale, self.block_size, 3, 0.004)

