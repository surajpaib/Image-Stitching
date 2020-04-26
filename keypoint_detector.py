import cv2
import numpy as np


def pixel_neighbourhood(image, keypoints, patch_size):
    neighbourhood_vectors = np.zeros((len(keypoints), np.square(2*int(np.round(patch_size/2)))*3))
    print(neighbourhood_vectors.shape)
    for idx, keypoint in enumerate(keypoints):
        neighbourhood_patch = image[keypoint[0] - int(np.round(patch_size/2)):keypoint[0] + int(np.round(patch_size/2)), \
            keypoint[1] - int(np.round(patch_size/2)):keypoint[1] + int(np.round(patch_size/2))]

        neighbourhood_vectors[idx] = neighbourhood_patch.flatten()
        print(neighbourhood_patch.flatten().shape)


    return neighbourhood_vectors


class KeypointDetector:
    def __init__(self, block_size=2, keypoint_threshold=0.01, descriptor_method='pixel_neighbourhood', patch_size=5):
        self.block_size = block_size
        self.keypoint_threshold = keypoint_threshold
        self.descriptor_method = descriptor_method
        self.patch_size = patch_size

    def set_image(self, image):
        self.image = image
        self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def get_keypoints(self):
        keypoints = cv2.cornerHarris(self.grayscale, self.block_size, 3, 0.004)
        self.keypoints = np.argwhere(keypoints>self.keypoint_threshold*keypoints.max())
        return self.keypoints

    def get_descriptors(self):
        # func_name = eval()
        return pixel_neighbourhood(self.image, self.keypoints, self.patch_size)



