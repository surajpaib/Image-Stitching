import cv2
import numpy as np
import logging

from utils import array2opencvkp

logger = logging.getLogger(__name__)

# Add epsilon to deal with numpy favoring even rounding values.
eps = 0.000001

class KeypointDetector:
    def __init__(self, block_size=2, keypoint_threshold=0.01, descriptor_method='pixel_neighbourhood', patch_size=5):
        """
        KeypointDetector class to compute keypoints and descriptors for an image
        
        Parameters
        -------------------------------------------------------------
        block_size: Size of window used for Harris keypoint detection
        keypoint_threshold: Threshold of values to consider as a keypoint (edge), indicates keypoints considered as a percentage of max keypoint edge score.
        descriptor_method: Method chosen to extract descriptors from an image.
        patch_size: If a custom descriptor_method is chosen, patch_size determines how much of the neighbourhood to consider

        Methods
        -------------------------------------------------------------
        detect_compute_descriptor
        get_keypoints
        get_descriptors
        pixel_neighbourhood
        sift
        """
        self.block_size = block_size
        self.keypoint_threshold = keypoint_threshold
        self.descriptor_method = descriptor_method
        self.patch_size = patch_size 

    def detect_compute_descriptor(self, image):
        """
        Function to get both keypoints are descriptors from an image
        image: numpy array of the image

        returns
        keypoints, descriptors
        """
        self.image = image
        self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.get_keypoints()
        self.get_descriptors()
        self.keypoints = array2opencvkp(self.keypoints)
        return self.keypoints, self.descriptors

    def get_keypoints(self):
        """
        Function to get keypoints from the image using cornerHarris method from opencv with the set parameters. 
        Use threshold to select keypoints with a good edge score, this threshold is a parameter of the keypoint 
        detection
        
        returns
        keypoints
        """
        self.harris_keypoints = cv2.cornerHarris(self.grayscale, self.block_size, 3, 0.04)
        self.keypoints = np.argwhere(self.harris_keypoints>self.keypoint_threshold*self.harris_keypoints.max())

        return self.keypoints

    def get_descriptors(self):
        """
        Compute descriptors for the keypoints obtained in the get_keypoints method.
        A descriptor is a n-element vector for each keypoint.
        """

        # Get descriptor function based on argument
        desc_func = eval('self.{}'.format(self.descriptor_method))

        logger.info("Computing descriptor using: {}".format(self.descriptor_method))
        
        # The relevant descriptor function is called
        self.descriptors = desc_func()

        # Delete descriptors and corrensponding keypoints that do not have the patch size pixels in their neighbourhood (corner of image pixels) 
        empty_descriptors = np.where(~self.descriptors.any(axis=1))[0]
        self.descriptors = np.delete(self.descriptors, empty_descriptors, axis=0)
        self.keypoints = np.delete(self.keypoints, empty_descriptors, axis=0)
        
        logger.info("Descriptor of size {} computed".format(self.descriptors.shape))
        logger.info("Descriptor of size {} computed".format(self.descriptors.shape))
    
        return self.descriptors

    def pixel_neighbourhood(self):
        """
        A pixel neighbourhood around each keypoint is computed in the RGB image. The size of the neighbourhood is determined by patch_size parameter.
        The vector size is 2x(patch_size/2)^2 for each channel. The pixel neighbourhood is then flattened. 
        """
        neighbourhood_vectors = np.zeros((len(self.keypoints), np.square(2*int(np.round(self.patch_size/2 + eps)))* 3))

        # For each keypoint, collect neighbour pixels and flatten.
        for idx, keypoint in enumerate(self.keypoints):
            neighbourhood_patch = self.image[keypoint[0] - int(np.round(self.patch_size/2 + eps)):keypoint[0] + int(np.round(self.patch_size/2 + eps)), \
                keypoint[1] - int(np.round(self.patch_size/2 + eps)):keypoint[1] + int(np.round(self.patch_size/2 + eps))]

            if neighbourhood_patch.size == np.square(2*int(np.round(self.patch_size/2 + eps)))* 3:
                neighbourhood_vectors[idx] = neighbourhood_patch.flatten()

        return neighbourhood_vectors.astype(np.float32)


    def sift(self):
        """
        Get SIFT descriptors for keypoints from Harris corner
        """
        sift = cv2.xfeatures2d.SIFT_create()

        kp = array2opencvkp(self.keypoints)
        _, self.descriptors = sift.compute(self.grayscale, kp)

        return self.descriptors


    def brief(self):
        """
        Get SIFT descriptors for keypoints from Harris corner
        """
        sift = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        kp = array2opencvkp(self.keypoints)

        _, self.descriptors = brief.compute(self.grayscale, kp)

        return self.descriptors


    # def full_sift(self):
    #     """
    #     Get SIFT keypoints and descriptors
    #     """
    #     sift = cv2.xfeatures2d.SIFT_create()

    #     kp = array2opencvkp(self.keypoints)
    #     self.keypoints, self.descriptors = sift.compute(self.grayscale, kp)

    #     return self.descriptors
