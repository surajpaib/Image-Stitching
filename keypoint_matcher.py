import numpy as np
import cv2
import logging
import time

logger = logging.getLogger(__name__)


def normalize(vector):
    """
    Normalize numpy array using mean and variance
    """
    return (vector - vector.mean())/vector.std()

def euclidean(query_desc, train_desc):
    """
    inputs
    query_desc: numpy array of descriptors for query image
    train_desc: numpy array of descriptors for train image

    returns
    distance: Euclidean distance between two arrays
    """
    distance = np.linalg.norm(query_desc - train_desc)
    return distance

def correlation(query_desc, train_desc):
    """
    inputs
    query_desc: numpy array of descriptors for query image
    train_desc: numpy array of descriptors for train image

    returns
    correlation: Negative correlation between two array. Negative sign is added since lower is better for the distance
    """
    correlation = -(np.correlate(query_desc, train_desc)/query_desc.shape[0])
    return correlation


class Matcher:
    def __init__(self, matching_method='correlation'):
        """
        Matcher class to match descriptors using a matching method
        
        Parameters
        -------------------------------------------------------------
        matching_method: Method used to compute distance between the descriptors

        Methods
        -------------------------------------------------------------
        get_distance_func
        match
        query_image_matching
        """
        self.matching_method = matching_method

        logging.info('\n')
        logger.info("Using {} method for matching".format(self.matching_method))

    def get_distance_func(self):
        """
        Get the distance function selected from the parameter and perform necessary pre-processing
        """
        self.distance_func = eval(self.matching_method)

        # Normalize values if euclidean
        if self.matching_method == "euclidean":
            self.desc1 = normalize(self.desc1)
            self.desc2 = normalize(self.desc2)


    def match(self, descriptor1, descriptor2):
        """
        Get matches between two descriptors by calling the descriptor matching function
        """
        self.desc1 = descriptor1
        self.desc2 = descriptor2           

        # Set distance function and preprocessing
        self.get_distance_func()


        start_time = time.time()

        # Call descriptor_matching function to get the actual matches
        self.matches = self.descriptor_matching()
        total_time = time.time() - start_time
        logger.info("Time taken for matching: {} seconds".format(np.round(total_time)))

        return self.matches

    def descriptor_matching(self):
        """
        Compute matches between descriptors. 
        For descriptors from first image, a brute force best match is calculated with all descriptors from the second image.
        The best match is returned for each descriptor
        """
        matches = []

        # Iterate over each descriptor in image 1
        for query_idx, query_desc in enumerate(self.desc1):
            best_match = None    
            # Iterate over each descriptor in image 2
            for train_idx, train_desc in enumerate(self.desc2):
                
                # Compute distance as per set distance function
                distance = self.distance_func(query_desc, train_desc)
            
                # Check with best DMatch object if distance is lower, replace the DMatch object with current best
                if best_match is None or distance <= best_match.distance:
                    best_match = cv2.DMatch(query_idx, train_idx, distance)
 
            # Best match for each descriptor in image 1 aggregated.
            matches.append(best_match)

        return matches

