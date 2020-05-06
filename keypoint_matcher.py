import numpy as np
import cv2
import logging
import time

logger = logging.getLogger(__name__)


def normalize(vector):
    """
    Normalize numpy array using mean and variance
    """
    return (vector - vector.min())/(vector.max() - vector.min())

def euclidean(query_desc, train_desc):
    """
    inputs
    query_desc: numpy array of one descriptors for query image
    train_desc: numpy array of all descriptors for train image

    returns
    distance: Euclidean distance for all descriptors in train image with the descriptor in query image
    """

    distance = np.linalg.norm(query_desc - train_desc, axis=1)
    return distance

def hamming(query_desc, train_desc):
    """
    inputs
    query_desc: numpy array of one descriptors for query image
    train_desc: numpy array of all descriptors for train image

    returns
    distance: Hamming distance for all descriptors in train image with the descriptor in query image
    """
    distance = np.count_nonzero(np.bitwise_xor(query_desc, train_desc), axis=1)

    return distance


def correlation(query_desc, train_desc):
    """
    inputs
    query_desc: numpy array of one descriptors for query image
    train_desc: numpy array of all descriptors for train image

    returns
    correlation: Normalized correlation array for all descriptors in train image with the descriptor in query image
    """

    # Dot product between the vector and matrix gives the valid correlation for each descriptor in the matrix with the vector.
    # Negative since we want distance and lower is better
    correlation = -np.dot(query_desc, train_desc.T)
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

        # Normalize descriptors
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
            
            # Distance function calculates the distance between each descriptor in image to all the descriptors in image 2
            distance = self.distance_func(query_desc, self.desc2)
            # The descriptor with the lowest distance is taken as the best match for the descriptor in image 1
            best_match_idx = np.argsort(distance, axis=0)[0]

            best_match = cv2.DMatch(query_idx, best_match_idx, distance[best_match_idx])
            matches.append(best_match)

        return matches

