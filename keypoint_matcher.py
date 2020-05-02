import numpy as np
import cv2
import logging
import time

logger = logging.getLogger(__name__)


def normalize(vector):
    return (vector - vector.mean())/vector.std()

def euclidean(query_desc, train_desc):
    distance = np.linalg.norm(query_desc - train_desc)
    return distance

def correlation(query_desc, train_desc):
    correlation = -(np.correlate(query_desc, train_desc)/query_desc.shape[0])
    return correlation


class Matcher:
    def __init__(self, matching_method='correlation', n_matches=200, match_threshold=0.9):
        self.matching_method = matching_method
        self.n_matches = n_matches
        self.match_threshold = match_threshold

        logging.info('\n')
        logger.info("Using {} method for matching".format(self.matching_method))

    def get_distance_func(self):
        self.distance_func = eval(self.matching_method)

        if self.matching_method == "euclidean":
            self.desc1 = normalize(self.desc1)
            self.desc2 = normalize(self.desc2)



    def match(self, descriptor1, descriptor2):
        self.desc1 = descriptor1
        self.desc2 = descriptor2           

        self.get_distance_func()


        start_time = time.time()
        self.matches = self.query_image_matching()
        total_time = time.time() - start_time
        logger.info("Time taken for matching: {} seconds".format(np.round(total_time)))

        return self.matches

    def query_image_matching(self):

        matches = []
        for query_idx, query_desc in enumerate(self.desc1):
            best_match = None    
            for train_idx, train_desc in enumerate(self.desc2):
                
                distance = self.distance_func(query_desc, train_desc)
            
                if best_match is None or distance <= best_match.distance:
                    best_match = cv2.DMatch(query_idx, train_idx, distance)
 
            matches.append(best_match)

        return matches

