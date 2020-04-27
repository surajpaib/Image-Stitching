import numpy as np
import cv2

def normalize(vector):
    return (vector - vector.min())/(vector.max() - vector.min())


def euclidean(query_desc, train_desc):
    distance = np.linalg.norm(query_desc - train_desc)
    return distance

def correlation(query_desc, train_desc):
    distance = np.correlate(query_desc, train_desc)
    return np.abs(distance)


class Matcher:
    def __init__(self, matching_method='euclidean', n_matches=200, match_threshold=0.9):
        self.matching_method = matching_method
        self.n_matches = n_matches
        self.match_threshold = match_threshold

    def match(self, descriptor1, descriptor2, k=2):
        self.k = k
        self.desc1 = descriptor1
        self.desc2 = descriptor2

        self.distance_func = eval(self.matching_method)

        self.matches = self.query_image_matching()

        return self.matches

    def query_image_matching(self):


        matches = []
        for query_idx, query_desc in enumerate(self.desc2):
            matches_per_descriptor = []
            for train_idx, train_desc in enumerate(self.desc1):
                
                
                distance = self.distance_func(query_desc, train_desc)
                match = cv2.DMatch(query_idx, train_idx, distance)

                if len(matches_per_descriptor) >= self.k:
                    matches_per_descriptor.sort(key=lambda x: x.distance)
                    if distance <= matches_per_descriptor[-1].distance:
                        matches_per_descriptor[-1] = match
                else:
                    matches_per_descriptor.append(match)
            
            matches_per_descriptor.sort(key=lambda x: x.distance)
            
            matches.append(matches_per_descriptor)

        return matches

