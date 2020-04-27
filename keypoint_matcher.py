import numpy as np
import cv2

class Matcher:
    def __init__(self, matching_method='correlation', n_matches=200, match_threshold=0.9):
        self.matching_method = matching_method
        self.n_matches = n_matches
        self.match_threshold = match_threshold

    def match(self, descriptor1, descriptor2, match_per_descriptor=2):
        self.k = match_per_descriptor
        self.desc1 = descriptor1
        self.desc2 = descriptor2

        matching_func = eval('self.{}'.format(self.matching_method))
        
        self.matches = matching_func()

        return self.matches

    def correlation(self):
        matches = []
        for query_idx, query_desc in enumerate(self.desc2):
            matches_per_descriptor = []
            for train_idx, train_desc in enumerate(self.desc1):

                corr_score = np.correlate(train_desc, query_desc)
                match = cv2.DMatch(query_idx, train_idx, corr_score)

                if len(matches_per_descriptor) >= self.k:
                    matches_per_descriptor.sort(key=lambda x: x.distance)
                    if matches_per_descriptor[-1].distance > corr_score:
                        matches_per_descriptor[-1] = match
                else:
                    matches_per_descriptor.append(match)

            matches.append(matches_per_descriptor)

        return matches


