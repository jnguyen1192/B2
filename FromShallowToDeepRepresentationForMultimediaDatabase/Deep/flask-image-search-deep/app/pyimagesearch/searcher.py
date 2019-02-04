# import the necessary packages
import numpy as np

from pyimagesearch.ImageRetrieveDeep import ImageRetrieveDeep


class Searcher:
    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath

    def search(self, des1, img_paths, des_paths, limit=10):
        # initialize our dictionary of results
        results = {}
        ird = ImageRetrieveDeep(img_paths, des_paths)
        # open the index file for reading
        path_imgs = ird.holiday_images(128)
        for path_img in path_imgs:
            # parse out the image ID and features, then compute the
            # cosin similarity between the features in our index
            # and our query features
            path_des2 = ird.build_path_des(path_img)
            print("Before r_mac_descriptor_using_path_des")
            des2 = ird.r_mac_descriptor_using_path_des(path_des2)
            # id = path_img split / le dernier
            print("Before id")
            id = path_img.split("/")[-1]
            print("Before score")
            score = ird.matcher(des1, des2)
            # now that we have the distance between the two feature
            # vectors, we can udpate the results dictionary -- the
            # key is the current image ID in the index and the
            # value is the distance we just computed, representing
            # how 'similar' the image in the index is to our query
            print("Before results[id]")
            results[id] = score

        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        print("Before sorted")
        results = sorted([(v, k) for (k, v) in results.items()])

        # return our (limited) results
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d
