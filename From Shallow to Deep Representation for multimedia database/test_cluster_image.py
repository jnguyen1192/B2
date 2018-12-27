import unittest

import cv2 as cv
import matplotlib.pyplot as plt
from RandomTransformation import RandomTransformation


class TestClusterImage(unittest.TestCase):

    # TODO cluster image by directory
    def test_create_directory(self):
        # create a directory in cluster by there number
        import os
        path = 'cluster/1'
        os.makedirs(path, exist_ok=True)

    def test_add_image_on_directory(self):
        # add an image on a specific directory
        import os
        base_name = '122302.jpg'
        holiday_directory = 'holiday_dataset'
        cluster_directory = 'cluster/1'
        path_holiday = os.path.join(holiday_directory, base_name)
        img = cv.imread(path_holiday, 0)  # trainImage
        path_cluster = os.path.join(cluster_directory, base_name)
        cv.imwrite(path_cluster, img)

    def test_cluster_image(self):
        # create the cluster class
        import os

        class cluster_img:
            """
            Class to specified each cluster of image
            """
            def __init__(self, des, path_img):
                """
                Init the descriptor file and the list of images
                :param df: descriptor file
                :param path_img: the first img of the list
                """
                self.des = des
                self.list_path_img = [path_img]

            def add_img(self, path_img):
                """
                Add an image on the list of image
                :param path_img: the path of the image
                """
                self.list_path_img.append(path_img)

            def is_match_cluster(self, des, threshold=400):
                """
                Know if the descriptor matches with our cluster
                :param des: the descriptor to test
                :param threshold: the threshold for the match
                :return: True if they match
                """
                # FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)  # or pass empty dictionary
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(self.des, des, k=2)
                # Need to draw only good matches, so create a mask
                matchesMask = [[0, 0] for i in range(len(matches))]
                count = 0
                # ratio test as per Lowe's paper
                for i, (m, n) in enumerate(matches):
                    if m.distance < 0.7 * n.distance:
                        matchesMask[i] = [1, 0]
                        count += 1
                return count > threshold

            def build_directory(self, num, cluster_directory='cluster'):
                """
                Build the directory containing same images
                :param num: number of the cluster
                """
                # path for the cluster directory
                path_cluster = os.path.join(cluster_directory, str(num))
                # create the directory
                os.makedirs(path_cluster, exist_ok=True)
                # browse on theimage path list
                for path_img in self.list_path_img:
                    # get the holiday img
                    img = cv.imread(path_img, 0)  # holidayImage
                    # get the holiday img name
                    base_name = os.path.basename(path_img)
                    # create the path cluster img
                    path_cluster_img = os.path.join(path_cluster, base_name)
                    # add the new img on the cluster directory
                    cv.imwrite(path_cluster_img, img)


if __name__ == '__main__':
    unittest.main()
