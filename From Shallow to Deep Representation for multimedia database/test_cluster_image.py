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


if __name__ == '__main__':
    unittest.main()
