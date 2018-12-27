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


if __name__ == '__main__':
    unittest.main()
