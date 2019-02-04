import unittest

import ImageRetrieveDeep as ird

import numpy as np

import warnings


class TestImageRetrievalDeep(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("ignore")

    def test_cosine_similarity_true(self):
        """ Test cosine similarity"""
        deep = ird.ImageRetrieveDeep("")
        rmac_des1 = np.load("rmac_sample.txt.npy")
        rmac_des2 = rmac_des1

        assert deep.matcher(rmac_des1, rmac_des2) == 1.0

    def test_build_path_des(self):
        """ Test build the path of the descriptor"""
        deep = ird.ImageRetrieveDeep("data", "descriptor")
        assert deep.build_path_des("data/103800.jpg") == "descriptor/103800"

    def test_build_descriptor_directory_using_images(self):
        """ Test building descriptor directory"""
        deep = ird.ImageRetrieveDeep("data/dataset", "descriptor")
        res = deep.build_descriptor_directory_using_images(812)
        assert res == 0


if __name__ == '__main__':
    unittest.main()
