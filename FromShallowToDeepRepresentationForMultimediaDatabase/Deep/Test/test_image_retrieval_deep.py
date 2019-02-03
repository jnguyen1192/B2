import unittest

import ImageRetrieveDeep as ird
import utils as ut

import numpy as np

import warnings


class TestImageRetrievalDeep(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("ignore")

    def test_create_descriptor_image_directory(self):
        """Test creation of rmac descriptor"""
        deep = ird.ImageRetrieveDeep("")
        file = ut.DATA_DIR + 'sample.jpg'
        des = deep.r_mac_descriptor(file)
        print(type(des))
        print(des)
        np.save("rmac_sample.txt", des)
        assert True

    def test_cosine_similarity_true(self):
        """ Test cosine similarity"""
        deep = ird.ImageRetrieveDeep("")
        rmac_des1 = np.load("rmac_sample.txt.npy")
        rmac_des2 = rmac_des1

        assert deep.matcher(rmac_des1, rmac_des2) == 1.0


if __name__ == '__main__':
    unittest.main()
