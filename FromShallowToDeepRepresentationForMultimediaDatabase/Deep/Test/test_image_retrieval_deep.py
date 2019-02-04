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

    def test_read_imgs(self):
        """ Test read image on directory"""
        deep = ird.ImageRetrieveDeep("data/dataset")
        print(len(deep.holiday_images(814)))

    def test_build_path_des(self):
        """ Test build the path of the descriptor"""
        deep = ird.ImageRetrieveDeep("data", "descriptor")
        print(deep.build_path_des("data/103800.jpg"))

    def test_build_descriptor_directory_using_images(self):
        """ Test building descriptor directory"""
        deep = ird.ImageRetrieveDeep("data/dataset", "descriptor")
        res = deep.build_descriptor_directory_using_images(812)
        assert res == 0

    def test_read_mat_file_extension(self):
        """ Try to read a mat file"""
        import scipy
        mat = scipy.io.loadmat("PCAmatrices.mat")
        print(mat)


if __name__ == '__main__':
    unittest.main()
