import unittest

import os

import numpy as np
from ImageRetrieveGlobal import ImageRetrieveGlobal


class TestImageRetrievalLocal(unittest.TestCase):

    def test_create_descriptor_image_directory(self):
        irg = ImageRetrieveGlobal("../dataset_jpg")
        irg.build_descriptor_directory_using_images(10)
        assert True

    def test_build_path_des(self):
        """
        Test if the build path works
        """
        # the path of the image
        path_img = "../dataset_jpg/126400.jpg"
        # the object with methods
        irg = ImageRetrieveGlobal("../dataset_jpg")
        # the path of the descriptor
        path_des = irg.build_path_des(path_img)
        assert path_des == "../dataset_des/126400.txt"

    def test_list_with_descriptor(self):
        """
        Test if the list containing the descriptors of the ten
        first images is build correctly.
        """
        irg = ImageRetrieveGlobal("../dataset_jpg")
        l = irg.get_list_using_descriptors_on_directory(6)
        assert len(l) == 83426

    def test_k_means_on_descriptor_list(self):
        """
        Test the K-means algorithm using K-means
        """
        irg = ImageRetrieveGlobal("../dataset_jpg")
        X = irg.get_list_using_descriptors_on_directory(30)

        ret, label, center = irg.k_means_(X)

        # add the descriptor on the descriptor directory with the name of the image
        np.savetxt("labels.txt", label, fmt='%i')
        # add the descriptor on the descriptor directory with the name of the image
        np.savetxt("centers.txt", center)
        print("labels ", label)
        print("centers ", center)
        assert True

    def test_save_descriptor(self):
        """
        Test the save with a descriptor
        """
        irg = ImageRetrieveGlobal("../dataset_jpg")
        path_img = "../dataset_jpg/100000.jpg"
        path_des = "des.txt"
        des = irg.extract_descriptor_from_path_img(path_img)
        # add the descriptor in integer on the descriptor directory with the name of the image
        np.savetxt(path_des, des, fmt='%i')
        # get the descriptor from the descriptor file
        des_test = np.loadtxt(path_des)
        # test if the two descriptors are equals
        assert(np.array_equal(des, des_test))
        # remove the new descriptor file
        os.remove(path_des)


if __name__ == '__main__':
    unittest.main()
