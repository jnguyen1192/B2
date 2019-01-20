import unittest

import os

import numpy as np
import cv2 as cv
from ImageRetrieveLocal import ImageRetrieveLocal


class TestImageRetrievalLocal(unittest.TestCase):

    def test_code(self):
        path_img = "../cluster_res_1/0/126400.jpg"
        os.system("mkdir " + str(0))
        print(path_img.split("/")[-1])
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        img = cv.imread(path_img, 0)
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        np.savetxt("0/des", des)

    def test_save_des_with_img_into_new_cluster(self):
        """
        Test if the function to save descriptor and image on a new directory works.
        """
        irl = ImageRetrieveLocal("../holiday_dataset")
        path_img = "../cluster_res_1/0/126400.jpg"

        des = irl.extract_descriptor_from_path_img(path_img)

        assert(irl.save_des_with_img_into_new_cluster(des, path_img) == 0)

    def test_get_des_from_path_des(self):
        """
        Test if the function to get a descriptor using a path of the file containing
        the descriptor.
        """
        path_img = "../cluster_res_1/0/126400.jpg"
        irl = ImageRetrieveLocal("../holiday_dataset")
        path_des = str(0) + "/des"

        des1 = irl.get_des_from_path_des(path_des)
        des2 = irl.extract_descriptor_from_path_img(path_img)

        assert(np.array_equal(des1, des2))




if __name__ == '__main__':
    unittest.main()
