import unittest

import os

import numpy as np
import cv2 as cv
from ImageRetrieveGlobal import ImageRetrieveGlobal


class TestImageRetrievalLocal(unittest.TestCase):

    def test_create_descriptor_image_directory(self):
        irg = ImageRetrieveGlobal("../dataset_jpg")
        irg.build_descriptor_directory(20)

    def test_1_code(self):
        # descriptor directory
        path_img = "../dataset_jpg/126400.jpg"

        def build_path_des(path_img):
            """
            Build the path with the descriptor directory for the descriptor
            :param path_img: the path of the current image
            :return: the path of the descriptor
            """
            des_dir = "dataset_des"
            # split the path of the image
            split_res = path_img.split("/")
            # get the image name without the extension .jpg
            img_name_without_extension = split_res[-1].split(".jpg")[0]
            # build the beginning of the path
            beg_path = os.path.join(*split_res[:-2])
            # build the end of the path
            end_path = os.path.join(des_dir, img_name_without_extension + ".txt")
            # build the descriptor path
            path_des = os.path.join(beg_path, end_path)
            return path_des
        path_des = build_path_des(path_img)
        print("path_des ", path_des)


    def test_0_code(self):
        path_img = "../../cluster_res_1/0/126400.jpg"
        os.system("mkdir -p " + str(0))
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        img = cv.imread(path_img, 0)
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        np.savetxt("0/des", des)
        assert(True)

    def test_extract_descriptor_from_path_img(self):
        """
        Test if the function extract descriptor from path image
        return the correct descriptor.
        """
        irl = ImageRetrieveGlobal("../../holiday_dataset")

        path_img = "../../cluster_res_1/0/126400.jpg"

        des1 = irl.extract_descriptor_from_path_img(path_img)
        des2 = irl.get_des_from_path_des("0/des")

        assert(np.array_equal(des1, des2))

    def test_save_des_with_img_into_new_cluster(self):
        """
        Test if the function to save descriptor and image on a new directory works.
        """
        irl = ImageRetrieveGlobal("../holiday_dataset")
        path_img = "../../cluster_res_1/0/126400.jpg"

        des = irl.extract_descriptor_from_path_img(path_img)

        assert(irl.save_des_in_descriptor_directory(des, path_img) == 0)

    def test_get_des_from_path_des(self):
        """
        Test if the function to get a descriptor using a path of the file containing
        the descriptor.
        """
        path_img = "../../cluster_res_1/0/126400.jpg"
        irl = ImageRetrieveGlobal("../../holiday_dataset")
        path_des = str(0) + "/des"

        des1 = irl.get_des_from_path_des(path_des)
        des2 = irl.extract_descriptor_from_path_img(path_img)

        assert(np.array_equal(des1, des2))

    def test_compare_path_des_and_des_true(self):
        """
        Test if the function compare descriptor works with true case.
        """
        irl = ImageRetrieveGlobal("../holiday_dataset")

        path_des = str(0) + "/des"
        path_img = "../../cluster_res_1/0/126401.jpg"

        des = irl.extract_descriptor_from_path_img(path_img)

        assert(irl.compare_path_des_and_des(path_des, des))

    def test_compare_path_des_and_des_false(self):
        """
        Test if the function compare descriptor works with false case.
        """
        irl = ImageRetrieveGlobal("../../holiday_dataset")

        path_des = str(0) + "/des"
        path_img = "../../cluster_res_1/1/126402.jpg"

        des = irl.extract_descriptor_from_path_img(path_img)

        assert(not irl.compare_path_des_and_des(path_des, des))

    def test_add_img_on_cluster(self):
        """
        Test if the image is add on the cluster
        """
        irl = ImageRetrieveGlobal("../../holiday_dataset")
        path_img = "../../cluster_res_1/0/126401.jpg"
        num_cluster = 0
        assert(irl.add_img_on_cluster(path_img, num_cluster) == 0)

    def test_holiday_images(self):
        """
        Test if 50 images are selected
        """
        nb_imgs = 50
        irl = ImageRetrieveGlobal("../../holiday_dataset")
        image_list = irl.holiday_images(nb_imgs)
        assert(len(image_list) == nb_imgs)

    def test_exec(self):
        """
        Test if execution algorithm to retrieve local image works.
        """
        irl = ImageRetrieveGlobal("../../holiday_dataset")
        assert(irl.exec(812, 250) == 0)


if __name__ == '__main__':
    unittest.main()
