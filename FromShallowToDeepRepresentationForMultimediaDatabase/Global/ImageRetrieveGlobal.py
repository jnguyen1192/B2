import cv2 as cv
import numpy as np
import logging
import os
import glob


class ImageRetrieveGlobal:
    def __init__(self, path_imgs, des_dir="dataset_des"):
        self.path_imgs = path_imgs
        self.des_dir = des_dir

    def holiday_images(self, number_img):
        """
        Get the first name of image on the directory specified
        :param number: number of name image
        :param directory: path of the directory
        :return:
        """
        image_list = []
        for index, filename in enumerate(glob.glob(self.path_imgs + '/*.jpg')):  # assuming gif
            image_list.append(filename)
            if index >= number_img-1:
                break
        return image_list

    def extract_descriptor_from_path_img(self, path_img):
        """
        Get the descriptor from the path of the img
        :param path_img: the path of the descriptor
        :return: the descriptor
        """
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        img = cv.imread(path_img, 0)
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        return des

    def build_path_des(self, path_img):
        """
        Build the path with the descriptor directory for the descriptor
        :param path_img: the path of the current image
        :return: the path of the descriptor
        """
        des_dir = self.des_dir
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

    def get_des_from_path_des(self, path_des):
        """
        Get the descriptor from the path
        :param path_des: the path of the descriptor
        :return: the descriptor
        """
        return np.loadtxt(path_des, dtype=np.float32)

    def build_descriptor_directory_using_images(self, nb_img):
        """
        Method to build a directory with descriptor for each image.
        :param nb_img: number of descriptor we will create
        :return: 0 if it works else -1
        """
        try:
            path_imgs = self.holiday_images(nb_img)
            # browse images
            for path_img in path_imgs:
                # get the descriptor of the current image
                des = self.extract_descriptor_from_path_img(path_img)
                # build the descriptor path using the path of the image
                path_des = self.build_path_des(path_img)
                # add the descriptor on the descriptor directory with the name of the image
                np.savetxt(path_des, des, fmt='%i')
        except:
            logging.ERROR("Exec not working")
            return -1
        return 0

    def get_list_using_descriptors_on_directory(self, nb_img):
        """
        Build a list of descriptor from directory.
        :param nb_img: number of descriptor we will create
        :return: 0 if it works else -1
        """
        try:
            c = False
            X = ""
            path_imgs = self.holiday_images(nb_img)
            # browse images
            for path_img in path_imgs:
                # build the descriptor path using the path of the image
                path_des = self.build_path_des(path_img)
                # load the descriptor
                des = np.loadtxt(path_des)
                # add the descriptor on the list of descriptor
                if not c:
                    X = des
                    c = True
                else:
                    X = np.concatenate((X, des))
        except:
            logging.ERROR("Exec not working")
            return -1
        return X

    def k_means_(self, X, k=300):
        """
        Use K-means algorithm with the descriptor and the number of cluster
        :param X: the list of descriptor
        :return: ret, label and center
        """
        # convert to np.float32
        Z = np.float32(X)

        # define criteria and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        return cv.kmeans(Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

