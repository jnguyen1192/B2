import cv2 as cv
import numpy as np
import logging
import os
import glob


class ImageRetrieveGlobal:
    def __init__(self, path_imgs, des_dir="dataset_des"):
        self.nb_cluster = 0
        self.path_imgs = path_imgs
        self.des_dir = des_dir

    def incr_nb_cluster(self):
        """
        Increment the number of current cluster
        """
        self.nb_cluster = self.nb_cluster + 1

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

    def save_des_in_descriptor_directory(self, des, path_img):
        """
        Save the descriptor in the descriptor directory
        into a new cluster.
        :param des: the descriptor as a numpy array
        :return: 0 if the save works else -1
        """
        try:
            # create a new directory with the nb of cluster
            os.system("mkdir -p " + str(self.nb_cluster))
            # add a file with the descriptor on this directory
            np.savetxt(str(self.nb_cluster) + "/des", des)
            # add a file with the img on this directory using the path of the img
            os.system("cp " + path_img + " " + str(self.nb_cluster) + "/" + path_img.split("/")[-1])
            self.incr_nb_cluster()
        except:
            logging.ERROR("Save not working on the descriptor directory")
            return -1
        return 0

    def get_des_from_path_des(self, path_des):
        """
        Get the descriptor from the path
        :param path_des: the path of the descriptor
        :return: the descriptor
        """
        return np.loadtxt(path_des, dtype=np.float32)

    def compare_path_des_and_des(self, path_des1,  des2, threshold=400):
        """
        Compare the two descriptor when the result is between the minimum threshold and the maximum threshold
        :param desc1: the first descriptor
        :param desc2:   the second descriptor
        :return: True if they matched else False
        """
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        count = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        # get des using path
        des1 = self.get_des_from_path_des(path_des1)
        # create the fast knn matcher
        matches = flann.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                count += 1
        return count > threshold

    def add_img_on_cluster(self, path_img, num_cluster):
        """
        Add a new image on the specific cluster.
        :param path_img: the path of the img
        :param num_cluster: the number of the cluster we will add the image
        :return: 0 if it works else -1
        """
        try:
            # add a file with the img on this directory using the path of the img
            os.system("cp " + path_img + " " + str(num_cluster) + "/" + path_img.split("/")[-1])
        except:
            logging.ERROR("Add on cluster " + str(num_cluster) + " not working on")
            return -1
        return 0

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
                np.savetxt(path_des, des)
        except:
            logging.ERROR("Exec not working")
            return -1
        return 0

    def exec(self, nb_img=50, threshold=400):
        """
        Execute the algorithm to create the cluster with each image,
        each cluster has only one descriptor.
        :param nb_img: the number of image we will use
        :return: 0 if it works else -1
        """
        try:
            # browse images
            path_imgs = self.holiday_images(nb_img)

            for path_img in path_imgs:
                # get the descriptor of the current image
                des1 = self.extract_descriptor_from_path_img(path_img)
                if self.nb_cluster == 0:
                    # add the first cluster
                    self.save_des_in_descriptor_directory(des1, path_img)
                    continue
                cluster_match = False
                for i in range(self.nb_cluster):
                    path_des = str(i) + "/des"
                    #print(path_des)
                    if self.compare_path_des_and_des(path_des, des1, threshold):
                        self.add_img_on_cluster(path_img, i)
                        cluster_match = True
                        break
                # case not match create a new cluster
                if not cluster_match:
                    self.save_des_in_descriptor_directory(des1, path_img)
        except:
            logging.ERROR("Exec not working")
            return -1
        return 0

