import numpy as np
import random as rd
import cv2 as cv


class RandomTransformation:
    """
    Class to give a random transformation from an image input
    """

    def scal_img(self, img):
        """
        Scale the image
        :param img: the input image
        :return: the ouput image
        """
        height, width = img.shape[:2]
        res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
        return res

    def tran_img(self, img):
        """
        Translate the image
        :param img: the input image
        :return: the ouput image
        """
        rows, cols = img.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
        dst = cv.warpAffine(img, M, (cols, rows))
        return dst

    def rota_img(self, img):
        """
        Rotate the image
        :param img: the input image
        :return: the ouput image
        """
        rows, cols = img.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
        dst = cv.warpAffine(img, M, (cols, rows))
        return dst

    def pers_img(self, img):
        """
        Perspective transformation on the image
        :param img: the input image
        :return: the ouput image
        """
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv.getPerspectiveTransform(pts1, pts2)
        dst = cv.warpPerspective(img, M, (300, 300))
        return dst

    def select_transformation(self, name_img, num):
        """
        Select a transformation using the number
        :param name_img: the name of the image
        :param num: the number of the transformation
        :return: the output image
        """
        if num == 1:
            img = cv.imread(name_img)
            return self.scal_img(img)
        if num == 2:
            img = cv.imread(name_img, 0)
            return self.tran_img(img)
        if num == 3:
            img = cv.imread(name_img, 0)
            return self.rota_img(img)
        if num == 4:
            img = cv.imread(name_img)
            return self.pers_img(img)

    def apply_random_transformation(self, name_img):
        """
        Give the random transformation image
        :param name_img: the name of the image
        :return: the output image transforme
        """
        # random number between 1 and 4
        random_number = rd.randint(1, 4)
        return self.select_transformation(name_img, random_number)
