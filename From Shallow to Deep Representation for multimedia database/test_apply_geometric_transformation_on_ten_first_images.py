import unittest

import cv2 as cv
import matplotlib.pyplot as plt


class TestApplyGeometricTransformationOnTenFirstImages(unittest.TestCase):

    def test_select_ten_first_images_on_set(self):
        from PIL import Image
        import glob
        image_list = []
        for index, filename in enumerate(glob.glob('training/*.jpg')):  # assuming gif
            im = Image.open(filename)
            image_list.append(filename)
            if index == 10:
                break
        print(image_list)

    def test_random_transform_into_img(self):
        # select a transformation using random input
        import numpy as np
        import random as rd

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
                print(img.shape)
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

        rt = RandomTransformation()
        name_img = 'messi5.jpg'
        img = cv.imread(name_img)
        dst = rt.apply_random_transformation(name_img)
        plt.subplot(121), plt.imshow(img), plt.title('Input')
        plt.subplot(122), plt.imshow(dst), plt.title('Output')
        plt.show()


    def test_scaling(self):
        import numpy as np
        import cv2 as cv
        img = cv.imread('messi5.jpg')
        res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        # OR
        height, width = img.shape[:2]
        res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
        cv.imshow('img', res)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def test_translations(self):
        import numpy as np
        import cv2 as cv
        img = cv.imread('messi5.jpg', 0)
        rows, cols = img.shape
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        dst = cv.warpAffine(img, M, (cols, rows))
        cv.imshow('img', dst)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def test_rotation(self):
        img = cv.imread('messi5.jpg', 0)
        rows, cols = img.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
        dst = cv.warpAffine(img, M, (cols, rows))
        cv.imshow('img', dst)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def test_perspective_transformation(self):
        import numpy as np
        img = cv.imread('sudoku.png')
        rows, cols, ch = img.shape
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv.getPerspectiveTransform(pts1, pts2)
        dst = cv.warpPerspective(img, M, (300, 300))
        plt.subplot(121), plt.imshow(img), plt.title('Input')
        plt.subplot(122), plt.imshow(dst), plt.title('Output')
        plt.show()


if __name__ == '__main__':
    unittest.main()
