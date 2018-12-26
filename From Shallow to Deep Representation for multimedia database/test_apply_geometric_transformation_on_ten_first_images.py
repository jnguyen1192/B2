import unittest

import cv2 as cv
import matplotlib.pyplot as plt
from RandomTransformation import RandomTransformation


class TestApplyGeometricTransformationOnTenFirstImages(unittest.TestCase):

    def test_apply_transformation_on_ten_images(self):
        # apply transformation on the first ten images
        import glob
        import os

        def first_images(number=10, directory='training'):
            """
            Get the first name of image on the directory specified
            :param number: number of name image
            :param directory: path of the directory
            :return:
            """
            image_list = []
            for index, filename in enumerate(glob.glob(directory + '/*.jpg')):  # assuming gif
                image_list.append(filename)
                if index == number:
                    break
            return image_list
        rt = RandomTransformation()
        first_ten_images = first_images()
        # apply transformation on each image and stock them into image_to_find_match directory
        for name_img in first_ten_images:
            dst = rt.apply_random_transformation(name_img)
            base_name = os.path.basename(name_img)
            cv.imwrite('image_to_find_match/' + base_name[:-4] + '_new.jpg', dst)

    def test_select_ten_first_images_on_set(self):
        # select the ten first images of the training repository
        from PIL import Image
        import glob
        import os
        image_list = []
        for index, filename in enumerate(glob.glob('training/*.jpg')):  # assuming gif
            im = Image.open(filename)
            print(os.path.basename(filename))
            image_list.append(filename)
            if index == 10:
                break
        print(image_list)

    def test_random_transform_into_img(self):
        # select a transformation using random input
        rt = RandomTransformation()
        name_img = 'messi5.jpg'
        #img = cv.imread(name_img)
        dst = rt.apply_random_transformation(name_img)
        #plt.subplot(121), plt.imshow(img), plt.title('Input')
        #plt.subplot(122), plt.imshow(dst), plt.title('Output')
        #plt.show()
        cv.imwrite('image_to_find_match/messi5_new.jpg', dst)

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
