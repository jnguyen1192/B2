import unittest

import cv2 as cv
import matplotlib.pyplot as plt


class TestGeometricTransformationImages(unittest.TestCase):

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
