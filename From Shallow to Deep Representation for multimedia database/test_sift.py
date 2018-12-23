import unittest

import cv2 as cv
import matplotlib.pyplot as plt


class TestSift(unittest.TestCase):
    def test_sift_keypoints(self):
        import numpy as np
        import cv2 as cv
        img = cv.imread('fly.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)
        img = cv.drawKeypoints(gray, kp, img)
        cv.imwrite('fly_sift_keypoints.jpg', img)

    def test_sift_keypoints_rich(self):
        import numpy as np
        import cv2 as cv
        img = cv.imread('fly.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)
        img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite('sift_rich_keypoints.jpg', img)

    def test_sift_descriptor(self):
        import numpy as np
        import cv2 as cv
        img = cv.imread('fly.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)
        img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite('sift_rich_keypoints.jpg', img)

        #TODO Now to calculate the descriptor, OpenCV provides two methods.
        #Since you already found keypoints, you can call sift.compute() which
        #       computes the descriptors from the keypoints we have found.
        #       Eg: kp,des = sift.compute(gray,kp)
        #If you didn't find keypoints, directly find keypoints and descriptors
        #       in a single step with the function, sift.detectAndCompute().






if __name__ == '__main__':
    unittest.main()