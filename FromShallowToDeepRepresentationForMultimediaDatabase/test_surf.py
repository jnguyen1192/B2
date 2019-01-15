import unittest

import cv2 as cv
import matplotlib.pyplot as plt


class TestStringMethods(unittest.TestCase):

    # PART 2
    def test_len_descriptor(self):
        img = cv.imread('fly.png', 0)
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv.xfeatures2d.SURF_create(400)
        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(img, None)
        print(len(kp))

    def test_len_descriptor_using_threshold(self):
        img = cv.imread('fly.png', 0)
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv.xfeatures2d.SURF_create(400)
        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(img, None)
        #print(len(kp))
        # Check present Hessian threshold
        print(surf.getHessianThreshold())
        # We set it to some 50000. Remember, it is just for representing in picture.
        # In actual cases, it is better to have a value 300-500
        surf.setHessianThreshold(50000)
        # Again compute keypoints and check its number.
        kp, des = surf.detectAndCompute(img, None)
        print(len(kp))

    def test_draw_image_with_low_threshold(self):
        img = cv.imread('fly.png', 0)
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv.xfeatures2d.SURF_create(400)
        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(img, None)
        #print(len(kp))
        # Check present Hessian threshold
        #print(surf.getHessianThreshold())
        # We set it to some 50000. Remember, it is just for representing in picture.
        # In actual cases, it is better to have a value 300-500
        surf.setHessianThreshold(50000)
        # Again compute keypoints and check its number.
        kp, des = surf.detectAndCompute(img, None)
        #print(len(kp))
        img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        plt.imshow(img2), plt.show()

    def test_u_surf(self):
        img = cv.imread('fly.png', 0)
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv.xfeatures2d.SURF_create(400)
        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(img, None)

        # Check upright flag, if it False, set it to True
        print(surf.getUpright())
        surf.setUpright(True)
        # Recompute the feature points and draw it
        kp = surf.detect(img, None)
        img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        plt.imshow(img2), plt.show()

    def test_descriptor_size(self):
        img = cv.imread('fly.png', 0)
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv.xfeatures2d.SURF_create(400)
        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(img, None)

        # Check upright flag, if it False, set it to True
        print(surf.getUpright())
        surf.setUpright(True)

        # Find size of descriptor
        print(surf.descriptorSize())

        # That means flag, "extended" is False.
        surf.getExtended()

        # So we make it to True to get 128-dim descriptors.
        surf.setExtended(True)
        kp, des = surf.detectAndCompute(img, None)
        print(surf.descriptorSize())
        print(des.shape)





if __name__ == '__main__':
    unittest.main()