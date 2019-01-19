import unittest

import cv2 as cv
import matplotlib.pyplot as plt
from RandomTransformation import RandomTransformation


class TestFindMatchOnTenFirstImagesTransformed(unittest.TestCase):

    # TODO find match with the first image transformed and original image
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

    def test_is_match(self):
        import cv2 as cv

        def is_match(img1, img2, threshold=50):
            # Initiate SIFT detector
            sift = cv.xfeatures2d.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            count = 0
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
                    count += 1
            #print(count)
            return count > threshold
        img1 = cv.imread('box.png', 0)  # queryImage
        img2 = cv.imread('box_in_scene.png', 0)  # trainImage
        print(is_match(img1, img2))

    def test_original_image_retrieve(self):
        import cv2 as cv
        import glob

        def is_match(img1, img2, threshold=400):
            # Initiate SIFT detector
            sift = cv.xfeatures2d.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            count = 0
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
                    count += 1
            #print(count)
            return count > threshold

        def train_images(directory='training'):
            """
            Get the first name of image on the directory specified
            :param number: number of name image
            :param directory: path of the directory
            :return:
            """
            image_list = []
            for index, filename in enumerate(glob.glob(directory + '/*.jpg')):  # assuming gif
                image_list.append(filename)
            return image_list

        def get_images_transformed(img1, tis=train_images()):
            """
            Get all the transformed image from the input image
            :param img1: the input image
            :param tis: the train images
            :return: the image which seems like the input image
            """
            image_list = []
            for train_image in tis:
                img2 = cv.imread(train_image, 0)  # trainImage
                # print(train_image)
                # case image is found
                if is_match(img1, img2):
                    #print("image trouvee ", train_image)
                    image_list.append(train_image)
                    #print(train_image)
            if len(image_list) == 0:
                return -1
            return image_list


        #tis = train_images()
        img1 = cv.imread('image_to_find_match/009885_new.jpg', 0)  # queryImage
        print("image a trouvee 009885_new.jpg")
        image_list = get_images_transformed(img1)
        print(image_list)



        #img2 = cv.imread('box_in_scene.png', 0)  # trainImage
        #print(is_match(img1, img2))


if __name__ == '__main__':
    unittest.main()
