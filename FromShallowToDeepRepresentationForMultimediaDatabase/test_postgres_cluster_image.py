import unittest

import cv2 as cv
import matplotlib.pyplot as plt
from RandomTransformation import RandomTransformation


class TestPostgresClusterImage(unittest.TestCase):

    # TODO cluster image by directory
    def test_create_directory(self):
        # create a directory in cluster by there number
        import os
        path = 'cluster/1'
        os.makedirs(path, exist_ok=True)

    def test_add_image_on_directory(self):
        # add an image on a specific directory
        import os
        base_name = '122302.jpg'
        holiday_directory = 'holiday_dataset'
        cluster_directory = 'cluster/1'
        path_holiday = os.path.join(holiday_directory, base_name)
        img = cv.imread(path_holiday, 0)  # trainImage
        #print(img)
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        """for f in kp:
            print("kp ", kp)
            print(f.angle)
            print(f.class_id)
            print(f.octave)
            print(f.response)
            print(f.pt)
            print(f.size)
        """
        print(type(des))
        print("des ", str(des) + "ok")
        #path_cluster = os.path.join(cluster_directory, base_name)
        #cv.imwrite(path_cluster, img)

    def test_postgres_connect(self):
        # connect to the database
        from connect import insert_image_with_descriptor

        import os
        base_name = '122302.jpg'
        holiday_directory = 'holiday_dataset'
        path_holiday = os.path.join(holiday_directory, base_name)
        img = cv.imread(path_holiday, 0)  # trainImage
        #print(img)
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)

        insert_image_with_descriptor(path_holiday, str(des))

    def test_np_array_string(self):
        # https://stackoverflow.com/questions/10529351/using-a-psycopg2-converter-to-retrieve-bytea-data-from-postgresql
        import os
        import psycopg2 as psql
        import numpy as np
        import io

        # converts from python to postgres
        def _adapt_array(text):
            from tempfile import TemporaryFile
            outfile = TemporaryFile()
            np.savetxt(outfile, text)
            #print(text)
            outfile.seek(0)
            return outfile.read()

        # converts from postgres to python
        def _typecast_array(value, cur):
            if value is None:
                return None

            data = psql.BINARY(value, cur)
            bdata = io.BytesIO(data)
            bdata.seek(0)
            return np.load(bdata)

        
        base_name = '122302.jpg'
        holiday_directory = 'holiday_dataset'
        path_holiday = os.path.join(holiday_directory, base_name)
        img = cv.imread(path_holiday, 0)  # trainImage
        #print(img)
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        #des = np.array([1., 2., 3.], float)
        my_string = des
        print(my_string)
        print(len(my_string))
        print(_adapt_array(my_string))
        #print(my_string)
        #print("my_string ", my_string)
        #new_array = np.fromstring(my_string)
        #print(_typecast_array(_adapt_array(des), "numpy"))
        #print(len(new_array))
        #print(new_array)

    def test_cluster_image(self):
        # create the cluster class
        import os
        import glob

        class ClusterImg:
            """
            Class to specified each cluster of image
            """
            def __init__(self, des, path_img):
                """
                Init the descriptor file and the list of images
                :param df: descriptor file
                :param path_img: the first img of the list
                """
                self.des = des
                self.list_path_img = [path_img]

            def add_img(self, path_img):
                """
                Add an image on the list of image
                :param path_img: the path of the image
                """
                self.list_path_img.append(path_img)

            def is_match_cluster(self, des, threshold=400):
                """
                Know if the descriptor matches with our cluster
                :param des: the descriptor to test
                :param threshold: the threshold for the match
                :return: True if they match
                """
                # FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)  # or pass empty dictionary
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(self.des, des, k=2)
                # Need to draw only good matches, so create a mask
                matchesMask = [[0, 0] for i in range(len(matches))]
                count = 0
                # ratio test as per Lowe's paper
                for i, (m, n) in enumerate(matches):
                    if m.distance < 0.7 * n.distance:
                        matchesMask[i] = [1, 0]
                        count += 1
                return count > threshold

            def build_directory(self, num, cluster_directory='cluster'):
                """
                Build the directory containing same images
                :param num: number of the cluster
                """
                # path for the cluster directory
                path_cluster = os.path.join(cluster_directory, str(num))
                # create the directory
                os.makedirs(path_cluster, exist_ok=True)
                # browse on theimage path list
                for path_img in self.list_path_img:
                    # get the holiday img
                    img = cv.imread(path_img, 0)  # holidayImage
                    # get the holiday img name
                    base_name = os.path.basename(path_img)
                    # create the path cluster img
                    path_cluster_img = os.path.join(path_cluster, base_name)
                    # add the new img on the cluster directory
                    cv.imwrite(path_cluster_img, img)

        def holiday_images(directory='holiday_dataset'):
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
        # all holiday img
        holiday_dataset = holiday_images()
        # declare cluster list
        clusters = []
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # browse all img
        for path_img in holiday_dataset:
            img = cv.imread(path_img, 0)
            # find the keypoints and descriptors with SIFT
            kp, des = sift.detectAndCompute(img, None)
            if len(clusters) == 0:
                # add the first cluster
                clusters.append(ClusterImg(des, path_img))
                continue
            cluster_match = False
            # for each img
            for cluster in clusters:
                # if it matches with cluster
                if cluster.is_match_cluster(des):
                    # add in current cluster the path_img
                    cluster.add_img(path_img)
                    cluster_match = True
                    break
            # if not match create new cluster with descriptor and path_img
            if not cluster_match:
                clusters.append(ClusterImg(des, path_img))
        # write clusters on directories
        for index, cluster in enumerate(clusters):
            cluster.build_directory(index)


if __name__ == '__main__':
    unittest.main()
