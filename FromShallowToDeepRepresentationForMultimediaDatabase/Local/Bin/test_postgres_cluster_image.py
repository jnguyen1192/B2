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

    def test_postgres_insert_image_with_descriptor(self):
        # connect to the database
        from postgres_database import insert_image_with_descriptor

        import numpy as np

        # converts from python to postgres
        def _adapt_array(text):
            from tempfile import TemporaryFile
            outfile = TemporaryFile()
            np.savetxt(outfile, text)
            outfile.seek(0)
            return outfile.read()
        import os
        base_name = '122302.jpg'
        holiday_directory = 'holiday_dataset'
        path_holiday = os.path.join(holiday_directory, base_name)
        # trainImage
        img = cv.imread(path_holiday, 0)
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        # insert path image with descriptor in postgres database
        insert_image_with_descriptor(path_holiday, _adapt_array(des))

    def test_get_number_cluster(self):
        from postgres_database import get_number_cluster
        print(get_number_cluster())

    def test_case_cluster_exists(self):
        from postgres_database import get_number_cluster
        if get_number_cluster() == 0:
            print("No cluster, we will create our first cluster")
        else:
            print("Cluster exist")

    def test_case_first_cluster(self):
        from postgres_database import get_number_cluster, insert_cluster_with_descriptor, insert_has_cluster_with_num_cluster_and_name_image
        import os

        number_cluster = get_number_cluster()
        base_name = '122302.jpg'
        holiday_directory = 'holiday_dataset'
        path_holiday = os.path.join(holiday_directory, base_name)
        # trainImage
        img = cv.imread(path_holiday, 0)
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        if number_cluster == 0:
            print("No cluster, we will create our first cluster")
            # use the descriptor to add the new cluster row
            insert_cluster_with_descriptor(1, des)
            # use the path of image to add the new has_cluster row
            insert_has_cluster_with_num_cluster_and_name_image(1, path_holiday)
        else:
            print("Cluster exist")

    def test_case_other_cluster(self):
        from postgres_database import get_number_cluster, insert_cluster_with_descriptor, insert_has_cluster_with_num_cluster_and_name_image, get_descriptor_cluster_with_num_cluster
        import os

        def is_match_cluster(des1, des2, threshold=400):
            """
            Know if the descriptor matches with our cluster
            :param des1: the first descriptor to test
            :param des2: the second descriptor to test
            :param threshold: the threshold for the match
            :return: True if they match
            """
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params, search_params)
            print('des1 ', des1.shape)
            print('des2 ', des2.shape)
            matches = flann.knnMatch(des1, des2, k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            count = 0
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
                    count += 1
            return count > threshold

        number_cluster = get_number_cluster()
        base_name = '122400.jpg'  # 122302 122400
        holiday_directory = 'holiday_dataset'
        path_holiday = os.path.join(holiday_directory, base_name)
        # trainImage
        img = cv.imread(path_holiday, 0)
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des1 = sift.detectAndCompute(img, None)
        if number_cluster == 0:
            print("No cluster, we will create our first cluster")
            # use the descriptor to add the new cluster row
            insert_cluster_with_descriptor(1, des1)
            # use the path of image to add the new has_cluster row
            insert_has_cluster_with_num_cluster_and_name_image(1, path_holiday)
        else:
            print("Cluster exist")
            # we verify if the descriptor will match with one of the descriptor of each cluster
            is_cluster_match = False
            for num_cluster in range(1, number_cluster+1):
                des2 = get_descriptor_cluster_with_num_cluster(num_cluster)
                #print("des2 ", des2)
                # if match add to the cluster and break
                if is_match_cluster(des1, des2):
                    is_cluster_match = True
                    # use the path of image to add the current cluster with has_cluster row
                    insert_has_cluster_with_num_cluster_and_name_image(num_cluster, path_holiday)
                    break
            # else create a new cluster
            if not is_cluster_match:
                # use the descriptor to add the new cluster row
                insert_cluster_with_descriptor(number_cluster+1, des1)
                # use the path of image to add the new has_cluster row
                insert_has_cluster_with_num_cluster_and_name_image(number_cluster+1, path_holiday)

    def test_get_descriptor_from_cluster_using_num_cluster(self):
        from postgres_database import get_descriptor_cluster_with_num_cluster
        print(get_descriptor_cluster_with_num_cluster(1))

    def test_np_array_string(self):
        # https://stackoverflow.com/questions/10529351/using-a-psycopg2-converter-to-retrieve-bytea-data-from-postgresql
        import os
        import numpy as np

        # converts from python to postgres
        def _adapt_array(text):
            from tempfile import TemporaryFile
            outfile = TemporaryFile()
            np.savetxt(outfile, text)
            outfile.seek(0)  # Only needed here to simulate closing & reopening file
            return outfile.read()

        # converts from postgres to python
        def _typecast_array(string):
            from tempfile import TemporaryFile
            outfile = TemporaryFile()
            outfile.write(string)
            outfile.seek(0)  # Only needed here to simulate closing & reopening file
            return np.loadtxt(outfile)

        base_name = '122302.jpg'
        holiday_directory = 'holiday_dataset'
        path_holiday = os.path.join(holiday_directory, base_name)
        img = cv.imread(path_holiday, 0)  # trainImage

        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)

        my_string = _adapt_array(des)
        print("my_string ", my_string)

        new_array = _typecast_array(my_string)
        #print("new_array ", new_array)

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
