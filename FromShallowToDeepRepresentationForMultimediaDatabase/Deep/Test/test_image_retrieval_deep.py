import unittest

import ImageRetrieveDeep as ird
import utils as ut


class TestImageRetrievalDeep(unittest.TestCase):

    def test_create_descriptor_image_directory(self):
        """Test creation of rmac descriptor"""
        deep = ird.ImageRetrieveDeep()
        file = ut.DATA_DIR + 'sample.jpg'
        print(deep.r_mac_descriptor(file))
        assert True

    def test_cosine_similarity_true(self):
        """ Test cosine similarity"""
        deep = ird.ImageRetrieveDeep()
        rmac_des1 = deep.r_mac_descriptor(ut.DATA_DIR + 'sample.jpg')
        rmac_des2 = rmac_des1

        from sklearn.metrics.pairwise import cosine_similarity
        assert cosine_similarity(rmac_des1, rmac_des2)[0] == 1.0


if __name__ == '__main__':
    unittest.main()
