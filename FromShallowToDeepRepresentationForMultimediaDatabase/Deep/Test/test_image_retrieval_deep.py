import unittest

import keras_rmac.rmac as rm
import utils as ut


class TestImageRetrievalDeep(unittest.TestCase):

    def test_create_descriptor_image_directory(self):
        rm.r_mac_descriptor(ut.DATA_DIR + 'sample.jpg')
        assert True


if __name__ == '__main__':
    unittest.main()
