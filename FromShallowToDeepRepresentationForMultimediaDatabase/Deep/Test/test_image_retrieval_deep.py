import unittest

import os

import numpy as np
from ImageRetrieveDeep import ImageRetrieveDeep


class TestImageRetrievalLocal(unittest.TestCase):

    def test_create_descriptor_image_directory(self):
        irg = ImageRetrieveGlobal("../dataset_jpg")
        irg.build_descriptor_directory_using_images(10)
        assert True


if __name__ == '__main__':
    unittest.main()
