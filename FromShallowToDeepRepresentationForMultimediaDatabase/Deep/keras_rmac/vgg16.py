# -*- coding: utf-8 -*-

from __future__ import print_function

from keras import backend as K

import utils

K.set_image_dim_ordering('th')

import warnings
warnings.filterwarnings("ignore")
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.applications import vgg16


def VGG16(input_shape=(3, 224, 224)):
    from keras.applications import VGG16
    return VGG16(weights='imagenet', include_top=False, input_shape=input_shape)


if __name__ == '__main__':
    vgg_conv = VGG16((3, utils.IMG_SIZE, utils.IMG_SIZE))
    print(vgg_conv.summary())
