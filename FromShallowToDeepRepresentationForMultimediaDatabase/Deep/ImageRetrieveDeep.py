import numpy as np
import scipy.io
import glob

from RoiPooling import RoiPooling
import utils

from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications import VGG16

import keras.backend as K
K.set_image_dim_ordering('th')

from sklearn.metrics.pairwise import cosine_similarity

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide warning with tensorflow


class ImageRetrieveDeep:
    def __init__(self, path_imgs, des_dir="dataset_des"):
        self.path_imgs = path_imgs
        self.des_dir = des_dir

    def holiday_images(self, number_img):
        """
        Get the first name of image on the directory specified
        :param number: number of name image
        :param directory: path of the directory
        :return:
        """
        image_list = []
        for index, filename in enumerate(glob.glob(self.path_imgs + '/*.jpg')):  # assuming gif
            image_list.append(filename)
            if index >= number_img-1:
                break
        return image_list


    """
        RMAC descriptor
    """
    def addition(self, x):
        sum = K.sum(x, axis=1)
        return sum

    def weighting(self, input):
        x = input[0]
        w = input[1]
        w = K.repeat_elements(w, 512, axis=-1)
        out = x * w
        return out

    def get_size_vgg_feat_map(self, input_W, input_H):
        output_W = input_W
        output_H = input_H
        for i in range(1, 6):
            output_H = np.floor(output_H / 2)
            output_W = np.floor(output_W / 2)

        return output_W, output_H

    def rmac_regions(self, W, H, L):
        """
        Get the regions for the RMAC model
        :param W: the width of the image
        :param H: the height of the image
        :param L:
        :return: the regions of the image as a numpy array
        """
        ovr = 0.4  # desired overlap of neighboring regions
        steps = np.array([2, 3, 4, 5, 6, 7], dtype=np.float)  # possible regions for the long dimension

        w = min(W, H)

        b = (max(H, W) - w) / (steps - 1)
        idx = np.argmin(abs(((w ** 2 - w * b) / w ** 2) - ovr))  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd, Hd = 0, 0
        if H < W:
            Wd = idx + 1
        elif H > W:
            Hd = idx + 1

        regions = []

        for l in range(1, L + 1):

            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)

            b = (W - wl) / (l + Wd - 1)
            if np.isnan(b):  # for the first level
                b = 0
            cenW = np.floor(wl2 + np.arange(0, l + Wd) * b) - wl2  # center coordinates

            b = (H - wl) / (l + Hd - 1)
            if np.isnan(b):  # for the first level
                b = 0
            cenH = np.floor(wl2 + np.arange(0, l + Hd) * b) - wl2  # center coordinates

            for i_ in cenH:
                for j_ in cenW:
                    # R = np.array([i_, j_, wl, wl], dtype=np.int)
                    R = np.array([j_, i_, wl, wl], dtype=np.int)
                    if not min(R[2:]):
                        continue

                    regions.append(R)

        regions = np.asarray(regions)
        return regions

    def rmac(self, input_shape, num_rois):

        # Load VGG16
        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        # Regions as input
        in_roi = Input(shape=(num_rois, 4), name='input_roi')
        # ROI pooling
        x = RoiPooling([1], num_rois)([vgg16_model.layers[-5].output, in_roi])

        # Normalization
        x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

        # PCA
        x = TimeDistributed(Dense(512, name='pca',
                                  kernel_initializer='identity',
                                  bias_initializer='zeros'))(x)

        # Normalization
        x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

        # Addition
        rmac = Lambda(self.addition, output_shape=(512,), name='rmac')(x)

        # # Normalization
        rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

        # Define model
        model = Model([vgg16_model.input, in_roi], rmac_norm)

        # Load PCA weights
        mat = scipy.io.loadmat(utils.DATA_DIR + utils.PCA_FILE)
        b = np.squeeze(mat['bias'], axis=1)
        w = np.transpose(mat['weights'])
        model.layers[-4].set_weights([w, b])

        return model

    def r_mac_descriptor(self, file):
        # Load sample image
        # file = utils.DATA_DIR + 'sample.jpg'
        img = image.load_img(file)
        # Resize
        scale = utils.IMG_SIZE / max(img.size)
        new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))  # (utils.IMG_SIZE, utils.IMG_SIZE)
        #print('Original size: %s, Resized image: %s' % (str(img.size), str(new_size)))
        img = img.resize(new_size)
        # Mean substraction
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_image(x)

        # Load RMAC model
        Wmap, Hmap = self.get_size_vgg_feat_map(x.shape[3], x.shape[2])
        regions = self.rmac_regions(Wmap, Hmap, 7)
        #print('Loading RMAC model...')
        model = self.rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))

        # Compute RMAC vector
        #print('Extracting RMAC from image...')
        return model.predict([x, np.expand_dims(regions, axis=0)])

    """
        RMAC matcher
    """
    def matcher(self, des1, des2):
        return cosine_similarity(des1, des2)[0]
