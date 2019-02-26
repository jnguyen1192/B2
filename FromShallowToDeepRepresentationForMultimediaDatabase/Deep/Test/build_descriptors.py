import ImageRetrieveDeep as ird
import os, os.path


def build_descriptor():
    """ Building descriptor directory"""
    DIR = "data/dataset"
    nb_images = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    deep = ird.ImageRetrieveDeep(DIR, "descriptor")
    print(nb_images, " descriptors will be creating it takes 5 hours for 812 images\n You will wait approximatively ", nb_images*20 , " seconds.")
    deep.build_descriptor_directory_using_images(812)
    print(nb_images, " descriptors created")


if __name__ == '__main__':
    build_descriptor()

