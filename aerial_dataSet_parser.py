'''
---Data generator---
'''
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

np.random.seed(1000)

def aerialDataPrepare():

    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    IMG_CHANNELS = 3

    X_train = np.zeros((4000, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((4000, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    image_directory = 'aerial_dataSet/NEW2-AerialImageDataset/AerialImageDataset/train/images/'
    mask_directory = 'aerial_dataSet/NEW2-AerialImageDataset/AerialImageDataset/train/gt/'

    cords = (0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4487)
    #cords256 = (0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304)
    #cords128 = (0, 128, 256)
    image_resolution = 512

    train_images = os.listdir(image_directory)
    random.shuffle(train_images)
    n = 0
    for i, image_name in enumerate(train_images):
        print("Interation no.{}".format(i))
        if i == 40: break
        if '.tif' in image_name:
            img = cv2.imread(image_directory + image_name)
            label = cv2.imread(mask_directory + image_name)

            # sub_image = 0
            for i in cords:
                for j in cords:
                    new_img = img[i:i + image_resolution, j:j + image_resolution]
                    X_train[n] = np.array(new_img)

                    new_label = label[i:i + image_resolution, j:j + image_resolution]
                    new_label = new_label[:, :, 0:1]
                    new_label = new_label / 255
                    Y_train[n] = new_label
                    n+=1
                    # sub_image+=1
                    # new_img_name = image_name[0:-4]+'_{}'.format(sub_image)
                    # cv2.imwrite(img_dir+new_img_name+".tif", new_img)

    return X_train, Y_train

def aerialValidationData():

    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    IMG_CHANNELS = 3

    num_val_img = 100

    X_val = np.zeros((num_val_img, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

    image_directory = 'aerial_dataSet/NEW2-AerialImageDataset/AerialImageDataset/test/images/'

    cords = (0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4487)
    #cords256 = (0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304)
    #cords128 = (0, 128, 256)
    image_resolution = 512

    validation_images = os.listdir(image_directory)
    random.shuffle(validation_images)
    print(validation_images)
    n = 0

    for i, image_name in enumerate(validation_images):
        print("Interation no.{}".format(i))
        if i == num_val_img/100: break
        if '.tif' in image_name:
            img = cv2.imread(image_directory + image_name)

            # sub_image = 0
            for i in cords:
                for j in cords:
                    new_img = img[i:i + image_resolution, j:j + image_resolution]
                    X_val[n] = np.array(new_img)
                    n+=1
    return X_val

