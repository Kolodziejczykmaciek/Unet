#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=68HR_eyzk00


"""
@author: Sreenivas Bhattiprolu
"""
import sys
import time
from pycocotools.coco import COCO
import tensorflow as tf
import os
import numpy as np
import random
import skimage.io as io
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use('TkAgg')

from aerial_dataSet_parser import aerialDataPrepare
from aerial_dataSet_parser import aerialValidationData

seed = 42
np.random.seed = seed


# X_train, Y_train = aerialDataPrepare()
# print("Dl listy X_train: {} \nDl listy Y_train: {}".format(len(X_train), len(Y_train)))

X_val = aerialValidationData()

# for i in range(20):
#
#     ix = random.randint(0, len(X_val) - 1)
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.suptitle('Zdjecie z listy numer:  {}'.format(ix), fontsize=15)
#
#     im1 = ax1.imshow(X_val[ix])
#     ax1.set_title('Zdj, którego sieć nie zna')
#     im2 = ax2.imshow(np.squeeze(X_val[ix]))
#     ax2.set_title('Maska binarna z {}'.format("pies"))
#     plt.show()

# IMG_WIDTH = 512
# IMG_HEIGHT = 512
# IMG_CHANNELS = 3
#
# #Build the model
# inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
# s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
#
# #Contraction path
# c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
# c1 = tf.keras.layers.Dropout(0.1)(c1)
# c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
# p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
#
# c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
# c2 = tf.keras.layers.Dropout(0.1)(c2)
# c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
# p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
#
# c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
# c3 = tf.keras.layers.Dropout(0.2)(c3)
# c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
# p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
#
# c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
# c4 = tf.keras.layers.Dropout(0.2)(c4)
# c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
# p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
#
# c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
# c5 = tf.keras.layers.Dropout(0.3)(c5)
# c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
#
# #Expansive path
# u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
# u6 = tf.keras.layers.concatenate([u6, c4])
# c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
# c6 = tf.keras.layers.Dropout(0.2)(c6)
# c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
#
# u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
# u7 = tf.keras.layers.concatenate([u7, c3])
# c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
# c7 = tf.keras.layers.Dropout(0.2)(c7)
# c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
#
# u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
# u8 = tf.keras.layers.concatenate([u8, c2])
# c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
# c8 = tf.keras.layers.Dropout(0.1)(c8)
# c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
#
# u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
# u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
# c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
# c9 = tf.keras.layers.Dropout(0.1)(c9)
# c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
#
# outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
#
# model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
#
# ################################
# # Modelcheckpoint
# checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)
#
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
#     tf.keras.callbacks.TensorBoard(log_dir='logs')]
#
# results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=100, callbacks=callbacks)
# model.save(r"C:\Users\macie\PycharmProjects\pythonProject\AerialDataSet_p2_bs10_4ktrainimg")

#############################################################################################################
print("Loading model from save...")
model = tf.keras.models.load_model("AerialDataSet_p2_bs10_4ktrainimg")
print("Model has been loaded!")
X_val = X_val[0:1]
try:
    time_start = time.time()
    print('Segmentaion of {} images in progres...'.format(len(X_val)))
    predition_Xval = model.predict(X_val, verbose=1)
    time_stop = time.time()
    print("trwalo to: {}".format(time_stop-time_start))
except MemoryError:
    print("Za mało ramu !!!")
    sys.exit()

for n in range(1):
    ix = random.randint(0, len(X_val)-1)
    # Perform a sanity check on some random training samples
    # Display some validation images
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Segmentacja zdjeć geo, rand={}'.format(ix), fontsize=15)

    im1 = ax1.imshow(np.squeeze(X_val[0].astype(np.uint8)))
    ax1.set_title('Zdj, którego sieć nie zna')
    # im2 = ax2.imshow(np.squeeze(Y_val[ix]))
    # ax2.set_title('Maska binarna z {}'.format(classes))
    im2 = ax2.imshow(np.squeeze(predition_Xval[0]))
    ax2.set_title('to wyrzuca siec')
    plt.show()



