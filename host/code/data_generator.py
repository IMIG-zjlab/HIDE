from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import skimage.io as io
import cv2

filelist = []

def adjustData(img, recist, label):
    input = np.concatenate((img, recist), axis=3)
    input = input / 255
    label = label /255

    label[label > 0.5] = 1
    label[label <= 0.5] = 0
    return ( [input] , [label])


def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        # img[img<0.5]=0
        # img[img>=0.5]=1

        io.imsave(os.path.join(save_path,"%s"%filelist[i]),img)


def trainGenerator(batch_size, data_path, folder, aug_dict, seed = 1, interaction='recist'):
    patch_gene = ImageDataGenerator(**aug_dict).flow_from_directory(
        data_path, classes=[folder], batch_size=batch_size,
        save_prefix=None, class_mode=None, seed=seed)

    recist_gene = ImageDataGenerator(**aug_dict).flow_from_directory(
        data_path.replace('CT', interaction), classes=[folder], color_mode='grayscale',
        batch_size=batch_size, class_mode=None, seed=seed)

    label_gene = ImageDataGenerator(**aug_dict).flow_from_directory(
        data_path.replace('CT', 'label'), classes=[folder], color_mode='grayscale',
        batch_size=batch_size, class_mode=None, seed=seed)

    train_generator = zip(patch_gene, recist_gene, label_gene)

    while True:
        for (img, recist,label) in train_generator:
            input, gt = adjustData(img, recist, label)
            yield (input, gt)


def testGenerator(test_path, interaction='recist'):
    file_name = os.listdir(test_path)
    while True:
        for i in range(len(file_name)):
            filelist.append(file_name[i])
            img = cv2.imread(os.path.join(test_path, file_name[i]))

            recist = cv2.imread(os.path.join(test_path.replace('CT', interaction),
                                                   file_name[i]), cv2.IMREAD_GRAYSCALE)

            input = np.concatenate((img, recist[:, :, np.newaxis]), axis=2)
            input = input / 255.
            input = input[np.newaxis, :, :, :]

            yield ([input])