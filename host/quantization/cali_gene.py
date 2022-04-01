from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2

filelist = []

def load_calib_data(data_path, interaction='recist'):
    files = os.listdir(data_path)
    print(len(files))
    for file in files:
        img_local = cv2.imread(os.path.join(data_path, file))
        recist_local = cv2.imread(os.path.join(data_path.replace('CT', interaction),
                                               file), cv2.IMREAD_GRAYSCALE)

        input = np.concatenate((img_local, recist_local[:, :, np.newaxis]), axis=2)
        input = input / 255.

        filelist.append(input)

    files_array = np.array(filelist)
    print(files_array.shape)
    return files_array