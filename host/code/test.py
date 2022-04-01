import sys
import tensorflow as tf
from network import unet
from tensorflow.keras.optimizers import *
from data_generator import *
from tensorflow.keras.callbacks import *
import os
from loss import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import losses
from configuration import *


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr

def test(data_path, base_path, model_name = 'unet', num_test_sample = 1549):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    model = unet(input_size=(256, 256, 4))
    model.load_weights(os.path.join(base_path, model_name+'/model.h5'), by_name=True)

    opt = SGD(lr=4e-4, decay=1e-6, momentum=0.9, nesterov=True)
    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt, loss={'conv2d_23': dice_coef_loss},
                  loss_weights={'conv2d_23': 1}, metrics=[dice_coef, lr_metric])

    testGene = testGenerator(test_path=data_path, interaction='recist')

    results = model.predict_generator(testGene, steps=num_test_sample, verbose = 1)
    save_path = os.path.join(base_path, model_name+'/test_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saveResult(save_path, results)

if __name__ == '__main__':
    data_path = '/data/CT/test'
    base_path = '/base_path/'
    test(data_path, base_path)
