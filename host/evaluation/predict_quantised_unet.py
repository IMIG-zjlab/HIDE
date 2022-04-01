import sys
sys.path.append('..')

import tensorflow as tf
from tensorflow.keras.optimizers import *
from code.data_generator import *
from code.configuration import *
from code.loss import *
from tensorflow.keras.callbacks import *
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import losses
from tensorflow_model_optimization.quantization.keras import vitis_quantize

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def test(data_path, base_path, model_name = 'unet',  num_test_sample = 1549):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    model = tf.keras.models.load_model("/data/xilinx_challenge/tensorflow-code/QAT/unet/quantized.h5")

    opt = SGD(lr=4e-4, decay=1e-6, momentum=0.9, nesterov=True)
    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt, loss={'quant_conv2d_23_sigmoid_sigmoid': dice_coef_loss},
                  loss_weights={'quant_conv2d_23_sigmoid_sigmoid': 1}, metrics=[dice_coef, lr_metric])

    testGene = testGenerator(test_path=data_path, interaction='recist')

    results = model.predict_generator(testGene, steps=num_test_sample, verbose=1)
    save_path = os.path.join(base_path, model_name+'/test_quantised_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saveResult(save_path, results)

data_path = '/data/CT/test'
base_path = '/base_path/'
test(data_path, base_path)
