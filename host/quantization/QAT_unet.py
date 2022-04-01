import tensorflow as tf
from tensorflow.keras.optimizers import *
import sys
sys.path.append('..')

from loss import *
from code.network import unet
from code.data_generator import *
from code.configuration import *
from tensorflow.keras.callbacks import *
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import losses
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from cali_gene import *


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def QAT_train(data_path, base_path,calib_path, model_path, model_name='unet'):
    batch_size = 32
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    trainGene = trainGenerator(batch_size, data_path=data_path,
                               folder='train', aug_dict=aug_args, seed = 1, interaction='recist')
    devGene = trainGenerator(batch_size, data_path=data_path,
                             folder='dev', aug_dict=no_aug_args, seed = 1, interaction='recist')

    model = unet(input_size=(256, 256, 4))
    model.load_weights(model_path, by_name=True)

    calib_dataset = load_calib_data(data_path=calib_path)

    # q_aware stands for for quantization aware.
    quantizer = vitis_quantize.VitisQuantizer(model, '8bit_tqt')
    qat_model = quantizer.get_qat_model(
        init_quant=True,
        calib_dataset=calib_dataset,
        include_cle=True,
        freeze_bn_delay=1000)

    opt = SGD(lr=4e-4, decay=1e-6, momentum=0.9, nesterov=True)
    lr_metric = get_lr_metric(opt)
    qat_model.compile(
        optimizer=opt, loss={'quant_conv2d_23_sigmoid_sigmoid': dice_coef_loss}, loss_weights={'quant_conv2d_23_sigmoid_sigmoid': 1},
        metrics=[dice_coef, lr_metric])

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    csv_path = os.path.join(base_path, model_name+'.csv')
    csv_logger = CSVLogger(csv_path, append=True)

    qat_model.fit_generator(generator=trainGene, steps_per_epoch=int(5000 / batch_size),
                        epochs=50, validation_data=devGene,
                        validation_steps=40, verbose=1,
                        callbacks=[csv_logger])

    if not os.path.exists(os.path.join(base_path, model_name)):
        os.makedirs(os.path.join(base_path, model_name))

    quantized_model = vitis_quantize.VitisQuantizer.get_deploy_model(qat_model)
    quantized_model.save(os.path.join(base_path, model_name, 'quantized.h5'))


data_path = '/data/CT'
base_path = "/QAT/"
calib_path = "/data/calibration/CT/train/"
float_model = "./float_model.h5"
QAT_train(data_path, base_path, calib_path, float_model)

