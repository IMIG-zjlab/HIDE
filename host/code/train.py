import os
import sys
from network import unet
from loss import *
from data_generator import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import losses
import tensorflow as tf
from tensorflow.keras.optimizers import *

from configuration import *

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def train(data_path, save_path, model_name='unet'):
    batch_size = 32
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    trainGene = trainGenerator(batch_size, data_path=data_path,
                               folder='train', aug_dict=aug_args, seed=1, interaction='recist')
    devGene = trainGenerator(batch_size, data_path=data_path,
                             folder='dev', aug_dict=no_aug_args, seed=1, interaction='recist')

    model = unet(input_size=(256, 256, 4))
    model.summary()

    opt=SGD(lr=4e-4, decay=1e-6, momentum=0.9, nesterov=True)
    lr_metric = get_lr_metric(opt)

    model.compile(optimizer=opt, loss={'conv2d_23': dice_coef_loss}, loss_weights={'conv2d_23': 1}, metrics=[dice_coef, lr_metric])

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_path = os.path.join(save_path, model_name+'.csv')

    csv_logger = CSVLogger(csv_path, append=True)

    if not os.path.exists( os.path.join(save_path, model_name) ):
        os.makedirs( os.path.join(save_path, model_name) )

    model.fit_generator(generator=trainGene, steps_per_epoch=int(5000/batch_size),
                        epochs=50, validation_data=devGene,
                        validation_steps=40, verbose=1,
                        callbacks=[csv_logger])

    model.save(os.path.join(save_path, model_name+'/model.h5'))

if __name__ == '__main__':
    data_path = '/data/CT'
    save_path = '/base_path/'
    train(data_path, save_path)

