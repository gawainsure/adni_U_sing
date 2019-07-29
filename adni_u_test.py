from __future__ import print_function

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.convolutional import SeparableConv2D, Conv2DTranspose, Conv2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.optimizers import Adam

from keras import backend as K

batch_size = 8
epochs = 3

# define convolutional block
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = SeparableConv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), 
                        strides=(1, 1), padding='same', data_format='channels_last', 
                        dilation_rate=(1, 1), depth_multiplier=2, 
                        activation= 'linear', use_bias=True, 
                        depthwise_initializer='glorot_uniform', 
                        pointwise_initializer='glorot_uniform', 
                        bias_initializer='zeros', 
                        depthwise_regularizer=None, pointwise_regularizer=None, #?
                        bias_regularizer=None, activity_regularizer=None, #?
                        depthwise_constraint=None, pointwise_constraint=None, 
                        bias_constraint=None)(input_tensor)
    
    if batchnorm:
        x = BatchNormalization(axis=-1)(x)
    x = Activation("elu")(x)
    # second layer
    x = SeparableConv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), 
                        strides=(1, 1), padding='same', data_format='channels_last', 
                        dilation_rate=(1, 1), depth_multiplier=2, 
                        activation= 'linear', use_bias=True, 
                        depthwise_initializer='glorot_uniform', 
                        pointwise_initializer='glorot_uniform', 
                        bias_initializer='zeros', 
                        depthwise_regularizer=None, pointwise_regularizer=None, #?
                        bias_regularizer=None, activity_regularizer=None, #?
                        depthwise_constraint=None, pointwise_constraint=None, 
                        bias_constraint=None)(x)
    if batchnorm:
        x = BatchNormalization(axis=-1)(x)
    x = Activation("elu")(x)
    return x

# generate u_net architecture
def get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2), strides=None, padding='same', data_format='channels_last') (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2), strides=None, padding='same', data_format='channels_last') (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2), strides=None, padding='same', data_format='channels_last') (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2), strides=None, padding='same', data_format='channels_last') (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='relu') (c9) #'softmax' Pay attention to the activation function here!!
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# define cost function
def dice_coef(y_true, y_pred, smooth= 1e-8):
    # y_pred[y_pred<0]=0
    # y_pred[y_pred>5]=5
    intersection = K.sum(y_true * y_pred, axis=[1,2,3] )
    union = K.sum(y_true * y_true , axis=[1,2,3]) + K.sum(y_pred * y_pred, axis=[1,2,3] )
    return K.mean( (2 * intersection + smooth) / (union + smooth), axis=0)
  
   # y_true_f = K.batch_flatten(y_true)
   # y_pred_f = K.batch_flatten(y_pred)
   # intersection = K.batch_dot(y_true_f, K.transpose(y_pred_f), axes=1) # batch-axis=0
   # union = K.batch_dot(y_true_f, K.transpose(y_true_f), axes=1)+ K.batch_dot(y_pred_f, K.transpose(y_pred_f), axes=1)
   # return K.mean( (2 * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# compile input parameters
im_height = 176
im_width = 176

input_img = Input((im_height, im_width, 5), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef, 'mae'])
# 'categorical_crossentropy'; dice_coef_loss;'mean_squared_error'
# keras.metrics.categorical_accuracy; dice_coef; 'mae'
model.summary()

# the data, split between train and validation sets
#data_dir = '/media/rajlab/DATASETS/copy_ADNI/adni_u_train/train_2d_256/'
data_dir = '/home/gavingao/Documents/Jupyter/adni_U_sing/'

input_memmap = np.memmap(data_dir+'test_2d_256_lt', dtype='float32', mode='r', shape=(196*97, 176, 176, 5))
#input_img_ts = np.zeros(input_memmap.shape)
#input_img_ts[:] = input_memmap[:]
input_img_ts = np.zeros((196*20, 176, 176, 5))
input_img_ts = input_memmap[0:20*196,:,:,:] 
print('X-shape:'+str(input_img_ts.shape))

output_memmap = np.memmap(data_dir+'test_seg_256_lt', dtype='float32', mode='r', shape=(196*97, 176, 176))
#output_seg_ts = np.zeros(output_memmap.shape)
#output_seg_ts[:] = output_memmap[:] 
output_seg_ts = np.zeros((196*20, 176, 176,1))
output_seg_ts[:,:,:,0] = output_memmap[0:20*196, :, :]
print('y-shape:'+str(output_seg_ts.shape))

x_train, x_valid, y_train, y_valid = train_test_split(input_img_ts, output_seg_ts, test_size=0.20, random_state=42)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
 
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'test samples')


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_valid, y_valid))

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test Dice Coeff.:', score[1])
