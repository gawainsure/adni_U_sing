from __future__ import print_function

from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
import pickle

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

# define cost function
def dice_coef(y_true, y_pred, smooth= 1e-8):
    #y_true = K.argmax(y_true, axis=-1) # NOT for loss function use!!!
    #y_pred = K.argmax(y_pred, axis=-1) # NOT for loss function use!!!
    intersection = K.sum(y_true * y_pred, axis=[1,2,3] ) #[1,2,3]
    union = K.sum(y_true * y_true , axis=[1,2,3]) + K.sum(y_pred * y_pred, axis=[1,2,3] ) #[1,2,3]
   # return K.mean( (2 * intersection ) / (union), axis=0)
    return K.mean( (2 * intersection + smooth) / (union + smooth), axis=0)

   # The following cost funciton definition doesn't work because of one tricky 'transpose'-incompatible-issue of y_true and y_pred
   # y_true_f = K.batch_flatten(y_true)
   # y_pred_f = K.batch_flatten(y_pred)
   # intersection = K.batch_dot(y_true_f, K.transpose(y_pred_f), axes=1) # batch-axis=0
   # union = K.batch_dot(y_true_f, K.transpose(y_true_f), axes=1)+ K.batch_dot(y_pred_f, K.transpose(y_pred_f), axes=1)
   # return K.mean( (2 * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

#\/-------Control-Panel-------\/

# 1. Directories
data_dir = './train_lt_256/'
model_dir = './model_2d/'
model_name = 'model_2d_test'
history_dir = './history_2d/'
history_name = 'HistoryDict_test' #!!!!!Remember to double-check this name!!!!! 

# 2. Input parameters
im_height = 176
im_width = 176

# 3. Dataset spliting
num_tr = 20 #300
num_ts = 5 #97

# 4. U-net compiling parameters
kernel_size_handle = 3
num_filter_handle = 16
dropout_handle = 0.10
batchnorm_handle = True
conv_actv = 'linear' #linear; sigmoid
loss_function = dice_coef_loss  #dice_coef_loss; 'categorical_crossentropy'
performance_metrics = [dice_coef, 'categorical_accuracy']

# 4.1 Optimizer:
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# 5. U-net fitting parameters
batch_size = 16
epochs = 5
validation_handle = 0.3
checkpoint = keras.callbacks.ModelCheckpoint(model_dir+model_name,\
                                             monitor='categorical_accuracy',\
                                             verbose=0,\
                                             save_weights_only=False,\
                                             save_best_only=True,\
                                             mode='max',\
                                             period = 30)

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',\
                                          min_delta=1e-5,\
                                          patience=50,\
                                          verbose=0,\
                                          mode='min',\
                                          baseline=None,\
                                          restore_best_weights=True)

callbacks_list = [checkpoint, earlystop]

#/\-------Control-Panel-------/\

# define convolutional block
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = SeparableConv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), 
                        strides=(1, 1), padding='same', data_format='channels_last', 
                        dilation_rate=(1, 1), depth_multiplier=2, 
                        activation= conv_actv, use_bias=True, 
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
                        activation= conv_actv, use_bias=True, 
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
def get_unet(input_img, n_filters=num_filter_handle, dropout=dropout_handle, batchnorm=batchnorm_handle):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=kernel_size_handle, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2), strides=None, padding='same', data_format='channels_last') (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=kernel_size_handle, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2), strides=None, padding='same', data_format='channels_last') (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=kernel_size_handle, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2), strides=None, padding='same', data_format='channels_last') (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=kernel_size_handle, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2), strides=None, padding='same', data_format='channels_last') (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=kernel_size_handle, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (kernel_size_handle, kernel_size_handle), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=kernel_size_handle, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (kernel_size_handle, kernel_size_handle), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=kernel_size_handle, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (kernel_size_handle, kernel_size_handle), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=kernel_size_handle, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (kernel_size_handle, kernel_size_handle), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=kernel_size_handle, batchnorm=batchnorm)
    
    u10 = Conv2D(6, (1, 1), activation='sigmoid',strides=(1,1), padding='same') (c9) #'softmax' Pay attention to the activation function here!!
    outputs = Activation("softmax")(u10)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# initiate system directories
if not os.path.exists(model_dir):
       os.mkdir(model_dir)

if not os.path.exists(history_dir):
       os.mkdir(history_dir)

# compile input parameters
input_img = Input((im_height, im_width, 5), name='img')

# get model!
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

# compile model!
model.compile(optimizer=adam, loss=loss_function, metrics= performance_metrics)
# losses: 'categorical_crossentropy'; dice_coef_loss;'mean_squared_error'; 'binary_crossentropy'
# keras.metrics.categorical_accuracy; dice_coef; 'mae'; 'binary_accuracy'

model.summary()

# load the data, split between train and validation sets

#!!!!!!!!!(uncomment the following paragraph ONLY  when formally training)!!!!!!!!!!!!!!
input_memmap = np.memmap(data_dir+'train_2d_256_lt', dtype='float64', mode='r', shape=(196*300, 176, 176, 5))
input_img_tr = np.zeros((196*num_tr, 176, 176, 5))
input_img_tr = input_memmap[0:196*num_tr,:,:,:] 
print('X-shape training:'+str(input_img_tr.shape))
output_memmap = np.memmap(data_dir+'train_seg_256_onehot', dtype='float64', mode='r', shape=(196*300, 176, 176, 6))
output_seg_tr = np.zeros((196*num_tr, 176, 176, 6))
output_seg_tr[:,:,:,:] = output_memmap[0:196*num_tr, :, :, :]
print('y-shape training:'+str(output_seg_tr.shape))
#del input_memmap
#del output_memmap

input_memmap = np.memmap(data_dir+'test_2d_256_lt', dtype='float64', mode='r', shape=(196*97, 176, 176, 5))
input_img_ts = np.zeros((196*num_ts, 176, 176, 5))
input_img_ts = input_memmap[0:196*num_ts,:,:,:] 
print('X-shape testing:'+str(input_img_ts.shape))
output_memmap = np.memmap(data_dir+'test_seg_256_onehot', dtype='float64', mode='r', shape=(196*97, 176, 176, 6)) 
output_seg_ts = np.zeros((196*num_ts, 176, 176, 6))
output_seg_ts[:,:,:,:] = output_memmap[0:196*num_ts, :, :, :]
print('y-shape testing:'+str(output_seg_ts.shape))
del input_memmap
del output_memmap

x_train, x_valid, y_train, y_valid = train_test_split(input_img_tr, output_seg_tr, test_size=validation_handle, random_state=42)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
 
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'test samples')


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_data=(x_valid, y_valid))

with open(history_dir+history_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

score = model.evaluate(x_train, y_train, verbose=0)
print('Training loss:', score[0])
print('Training Dice Coeff.:', score[1])
print('Training Categorical CrossEntropy:', score[2])

score = model.evaluate(x_valid, y_valid, verbose=0)
print('Validation loss:', score[0])
print('Validation Dice Coeff.:', score[1])
print('Validation Categorical CrossEntropy:', score[2])

score = model.evaluate(input_img_ts, output_seg_ts, verbose=0)
print('Testing loss:', score[0])
print('Testing Dice Coeff.:', score[1])
print('Testing Categorical CrossEntropy:', score[2])

