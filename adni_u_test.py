'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.

Shamelessly copied/pasted from keras repository
'''

from __future__ import print_function

import keras
from keras.models import Model, load_model
from keras.layers.core import Input, BatchNormalization, Activation, Dense, Droupout 
from keras.layers.pooling import MaxPooling2D, GlobalMaxpool2D
from keras.layers.convolutional import SeparableConv2D, Conv2DTranspose, Conv2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.optimizers import Adam

from kearas import backend as K

batch_size = 128
epochs = 500

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
        x = BatchNormalization()(x)
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
        x = BatchNormalization()(x)
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
    p3 = MaxPooling2D((2, 2), strides=None, padding='', data_format='channels_last') (c3)
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
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# define cost function
def dice_coef(y_true, y_pred, smooth= 1e-8):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.batch_dot(y_true_f, K.transpose(y_pred_f), axes=1) # batch-axis=0
    union = K.batch_dot(y_true_f, K.transpose(y_true_f), axes=1)+ K.batch_dot(y_pred_f, K.transpose(y_pred_f), axes=1)
    return K.mean( (2 * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# compile input parameters
im_height = 176
im_width = 176

input_img = Input((im_height, im_width, 5), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

#history = model.fit(x_train, y_train,
#                    batch_size=batch_size,
#                    epochs=epochs,
#                    verbose=1,
#                    validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
