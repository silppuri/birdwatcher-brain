import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt

from time import time
from birdwatcher.generators import Generator, compose, stft, amplitude_to_db, read_audio, reshape, normalize_image, noise
from keras.layers.advanced_activations import PReLU
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, GlobalAveragePooling2D, add, Reshape
from keras.models import Model
from keras import backend as K
from kapre.time_frequency import Spectrogram, Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise

num_epochs = 30
image_height = 257
image_width = 515
train_length = 5464
test_length = 1389

classes = np.load('data/classes.npy')

train_generator = Generator('data/train.tfrecord', parser=compose(noise, read_audio))
test_generator = Generator('data/test.tfrecord', parser=read_audio)

callbacks = [
    TensorBoard(log_dir="logs/birdwatcher-{}".format(time()), write_images=True),
    ModelCheckpoint("models/birdwatcher.h5")
]

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = PReLU(name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = PReLU(name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = PReLU(name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

sr = 44100
def SqueezeNet(input_tensor=None, input_shape=(1, 44100*3), classes=len(classes)):
    inputs = Input(shape=input_shape)
    x = Spectrogram(n_dft=512)(inputs)
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(x)
    x = PReLU(name='prelu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    # first simple bypass
    fire2 = fire_module(x, fire_id=2, squeeze=16, expand=64)
    fire3 = fire_module(fire2, fire_id=3, squeeze=16, expand=64)
    x = add([fire2, fire3])
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)

    # second simple bypass
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
    fire5 = fire_module(maxpool1, fire_id=5, squeeze=32, expand=128)
    x = add([maxpool1, fire5])

    # third simple bypass
    fire6 = fire_module(x, fire_id=6, squeeze=48, expand=192)
    fire7 = fire_module(fire6, fire_id=7, squeeze=48, expand=192)
    x = add([fire6, fire7])
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    fire9 = fire_module(maxpool2, fire_id=9, squeeze=64, expand=256)
    x = add([maxpool2, fire9])

    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = PReLU(name='prelu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(inputs, out, name='squeezenet')
    return model

model = SqueezeNet()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
model.summary()

model.fit_generator(train_generator.next_batch(),
        callbacks=callbacks,
        epochs=num_epochs,
        steps_per_epoch=train_length // 32,
        validation_steps=test_length // 32,
        validation_data=test_generator.next_batch())
