'''
Keras (Tensorflow) implementation of model from "DeXpression: Deep Convolutional Neural Network for Expression Recognition" https://arxiv.org/pdf/1509.05371v2.pdf.

By Minkesh Asati
'''

import tensorflow as tf 
from tensorflow._api.v1.keras import preprocessing
from tensorflow._api.v1.keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, ReLU, BatchNormalization, concatenate, Flatten, GlobalAveragePooling2D
from tensorflow._api.v1.keras.models import Model
import numpy

inputs = Input(shape=(128, 128, 3))
pre_net = Conv2D(64, (7, 7), strides=(2, 2))(inputs)
pre_net = ReLU()(pre_net)
pre_net = MaxPool2D(pool_size=(3, 3), strides=(1, 1))(pre_net)
pre_net = BatchNormalization()(pre_net)

def feature_extractor(input_net):
    net_1 = Conv2D(96, 1, 1)(input_net)
    net_1 = ReLU()(net_1)
    net_1 = Conv2D(208, 3, 1)(net_1)
    net_1 = ReLU()(net_1)

    net_2 = MaxPool2D(3, 1)(input_net)
    net_2 = Conv2D(64, 1, 1)(net_2)
    net_2 = ReLU()(net_2)

    concat = concatenate(inputs=[net_1, net_2], axis=3)
    pooling_out = MaxPool2D(3, 2)(concat)

    return pooling_out

feat_ex_1 = feature_extractor(pre_net)
feat_ex_2 = feature_extractor(feat_ex_1)

net = GlobalAveragePooling2D()(feat_ex_2)
output = Dense(units=11, activation='softmax')(net)

model = Model(inputs=inputs, outputs=output)
#model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
