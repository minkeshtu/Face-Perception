import tensorflow as tf 
from tensorflow._api.v1.keras import preprocessing
from tensorflow._api.v1.keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, ReLU, BatchNormalization, concatenate, Flatten, GlobalAveragePooling2D
from tensorflow._api.v1.keras.models import Model

# This returns a tensor
inputs = Input(shape=(32, 32, 3))
pre_net = Conv2D(64, (7, 7), strides=(2, 2))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()