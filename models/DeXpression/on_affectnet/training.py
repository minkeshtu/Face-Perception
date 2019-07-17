'''
Training the DeXpression-model on Affectnet-dataset.
'''

import os
import sys
import tensorflow as tf 
from tensorflow._api.v1.keras.preprocessing import image
from tensorflow._api.v1.keras.models import Model
from tensorflow._api.v1.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import cv2 as cv
import h5py


ROOT_DIR = os.path.abspath('')
dataset_dir = os.path.join(ROOT_DIR, 'datasets/Affectnet')
sys.path.append(dataset_dir)
import data_loading as data

model_dir = os.path.join(ROOT_DIR, 'models/DeXpression')
sys.path.append(model_dir)
from model import model

image_size = (128, 128)
batch_size = 300

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

train_datagen = image.ImageDataGenerator(rescale=1./255)
val_datagen = image.ImageDataGenerator(rescale=1./255)

dataset_imgs_dir = data.dataset_dir

train_generator = train_datagen.flow_from_dataframe(data.train_dataframe, directory=f'{dataset_imgs_dir}Manually_Annotated_Images', x_col='subDirectory_filePath', y_col='expression', target_size=image_size, batch_size=batch_size, drop_duplicates=False)
val_generator = val_datagen.flow_from_dataframe(data.val_dataframe, directory=f'{dataset_imgs_dir}Manually_Annotated_Images', x_col='subDirectory_filePath', y_col='expression', target_size=image_size, batch_size=batch_size)

model.fit_generator(train_generator, steps_per_epoch=414798/300, epochs=38, callbacks=[early_stopping], validation_data=val_generator, workers=12)

model.save('model_epoch_38_image_size_128_batch_300_1.h5')