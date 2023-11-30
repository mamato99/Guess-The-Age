import os
from tqdm import tqdm
import tensorflow as tf

import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, save_model, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from keras import applications
from keras import optimizers

from keras.utils.vis_utils import plot_model
from keras.models import load_model
from tensorflow import keras

from CustomLoss import *

base_dir = "/user/2022_va_gr01/Francesco"
dataset_dir = "/mnt/sdc1/2022_va_gr01/data"
train_dir = dataset_dir + '/train'
valid_dir = dataset_dir + '/validation'
models_dir = base_dir + '/models'

## HARDWARE LIMITATIONS ##
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        
## LOAD DATA ##
class GeneratorWrapper(tf.keras.utils.Sequence):
    def __init__(self,datagen:keras.preprocessing.image.DirectoryIterator):
        self.datagen = datagen
        self.classes = datagen.classes
    
    def __len__(self):
        return len(self.datagen)
    
    def __getitem__(self, index):
        x,y = self.datagen[index]
        return x,y/100
    
    def __next__(self):
        x,y = next(self.datagen)
        return x,y/100

BATCH_SIZE = 64
TARGET_SIZE = (224,224)

preprocess = ImageDataGenerator(
    preprocessing_function= tf.keras.applications.resnet_v2.preprocess_input,
    horizontal_flip=True,
    rotation_range=0.2
)

print('> Training set generator:')
train_generator = preprocess.flow_from_directory(train_dir,
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='sparse',
    shuffle = True)
train_generator = GeneratorWrapper(train_generator)

print('> Validation set generator:')
val_generator = preprocess.flow_from_directory(valid_dir,
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='sparse',
    shuffle = True)
val_generator = GeneratorWrapper(val_generator)
# In total: 575073

## MODEL DEFINITION ##
from keras.applications.resnet_v2 import *

model_name = 'ResNet152V2_CL2'
base_model = ResNet152V2(weights='imagenet', include_top=False)

base_model.trainable = False

model=Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(1))

# Compile the model
cl = CustomLoss()
model.compile(optimizer='adam',
              loss= cl.AAR_loss,
              run_eagerly=True,
              metrics=[cl.AAR_metric, 'mean_squared_error'])

# Print summary
model.summary()

## TRAIN only Dense ##
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint,ProgbarLogger

model_dir = models_dir + '/'+ model_name

earlystopping = EarlyStopping(monitor="val_loss", patience=10)
modelcheckpoint = ModelCheckpoint(model_dir+'_only_dense', monitor="val_loss", save_best_only=True)

epochs_fine = 3
model.fit(
    train_generator,
    steps_per_epoch= len(train_generator.classes) // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps= len(val_generator.classes) // BATCH_SIZE,
    epochs=epochs_fine,
    callbacks=[
            earlystopping, 
            modelcheckpoint,
            ])

print("\n>> END WITH DENSE LAYER <<\n")

## TRAIN 25% of backbone ##
base_model.trainable = True

perc = 75 # percentuale dei layed da freezare
freezed_layers = len(base_model.layers)*perc//100
for layer in base_model.layers[:freezed_layers]:
    layer.trainable = False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss= cl.AAR_loss,
              run_eagerly=True,
              metrics=[cl.AAR_metric, 'mean_squared_error'])

# Print summary
model.summary()

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint,ProgbarLogger

model_dir = models_dir + '/'+ model_name

earlystopping = EarlyStopping(monitor="val_loss", patience=10)
modelcheckpoint = ModelCheckpoint(model_dir, monitor="val_loss", save_freq='epoch', save_best_only=True, verbose=1) # salva dopo ogni epoca

epochs_fine = 20
model.fit(
    train_generator,
    steps_per_epoch= len(train_generator.classes) // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps= len(val_generator.classes) // BATCH_SIZE,
    epochs=epochs_fine,
    callbacks=[
            earlystopping, 
            modelcheckpoint,
            ])
