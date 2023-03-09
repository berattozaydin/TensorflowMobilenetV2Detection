import pathlib
import time

import numpy as np
import tensorflow as tf
import cv2
import glob
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
#from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
BASEPATH = "C:/Users/berat/PycharmProjects/FORarm/peopledetec/baretinsan"
NOBASEPATH = "C:/Users/berat/PycharmProjects/FORarm/peopledetec/baret"

dataset = glob.glob(BASEPATH+'/*.jpg')
nodataset = glob.glob(NOBASEPATH+'/*.jpg')
print(len(dataset))
print(len(nodataset))
dataset.sort()
nodataset.sort()
trainDataSet = dataset
noTrainDataSet = nodataset

batch_size = 32
img_height = 224
img_width = 224
training_set = tf.keras.utils.image_dataset_from_directory(trainDataSet,validation_split=0.2,subset="training",seed=123,image_size=(img_height,img_width),batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(
 trainDataSet,
 validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
className = training_set.class_names
AUTOTUNE = tf.data.AUTOTUNE
#
training_set = training_set.cache().shuffle(1024).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalize_layer = tf.keras.layers.Rescaling(1./255)
normalized = training_set.map(lambda x,y:(normalize_layer(x),y))
image_batch,labels_batch = next(iter(normalized))
firstImg = image_batch[0]


countClass = len(className)
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])
basemodel = tf.keras.applications.VGG16(weights="imagenet",include_top=False,input_tensor=tf.keras.Input(shape=(img_height,img_width,3)))
basemodel.trainable=False
x = basemodel.output
x = tf.keras.layers.Flatten()(x)
bbox = tf.keras.layers.Dense(128,activation="relu")(x)
bbox = tf.keras.layers.Dense(64,activation="relu")(bbox)
bbox = tf.keras.layers.Dense(32,activation="relu")(bbox)
bbox = tf.keras.layers.Dense(4,activation="sigmoid")(bbox)
model = tf.keras.Model(inputs=basemodel.input,outputs=bbox)
model.compile(loss="mse",optimizer="adam")

t = time.time()
export = "C:/Users/berat/PycharmProjects/TersorflowMobilenetV2/{}".format(int(t))
model.save(export)
reloadModel = tf.keras.models.load_model(export)
camera = cv2.VideoCapture(0)
s = True
while s==False:
    ret,realFrame = camera.read()
    if realFrame is None:
        continue
    frame = cv2.resize(realFrame,(224,224))
    frame = tf.reshape(frame,[224,224,3])
    img = image.img_to_array(frame)
    expanded = np.expand_dims(img,axis=0)
    prediction = reloadModel.predict(expanded)
    #print(prediction)
    cv2.imshow("Camera Screen", realFrame)
