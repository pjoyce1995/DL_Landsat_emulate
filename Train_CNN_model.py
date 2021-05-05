
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Commonly used modules
import numpy as np
import os
import sys

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import IPython
from six.moves import urllib

import ee
import sys
import os

ee.Authenticate()
ee.Initialize()

from google.colab import drive

drive.mount('/content/gdrive')
os.chdir('/content/gdrive/MyDrive/Colab_data')

bands_use=[0,1,2,3,4,6]

from keras.layers import Conv2D, ReLU, BatchNormalization, UpSampling2D, Lambda
from keras.layers import Input, Flatten, Dense, concatenate
from keras.models import Model
import keras
import numpy as np


def make_model(inputs):
  down1=Conv2D(64, (3,3), padding='same')(inputs)
  down1=BatchNormalization()(down1)
  down1=ReLU()(down1)
  down1=Conv2D(64, (3,3), padding='same')(down1)
  down1=BatchNormalization()(down1)
  down1=ReLU()(down1)

  up1=concatenate([down1, inputs], axis=3)
  up1=Conv2D(64, (3,3), padding='same')(up1)
  up1=BatchNormalization()(up1)
  up1=ReLU()(up1)
  up1=Conv2D(64, (3,3), padding='same')(up1)
  up1=BatchNormalization()(up1)
  up1=ReLU()(up1)

  up2=concatenate([up1, inputs], axis=3)
  up2=Conv2D(64, (3,3), padding='same')(up2)
  up2=BatchNormalization()(up2)
  up2=ReLU()(up2)
  up2=Conv2D(64, (3,3), padding='same')(up2)
  up2=BatchNormalization()(up2)
  up2=ReLU()(up2)

  classify_norm=(Conv2D(filters = len(bands_use),kernel_size = (1, 1),activation = "sigmoid", dtype ='float32'))(up2)

  classify_norm_red= Lambda(lambda x: x[:,:,:,0:1])(classify_norm)
  classify_norm_green = Lambda(lambda x: x[:,:,:,1:2])(classify_norm)
  classify_norm_blue= Lambda(lambda x: x[:,:,:,2:3])(classify_norm)
  classify_norm_nir=Lambda(lambda x: x[:,:,:,3:4])(classify_norm)
  classify_norm_swir_1=Lambda(lambda x: x[:,:,:,4:5])(classify_norm)
  classify_norm_swir_2=Lambda(lambda x: x[:,:,:,5:6])(classify_norm)

  red256=UpSampling2D(size=(2, 2),interpolation='nearest')(classify_norm_red)
  green256=UpSampling2D(size=(2, 2),interpolation='nearest')(classify_norm_green)
  blue256=UpSampling2D(size=(2, 2),interpolation='nearest')(classify_norm_blue)
  nir256=UpSampling2D(size=(2, 2),interpolation='nearest')(classify_norm_nir)
  swir1_256=UpSampling2D(size=(2, 2),interpolation='nearest')(classify_norm_swir_1)
  swir2_256=UpSampling2D(size=(2, 2),interpolation='nearest')(classify_norm_swir_2)

  im256=concatenate([red256,green256,blue256,nir256,swir1_256,swir2_256], axis=3) #swir2_256
  im256=Conv2D(64, (3,3), padding='same')(im256)
  im256=BatchNormalization()(im256)
  im256=ReLU()(im256)
  im256=Conv2D(64, (3,3), padding='same')(im256)
  im256=BatchNormalization()(im256)
  im256=ReLU()(im256)

  classify_pan=(Conv2D(filters = 1,kernel_size = (1, 1),activation = "sigmoid", dtype ='float32'))(im256)
  model = Model(inputs=[inputs], outputs=[classify_norm,classify_pan])
  return model

############################################
#### model now
############################################


model_shape=Input((128, 128, len(bands_use)))
model = make_model(model_shape)
model.compile(optimizer='rmsprop', 
                  loss='mse',
                  metrics=['mae'], loss_weights=[1,1./(len(bands_use))])

def get_training_data(batch_size):
  """
  GENERATOR FOR TRAINING
  """
  while True:
    still_leftover_file=False #IF WE ARE HALFWAY THROUGH MAKING THE ARRAY (WE MAKE THIS TRUE WHEN WE FINISH ADDING TO ARRAY BEFORE IT'S LARGE ENOUGH)
    #2,15,16,17,18,22,24,30,33,34
    for batch in range(35):#[2,15,16,17,18,22,24,30,33,34]:#[2,15,16,17,18,22,24,30,33,34]:#range(35): 
      string_save_1='IM1_RUNNING_'+ str(batch)+'.npy'
      string_save_2='IM2_RUNNING_'+ str(batch)+'.npy'
      string_save_pan='IMpan_RUNNING_'+ str(batch)+'.npy'

      ALL_im_pan=np.load(string_save_pan)
      ALL_im1=np.load(string_save_1)
      ALL_im2=np.load(string_save_2)
      indexes_all=range(np.shape(ALL_im1)[0])
      indexes_train = [item for index, item in enumerate(indexes_all) if (index) % 10 != 0]
      

      train_images=np.stack([ALL_im1[q] for q in indexes_train],axis=0)[:,:,:,bands_use]
      ALL_im1=None
      train_labels_norm=np.stack([ALL_im2[q] for q in indexes_train],axis=0)[:,:,:,bands_use]
      ALL_im2=None
      train_labels_pan=np.stack([ALL_im_pan[q] for q in indexes_train],axis=0)[:,:,:,:]
      ALL_im_pan=None

      position_in_file=0
      len_file=np.shape(train_images)[0]
      while position_in_file<len_file-batch_size:
        if still_leftover_file==True:
          x_train=train_images[0:remainder,:,:,:]
          y_train_norm=train_labels_norm[0:remainder,:,:,:]
          y_train_pan=train_labels_pan[0:remainder,:,:,:]
          still_leftover_file=False
        else:
          x_train=train_images[position_in_file:position_in_file+batch_size,:,:,:]
          y_train_norm=train_labels_norm[position_in_file:position_in_file+batch_size,:,:,:]
          y_train_pan=train_labels_pan[position_in_file:position_in_file+batch_size,:,:,:]

        position_in_file+=batch_size
        yield(x_train,[y_train_norm,y_train_pan])

      #GETTING THE LAST FEW BITS OF THE FILE
      x_train=train_images[position_in_file:len_file,:,:,:]
      y_train_norm=train_labels_norm[position_in_file:len_file,:,:,:]
      y_train_pan=train_labels_pan[position_in_file:len_file,:,:,:]
      remainder=batch_size - (len_file-position_in_file)
      still_leftover_file=True

def get_testing_data(batch_size):
  """
  GENERATOR FOR TESTING
  """
  while True:
    still_leftover_file=False
    for batch in range(35): 
      string_save_1='IM1_RUNNING_'+ str(batch)+'.npy'
      string_save_2='IM2_RUNNING_'+ str(batch)+'.npy'
      string_save_pan='IMpan_RUNNING_'+ str(batch)+'.npy'

      ALL_im_pan=np.load(string_save_pan)
      ALL_im1=np.load(string_save_1)
      ALL_im2=np.load(string_save_2)

      #ALL_im2=np.delete(ALL_im2,(5),axis=3)
      #ALL_im1=np.delete(ALL_im1,(5),axis=3)

      indexes_all=range(np.shape(ALL_im1)[0])
      indexes_train = [item for index, item in enumerate(indexes_all) if (index) % 10 != 0]
      indexes_test = [item for index, item in enumerate(indexes_all) if (index) % 10 == 0]
      
      test_images=np.stack([ALL_im1[q] for q in indexes_test],axis=0)[:,:,:,bands_use]
      ALL_im1=None
      test_labels_norm=np.stack([ALL_im2[q] for q in indexes_test],axis=0)[:,:,:,bands_use]
      ALL_im2=None
      test_labels_pan=np.stack([ALL_im_pan[q] for q in indexes_test],axis=0)[:,:,:,:]
      ALL_im_pan=None

      position_in_file=0
      len_file=np.shape(test_images)[0]
      while position_in_file<len_file-batch_size:
        if still_leftover_file==True:
          x_test=test_images[0:remainder,:,:,:]
          y_test_norm=test_labels_norm[0:remainder,:,:,:]
          y_test_pan=test_labels_pan[0:remainder,:,:,:]
          still_leftover_file=False
        else:
          x_test=test_images[position_in_file:position_in_file+batch_size,:,:,:]
          y_test_norm=test_labels_norm[position_in_file:position_in_file+batch_size,:,:,:]
          y_test_pan=test_labels_pan[position_in_file:position_in_file+batch_size,:,:,:]

        position_in_file+=batch_size
        yield(x_test,[y_test_norm,y_test_pan])

      #GETTING THE LAST FEW BITS OF THE FILE
      x_test=test_images[position_in_file:len_file,:,:,:]
      y_test_norm=test_labels_norm[position_in_file:len_file,:,:,:]
      y_test_pan=test_labels_pan[position_in_file:len_file,:,:,:]
      remainder=batch_size - (len_file-position_in_file)
      still_leftover_file=True


###TRAIN CLASSICAL REGRESSION MODEL

import pandas as pd
from sklearn import linear_model

batch_size=64 
training_steps=int(32119*9./(10*batch_size))

gen=get_training_data(64)
coeffs=[]
for batchy in range(training_steps):
  x_train,[y_train_norm,y_train_pan]=next(gen)
  new_shape=np.shape(x_train)[0]*128*128
  x_train=x_train.reshape(new_shape,6)#64,128,128,6
  y_train_norm=y_train_norm.reshape(new_shape,6)

  df1 = pd.DataFrame(x_train,columns=['B1','B2','B3','B4','B5','B7'])
  df2 = pd.DataFrame(y_train_norm,columns=['B1','B2','B3','B4','B5','B7'])
  regr=linear_model.LinearRegression()
  regr.fit(df1,df2)
  if batchy==0:
    coeffs_final=regr.coef_
    intercepts_final=regr.intercept_
  else:
    coeffs_final=coeffs_final+regr.coef_
    intercepts_final=intercepts_final+regr.intercept_

test_dataset = (test_images,[test_labels_norm,test_labels_pan])
training_steps=int(32119*9./(10*batch_size)) #32119 #9738 was reduced for bad regions
testing_steps=int(32119*1./(10*batch_size))

from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=20, min_lr=0.00001)

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("best_model_ALL5.hdf5", monitor='loss', verbose=1,
    save_best_only=False, mode='auto', save_freq='epoch')

callbacks_list = [reduce_lr,checkpoint ]

batch_size=64 #Highest power of 2 possible with Colab for this model

history=model.fit(get_training_data(batch_size), epochs=250,verbose=2,shuffle=True,batch_size=batch_size,steps_per_epoch=training_steps, initial_epoch=0,callbacks=callbacks_list, validation_data=test_dataset)
