# -*- coding: utf-8 -*-
"""
Main File for Music_Brain_Project

@author: Ted Lewitt,Ben Hahn, Jack Elliot
"""

"""
WorkFlow

Thoughts:
    Use Sphinx to document, make the repos page a sphinx document
"""




 
"""
------------------------------------
Imports and Global Variables
------------------------------------
"""

import pandas as pd
import numpy as np
import nibabel as nib
import helperFunctions as hf
import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import nilearn
import os
from sklearn.model_selection import train_test_split


"""
--------------------------------------
Data Preprocessing 
i.e. Everything before model=Sequential()
--------------------------------------
"""

"""
Preprocessing the slider data from 10hz to 1hz to match fmri data
maxpool=1, minpool=2,mean=3,median=4
"""

filename=r".\sub-01_snl_l_enjoy_log.txt"
out=hf.sliderPre(filename,3)
print(out[1:10])
"""
Preprocessing the nii files to Time Series data
More on the Nifti Masker function used to accomplish this
http://nilearn.github.io/modules/generated/nilearn.input_data.NiftiMasker.html
"""
niiname=r".\sub-01_sadln_filtered_func_200hpf_cut20_standard.nii"
timeSeries=hf.niiToTS(niiname)
print(timeSeries.shape)
print(np.array(timeSeries)[1][1:10])

DATA_PATH_FOR_LABELS=""
DATA_PATH_FOR_NII=""

#Used to calculate output of the LSTM layer
MAX_SLIDER_VALUE=150


#Takes all the names of the label files and preprocesses them into data for the
#LSTM
label_array=[]
for f in os.listdir(DATA_PATH_FOR_LABELS):
    label_array.append(hf.sliderPre(os.path.join(DATA_PATH_FOR_LABELS,f),3))
label_array=np.asarray(label_array)
#Takes all the names of the nii files and preprocesses them into data for the
#LSTM
nii_array=[]
for f in os.listdir(DATA_PATH_FOR_NII):
    nii_array.append(hf.niiToTS(os.path.join(DATA_PATH_FOR_NII,f),3))
nii_array=np.asarray(nii_array)

#Global variable for percent to train on, test on and validate on
TRAIN_TEST_SPLIT=[.80,.20]
totalFiles=len(label_array)
train_subset_labels,test_subset_labels,train_subset_nii,test_subset_nii=train_test_split(label_array,nii_array,train_size=totalFiles*TRAIN_TEST_VALIDATION[0],test_size=totalFiles*TRAIN_TEST_VALIDATION[1])
"""
--------------------------------------
THE MODEL HERSELF (11/10 dont tell my girlfriend)
--------------------------------------
"""
model=Sequential()
model.add(LSTM(MAX_SLIDER_VALUE,activation=keras.layer.LeakyReLU(alpha=.025),dropout=.08))
print(model.summary())
###########################
LOSS='mean_squared_error'
OPTIMIZER='RMSprop'
model.complie(loss=LOSS,optimizer=OPTIMIZER, metrics=['acc','mae'])
#HyperParameters
EPOCHS=10
BATCH_SIZE=128
VALIDATION_SPLIT=
labels=
model.fit(train_subset_nii,train_subset_labels,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=.2)
"""
--------------------------------------
Data Visualization/Results/Extra Modifications
--------------------------------------
"""
'''