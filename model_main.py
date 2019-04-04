# -*- coding: utf-8 -*-
"""
Main File for Music_Brain_Project

@author: Ted Lewitt,Ben Hahn, Jack Elliot,Skylar Van Sijil MacMillan
------------------------------------
Imports
------------------------------------
"""

import pandas as pd
import numpy as np
import nibabel as nib
import helperFunctions as hf
import keras
from keras.models import Sequential
from keras.layers import Activation,Dense
from keras.layers.recurrent import LSTM
import nilearn
import os
from sklearn.model_selection import train_test_split

server=False
if server:
    DATA_PATH_FOR_NII=os.getcwd()+"/data/nifti/"
    DATA_PATH_FOR_LABELS=os.getcwd()+"/data/text/"
else:
    DATA_PATH_FOR_LABELS=r"C:\Users\Ted\Desktop\CAIS_MUSIC_BRAIN\TXT-Files"
    DATA_PATH_FOR_NII=r"C:\Users\Ted\Desktop\CAIS_MUSIC_BRAIN\NII-Files"
    DATA_PATH_FOR_PRENII=r"C:\Users\Ted\Desktop\CAIS_MUSIC_BRAIN\Preprocessed_Files"


'''
--------------------------------------
Data Preprocessing 
i.e. Everything before model=Sequential()
--------------------------------------
#########Label Preprocesssing#########
Preprocessing the slider data from 30hz to 1hz to match fmri data
For Argument #2: Maxpool=1, minpool=2,mean=3,median=4

filename=os.path.join(DATA_PATH_FOR_LABELS,"sub-01_snl_l_enjoy_log.txt")
print(filename)
out=np.array(hf.sliderPre(filename,3))
print(out.shape)
print(out)

#########NII Preprocesssing#########
Preprocessing the nii files to Time Series data
More on the Nifti Masker function used to accomplish this
http://nilearn.github.io/modules/generated/nilearn.input_data.NiftiMasker.html

hf.savePreNii(niiname,"testingsave")
timeseries=hf.loadPreNii("testingsave.npy")
print(timeseries.shape)

# remove cerebellum based on coordinates
# for voxel in niiname:
#   if voxel[y]<60 and voxel[z]<24:
#       niiname.remove(voxel)

# gray matter mask, might remove cerebellum by itself
# grayMatterMasked = hf.greyMatter(niiname)
# change niiname to grayMattterMasked in line 60

timeSeries=hf.niiToTS(niiname)
print(timeSeries.shape)

For Specific ROI in Brain use OtherNII
timeSeries=hf.otherNII(niiname)
print(timeSeries.shape)

'''



niiname=os.path.join(DATA_PATH_FOR_NII,"sub-01_sadln_filtered_func_200hpf_cut20_standard.nii")

#Takes all the names of the label files and preprocesses them into data for the LSTM
label_array=[]
#Deciding whether to use enjoyment or sadness labels
enjoy=False;
for f in os.listdir(DATA_PATH_FOR_LABELS):
    if enjoy:
        if "enjoy" not in f:
            label_array.append(np.array(hf.sliderPre(os.path.join(DATA_PATH_FOR_LABELS,f),3)))
    else:
        if "emo" not in f:
            label_array.append(np.array(hf.sliderPre(os.path.join(DATA_PATH_FOR_LABELS,f),3)))
label_array=np.array(label_array)

####Takes all the names of the nii files and preprocesses them into data for the LSTM###

#True when creating data to be preprocessed and saved
saving=False
if saving:
    count=0
    for f in os.listdir(DATA_PATH_FOR_NII):
        count+=1
        hf.savePreNii(os.path.join(DATA_PATH_FOR_NII,f),f + str(count))
#Used with otherNii when checking individual ROIS
ROITrain=False
#Using Saved Data or not using
useSaved=True
nii_array=[]
count=0
for f in os.listdir(DATA_PATH_FOR_NII): 
    if useSaved:
        nii_array.append(hf.loadPreNii(os.path.join(DATA_PATH_FOR_PRENII,f+str(count)+".npy")))
    else:
        if ROITrain:
            outArray,sizeValue=(hf.otherNii(os.path.join(DATA_PATH_FOR_NII,f),1))
            nii_array.append(np.array(outArray))
        else:
            outoutArray,sizeValue=(hf.NiitoTS(os.path.join(DATA_PATH_FOR_NII,f),1))
            nii_array.append(np.array(outArray))
    



#Global variable for percent to train on, test on and validate on
TRAIN_TEST_SPLIT=[.75,.25]
totalFiles=len(label_array)

train_subset_labels,test_subset_labels,train_subset_nii,test_subset_nii=train_test_split(label_array,nii_array,train_size=TRAIN_TEST_SPLIT[0],test_size=TRAIN_TEST_SPLIT[1])
train_subset_labels=np.array(train_subset_labels)
train_subset_nii=np.array(train_subset_nii)
test_subset_labels=np.array(test_subset_labels)
test_subset_nii=np.array(test_subset_nii)

"""
--------------------------------------
THE MODEL HERSELF (11/10 dont tell my girlfriend)
--------------------------------------
"""
#HyperParameters
MAX_SLIDER_VALUE=127
EPOCHS=1
BATCH_SIZE=1
LOSS='binary_crossentropy'
OPTIMIZER='Adam'
inputShape=(495,sizeValue)


model=Sequential()
model.add(LSTM(units=495, activation=keras.layers.LeakyReLU(alpha=.025),dropout=.08,input_shape=(495,20112),return_sequences=True))
model.add(keras.layers.TimeDistributed(Dense(2,activation='softmax')))

model.compile(loss=LOSS,optimizer=OPTIMIZER, metrics=['acc','mae'])

model.fit(train_subset_nii,train_subset_labels,epochs=EPOCHS,batch_size=BATCH_SIZE)

"""
--------------------------------------
Data Visualization/Results/Extra Modifications
--------------------------------------
"""
print(model.summary())
prediction=model.predict(test_subset_nii,verbose=1,batch_size=1)
score=model.evaluate(test_subset_nii,test_subset_labels,verbose=1, batch_size=1)

if enjoy:
    print("Using enjoyment files, predicting on 1 file")
    print(prediction)
    print("Using enjoyment files, evaluating on 1 file")
    print(score)
else:
    print("Using sadness files, predicting on 1 file")
    print(prediction)
    print("Using sadness file, evaluating on 1 file")
    print(score)
import pdb;pdb.set_trace()
