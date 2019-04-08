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
from sklearn import preprocessing

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

timeSeries=hf.niiToTS(niiname)
print(timeSeries.shape)

For Specific ROI in Brain use OtherNII
timeSeries=hf.otherNII(niiname)
print(timeSeries.shape)

'''

DELAY=6

#Deciding whether to use enjoyment or sadness labels
moveon=False
while not moveon:
    userInp=input("To use sadness data input s or to use enjoyment data input e: ")
    userInp=userInp.upper()
    if userInp=="S":
        enjoy=False
        moveon=True
    elif userInp=="E":
        enjoy=True
        moveon=True
    else:
        print("Try another input ya dingus")
'''

-------NEW METHOD--------------------------
For count 1-40
Check if a label file exists, check if a niifile exists or check if preprocessed niifile exists
Do preprocessing or load file
add to arrays simulaneously
'''

saving=False
if saving:
    endnum=int(input("What is todays date? Please use mdy with no slashes or dashes: "))
    for f in os.listdir(DATA_PATH_FOR_NII):
        hf.savePreNii(os.path.join(DATA_PATH_FOR_NII,f),f + str(endnum))
        
        
        
nii_array=[]
label_array=[]
useSaved=True
ROITrain=False
for count in range(1,40):
    gotlabel=False
    gotnii=False
    if count<10:
        count=str(0)+str(count)
    
    #First we preprocess the nii file
    niifile="sub-"+str(count)+"_sadln_filtered_func_200hpf_standard_aroma.nii"

    if useSaved:
        fullNii=os.path.join(DATA_PATH_FOR_PRENII,niifile+str(4619)+".npy")
        if os.path.isfile(fullNii):
            nii,sizeValue=hf.loadPreNii(fullNii)
            #nii=preprocessing.scale(nii)
            gotnii=True
    
    elif ROITrain: 
        fullNii=os.path.join(DATA_PATH_FOR_NII,niifile)
        if os.path.isfile(fullNii):
            nii,sizeValue=hf.otherNii(fullNii,1)
            gotnii=True
    else:
        fullNii=os.path.join(DATA_PATH_FOR_NII,niifile)
        if os.path.isfile(fullNii):
            nii,sizeValue=hf.niiToTS(fullNii)
            #nii=preprocessing.scale(nii)
            gotnii=True
            
    #Next up is the corresponding label data
    if enjoy:
        labelfile="sub-"+str(count)+"_snl_l_enjoy_log.txt"
        fullfile=os.path.join(DATA_PATH_FOR_LABELS,labelfile)
        if os.path.isfile(fullfile):        
            label=np.array(hf.sliderPre(fullfile,3))
            gotlabel=True 
           
             
    else:
        labelfile="sub-"+str(count)+"_snl_l_emo_log.txt"
        fullfile=os.path.join(DATA_PATH_FOR_LABELS,labelfile)
        if os.path.isfile(fullfile):
            label=np.array(hf.sliderPre(fullfile,3))
            gotlabel=True

    if gotlabel and gotnii:
        nii_array.append(nii)
        label_array.append(label)

nii_array=np.array(nii_array)
label_array=np.array(label_array)



print(nii_array.shape)
print(label_array.shape)


#Global variable for percent to train on, test on and validate on
TRAIN_TEST_SPLIT=[.75,.25]
totalFiles=len(label_array)

train_labels,test_labels,train_nii,test_nii=train_test_split(label_array,nii_array,train_size=TRAIN_TEST_SPLIT[0],test_size=TRAIN_TEST_SPLIT[1])
train_labels=np.array(train_labels)
train_nii=np.array(train_nii)
test_labels=np.array(test_labels)
test_nii=np.array(test_nii)

"""
--------------------------------------
THE MODEL HERSELF (11/10 dont tell my girlfriend)
--------------------------------------
"""
#HyperParameters
EPOCHS=4 #Probably should be changed
BATCH_SIZE=5 
LOSS='binary_crossentropy'
OPTIMIZER='Adam'
inputShape=(495-DELAY,sizeValue)


model=Sequential()
model.add(LSTM(units=inputShape[0], activation='tanh',dropout=.08,input_shape=inputShape,return_sequences=True))
model.add(keras.layers.TimeDistributed(Dense(2,activation='softmax')))

model.compile(loss=LOSS,optimizer=OPTIMIZER, metrics=['acc','mean_squared_error'])

model.fit(train_nii,train_labels,epochs=EPOCHS,batch_size=BATCH_SIZE)

"""
--------------------------------------
Data Visualization/Results/Extra Modifications
--------------------------------------
"""
print(model.summary())
prediction=model.predict(test_nii,verbose=1,batch_size=4)
score=model.evaluate(test_nii,test_labels,verbose=1, batch_size=4)

if enjoy:
    print("Using enjoyment files, predicting on test files")
    print(prediction)
    print("Using enjoyment files, evaluating on test files")
    print(score)
else:
    print("Using sadness files, predicting on test files")
    print(prediction)
    print("Using sadness file, evaluating on test files")
    print(score)
import pdb;pdb.set_trace()
