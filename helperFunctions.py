# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 19:51:13 2019

@author: Ted,Ben Skyler,Jack
"""

"""
All helper functions: -aphabetical or chronological???
"""
import numpy as np
import nilearn
from nilearn import image
from nilearn.input_data import NiftiMasker

def sliderPre(filename,method):
    """
    filename: File Name of text file
    method:
        1-Max
        2-Min
        3-Average
        4-Median
    """
    content=open(filename,'r')
    content1=content.read().split("\n")
    
    times=[]
    sliderValue=[]
    
    for i in range(0,len(content1)-1):
        times.append(content1[i].split("\t")[0])
        sliderValue.append(content1[i].split("\t")[1])  
    
    
    outputarray=[]
    methodselector=method
    for i in range(0,515):
        fullarray=np.array(sliderValue[30*i:30*i+29]).astype(np.float)
        if methodselector==1:
            outputarray.append(max(fullarray))
        elif methodselector==2:
            outputarray.append(min(fullarray))
        elif methodselector==3:
            outputarray.append(np.average(fullarray))
        elif methodselector==4:
            outputarray.append(np.median(fullarray))
    content.close()
    return outputarray
    

def niiToTS(filename):
    """
    Takes 4D .nii file and makes it into a 2D time series
    """
    nifti_masker=nilearn.input_data.NiftiMasker(standardize=True, mask_strategy='background',smoothing_fwhm=8)
    nifti_masker.fit(filename)
    masked=nifti_masker.transform(filename)
    return masked
    

def inputtoLSTM(label_data,train_data):
    """
    Takes label data array and training data array, both with same number of time dimesions
    and prepares them for the LSTM
    """
    