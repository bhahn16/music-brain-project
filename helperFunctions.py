# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 19:51:13 2019

@author: Ted,Ben Skyler,Jack
"""

"""
All helper functions: -aphabetical or chronological???
"""
import numpy as np
import os
import nilearn
from nilearn import image
from nilearn.masking import compute_epi_mask
from nilearn.plotting import plot_epi
from nilearn.input_data import NiftiMasker
from sklearn.preprocessing import OneHotEncoder

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
    #import pdb; pdb.set_trace()
    for i in range(0,len(content1)-1):
        times.append(content1[i].split("\t")[0])
        sliderValue.append(content1[i].split("\t")[1])  
    

    maxValue=127
    outputarray=[]
    methodselector=method
    for i in range(20,515):
        fullarray=np.array(sliderValue[30*i:30*i+29]).astype(np.float)
        if methodselector==1:
            outputarray.append(max(fullarray)/maxValue)
        elif methodselector==2:
            outputarray.append(min(fullarray)/maxValue)
        elif methodselector==3:
            num=np.average(fullarray)
            if num>maxValue/2:
                toOut=[0,1]
            else:
                toOut=[1,0]
            outputarray.append(toOut)
        elif methodselector==4:
            outputarray.append(np.median(fullarray)/maxValue)
    content.close()
    return outputarray

# def grayMatter(fileName):
#     grey_matter_masker = nilearn.masking.compute_gray_matter_mask(fileName, threshold, connected=True, opening=2, memory=None, verbose=0)
#     masked = grey_matter_masker.transform(fileName)
    
#     return masked 
    
    
def niiToTS(filename):
    """
    Takes 4D .nii file and makes it into a 2D time series
    """

         
    #nifti_masker=nilearn.input_data.NiftiMasker(standardize=True, mask_strategy='epi')
    nifti_masker=nilearn.input_data.NiftiMasker(standardize=True, mask_strategy='background',smoothing_fwhm=8)   

    nifti_masker.fit(filename)
    masked=nifti_masker.transform(filename)
    masked=np.array(masked)
    #np.pad(masked,(0,359320-masked.shape[1]),'constant')
    
    newmasked=np.zeros((495,359320))
    #redo this with np.pad
    for i in range(495):
        for j in range(masked.shape[1]):
            newmasked[i][j]=masked[i][j]
            
   # boole=np.array_equal(newmasked,masked)
   #  print(boole)
    return newmasked,newmasked.shape[1]
    

def savePreNii(filename,outputname):
    """
    Saves a preprocessed nii file to an outputfile
    """
    toSave=niiToTS(filename)
    np.save(outputname,toSave)
def loadPreNii(filename):
    """
    Loads a preprocessed NII file from savePreNii
    """
    return np.load(filename)

def otherNii(filename,timepoint):
    '''
    Uses an EPI mask of the 3D image at the timepoint input as features
    '''
    DATA_PATH_FOR_NII=r"C:\Users\Ted\Desktop\CAIS_MUSIC_BRAIN\NII-Files"

    niiname2=os.path.join(DATA_PATH_FOR_NII,"sub-01_sadln_filtered_func_200hpf_cut20_standard.nii")

    nilearn.plotting.show()
    result=nilearn.image.index_img(niiname2,timepoint)
    print(result.shape)
    plot_epi(result)
    mask_img=compute_epi_mask(result)
    nilearn.plotting.plot_roi(mask_img,result)
    masked_data=nilearn.masking.apply_mask(filename,mask_img)
    return masked_data,masked_data.shape[1]

