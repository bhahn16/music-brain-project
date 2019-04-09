# -*- coding: utf-8 -*-
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
from sklearn import preprocessing

DELAY=6

def sliderPre(filename,method):
    """
    filename: File Name of text file
    method:
        1-Max
        2-Min
        3-Average
        4-Median
        
        
        TODO:
            Cut off last 6 seconds-DONE-LINE 46
            
        
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
  #  for i in range(20,515):
    for i in range(20,515-DELAY): 
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
    

def niiToTS(filename):
    """
    Takes 4D .nii file and makes it into a 2D time series
    TODO:
        Chop off first 6 second -USES indexer and Index_img
        Figure how much to pad-no need, all data is 229,007
        Save all good preprocessed data on team drive
        
    """

         
    #nifti_masker=nilearn.input_data.NiftiMasker(standardize=True, mask_strategy='epi')
    nifti_masker=nilearn.input_data.NiftiMasker(standardize=True, mask_strategy='template')
    indexer=[i for i in range(0,495-DELAY)]
    result=nilearn.image.index_img(filename,indexer)
    nifti_masker.fit(result)
    masked=nifti_masker.transform(result)
    masked=np.array(masked)
    #print(masked.shape)

 
    #np.pad(masked,(0,359320-masked.shape[1]),'constant')
    '''
    newmasked=np.zeros((495,250000))
    #redo this with np.pad
    for i in range(495):
        for j in range(masked.shape[1]):
            newmasked[i][j]=masked[i][j]
   '''       
    return masked,masked.shape[1]
    

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
    matrix=np.load(filename)
    matrix=matrix[0][:][:]
    #import pdb;pdb.set_trace()
    return matrix,matrix.shape[1]

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


def randtestfctn():
    
    DATA_PATH_FOR_NII=r"C:\Users\Ted\Desktop\CAIS_MUSIC_BRAIN\NII-Files"
    niiname2=os.path.join(DATA_PATH_FOR_NII,"sub-01_sadln_filtered_func_200hpf_cut20_standard.nii")
    indexer=[i for i in range(2,480)]
    result=nilearn.image.index_img(niiname2,indexer)
    #import pdb;pdb.set_trace()
    nextimg=nilearn.image.index_img(niiname2,4)
    
    result=nilearn.image.concat_imgs([result,nextimg])
    print(result.shape)
def preproVis():
    filename=r"C:\Users\Ted\Desktop\CAIS_MUSIC_BRAIN\Preprocessed_Files\sub-31_sadln_filtered_func_200hpf_standard_aroma.nii4619.npy"
    matrix,shp=loadPreNii(filename)
    import matplotlib.pyplot as plt
    #import pdb;pdb.set_trace()
    somenums=np.linspace(0,489,num=489)
    plt.figure(2,figsize=(12,12))
    plt.subplot(221)
    plt.plot(somenums,matrix[:,220000])
    plt.xlabel("Time")
    plt.ylabel("Activation")
    plt.subplot(222)
  
    max_array=[]
    for i in range (229007):
        max_array.append(np.amax(matrix[:,i]))
    othernums=np.linspace(0,shp,num=shp)
    plt.plot(othernums,max_array)
    plt.xlabel("Feature")
    plt.ylabel("Max Activation for feature")
    plt.subplot(223)
    min_array=[]
    for i in range (229007):
        min_array.append(np.amin(matrix[:,i]))
    othernums=np.linspace(0,shp,num=shp)
    plt.plot(othernums,min_array)
    plt.xlabel("Feature")
    plt.ylabel("Min Activation for feature")
    plt.subplot(224)
    diff_array=[]
    index_array=[]
    count=0;
    for i in range(229007):
        num=max_array[i]-min_array[i]
        if num<=0.1:
            count+=1
            index_array.append(i)
        diff_array.append(num)
    plt.plot(othernums,diff_array)
    plt.xlabel("Feature")
    plt.ylabel("Max-Min Activation for feature")
    plt.show()
    print(count)
    return index_array
