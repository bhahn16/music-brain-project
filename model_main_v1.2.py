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
import helperFunctions_v1.2 as hf
import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM

#import scikit-learn


"""
--------------------------------------
Data Preprocessing 
i.e. Everything before model=modellib.model(<args>)
--------------------------------------
"""

"""
Preprocessing the slider data from 10hz to 1hz to match fmri data
maxpool=1, minpool=2,mean=3,median=4
"""

filename=r"C:\Users\Ted\Desktop\CAIS_MUSIC_BRAIN\sub-01_snl_l_enjoy_log.txt"
out=hf.sliderPre(filename,4)

    
'''
"""
--------------------------------------
THE MODEL HERSELF (11/10 dont tell my girlfriend)
--------------------------------------
"""
model=Sequential()
model.add(LSTM())
print(model.summary())
###########################
#HyperParameters
EPOCHS=
BATCH_SIZE=
VALIDATION_SPLIT=
labels=

model.fit()
"""
--------------------------------------
Data Visualization/Results/Extra Modifications
--------------------------------------
"""
'''