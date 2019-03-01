# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:02:47 2019

@author: Ted
""" 
import helperFunctions as hf
import os

class testSubject:
    DATA_PATH_FOR_LABELS=""
    DATA_PATH_FOR_NII=""
    def __init__(self,niifile,labelfile, labelmethod):
        self.niifile=os.path.join(DATA_PATH_FOR_NII,niifile)
        self.labelfile=os.path.join(DATA_PATH_FOR_LABELS,labelfile)
        self.labelmethod=labelmethod
    def getNII(self):
        return hf.niiToTS(self.niifile)
    def getLabel(self):
        return hf.sliderPre(self.labelfile,self.labelmethod)