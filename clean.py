from __future__ import division
import os
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch
import copy
from LCBnet_lib import *
from scipy import interpolate
import seaborn as sns

#===============================================================================
# Bulk Clean Data
#===============================================================================
#====== User Input

if __name__=='__main__':
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data/Extrema20150806/Extrema20150806'
    OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data/Extrema20150806/Clean/'
    #====== Find all the clima and Hydro
    Files=glob.glob(InPath+"/*/*")
    print(Files)
    if not os.path.exists(OutPath):
        os.makedirs(OutPath)
    for i in Files:
        data=ManageDataLCB(os.path.dirname(i)+"/",os.path.basename(i))
        print("Writing file "+OutPath+os.path.basename(i))
        data.write_clean(OutPath,os.path.basename(i))

#===============================================================================
#  Merge and Filter - Bulk cleanED data
#===============================================================================
#===============================================================================
# Merge the clean files
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data'
OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'


if not os.path.exists(OutPath):
    os.makedirs(OutPath)

AttSta=att_sta()
stations=AttSta.stations(['Ribeirao'])
# stations=['C06']
for sta in stations:
    # Permit to find all the find with the extension .TXTclear
    print(sta)
    matches = []
    datamerge=None
    for root, dirnames, filenames in os.walk(InPath):# find all the cleared files
        for filename in fnmatch.filter(filenames, sta+'*.TXTclear'):
            matches.append(os.path.join(root, filename))    
    for i in matches:
        print(i)
        data=ManageDataLCB(os.path.dirname(i)+"/",os.path.basename(i))
        try:
            datamerge.append_dataframe(data)
        except:
            datamerge=data
    datamerge.clean_dataframe()
    datamerge.write_dataframe(OutPath,sta+'clear_merge.TXT')






