##################################################################################
#	DESCRIPTION
#		Class "ManageDataLCB" -> Read data LCB observations, clean the possible error and write a new file
#		Class "LCB_Data" -> Contain the clean data from ONE WXT from the LCB data and module to transform them in specific way 
#		
#
#	TODO
#		Merge the different files from the same stations 
#=======================================================================================================================
#======= Import module
from __future__ import division
from lib import *
import os
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch
import os

#===============================================================================
# Bulk Clean Data
#===============================================================================
#====== User Input 
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data/Extrema14303/Extrema14303/'
OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data/Extrema14303/Clean/'

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
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data'
OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'



if not os.path.exists(OutPath):
	os.makedirs(OutPath)

stations=['C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15']

for s in stations:
	# Permit to find all the find with the extension .TXTclear
	matches = []
	datamerge=None
	for root, dirnames, filenames in os.walk(InPath):
		for filename in fnmatch.filter(filenames, s+'*.TXTclear'):
			matches.append(os.path.join(root, filename))	
	for i in matches:
		print(i)
		data=ManageDataLCB(os.path.dirname(i)+"/",os.path.basename(i))
		try:
			datamerge.append_dataframe(data)
		except:
			datamerge=data
	datamerge.clean_dataframe()
	datamerge.write_dataframe(OutPath,s+'clear_merge.TXT')


#===============================================================================
# Polar Plot
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Extrema14260/Clean/C/'
out='/home/thomas/'

for i in Files:
	tt=LCB_polarPlot('/home/thomas/PhD/obs-lcb/WXT/obs/Extrema14243/Clean/C/C0414247.TXT').plot()
	plt.savefig(out+"C0414247.TXT"+'.png', transparent=True)
	
for i in Files:
	tt=LCB_polarPlot(i).plot()
	plt.savefig(out+i[-18:]+'.png', transparent=True)

#===============================================================================
# Time serie vector plot
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")
net=LCB_net()

Files.remove('/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C15clear_merge.TXT')
Files.remove('/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT')
# create the net
network=[]
for i in Files:
	rr=LCB_station(i)
	rr.Set_From('2014-09-8 00:00:00')
	rr.Set_To('2014-09-9 00:00:00')
	rr.Data.index
	net.add(rr)
	network.append(rr)

for rr in network:
	rr.Set_By('H')
	rr.Set_From('2014-09-8 00:00:00')
	rr.Set_To('2014-09-9 00:00:00')
	oo=Vector([rr.DailyDiffnet()['Ua %'],rr.DailyDiffnet()['Ta C']],rr.daily_h()['Dm G'],rr.daily_h()['Sm m/s'])
	#oo.SetType('AnomalieH')
	#oo.SetType('type1')
	oo.SetType('AnomalieH')
	oo.SetTypeTwin('AnomalieT')
	oo.SetOption('twin',True)
	oo.plot()
	print(i+ '  '+rr.From+' '+rr.To)
	plt.savefig(out+rr.get_InPath()[-18:]+'.png', transparent=True,bbox_inches='tight')
	plt.close()

for rr in network:
	rr.Set_By('H')
	rr.Set_From('2014-09-1 00:00:00')
	rr.Set_To('2014-09-10 00:00:00')
	rr.Data['Pa H'].plot()
	plt.savefig(rr.get_InPath()[-18:-4]+'.png')
	plt.close()
	print(rr.Data.columns)
	print(rr.get_InPath())
	print(rr.Data['Ua %'])

#===============================================================================
# Simple plot of the overall Serie
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")
net=LCB_net()

for i in Files:
	rr=LCB_station(i)
	rr.Data['Ta C'].plot()
	plt.savefig(rr.get_InPath()[-18:]+'.png')
	plt.close()
	
	