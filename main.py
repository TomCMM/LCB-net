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


#===============================================================================
# Clean Data
#===============================================================================
#====== User Input 
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data/Extrema14274/Extrema14274'
OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data/Extrema14274/Clean/'

#====== Find all the clima and Hydro
Files=glob.glob(InPath+"/*/*")
if not os.path.exists(OutPath):
	os.makedirs(OutPath)

for i in Files:
	data=ManageDataLCB(os.path.dirname(i)+"/",os.path.basename(i))
	print("Writing file "+OutPath+os.path.basename(i))
	data.write_clean(OutPath,os.path.basename(i))

#===============================================================================
#  Merge Clean Data
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data'
OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'

import fnmatch
import os

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
	datamerge.write_dataframe(OutPath,s+'clear_merge.TXT')

data1=ManageDataLCB(os.path.dirname(matches[1])+"/",os.path.basename(matches[1]))
data1=data1.Dataframe
data2=ManageDataLCB(os.path.dirname(matches[2])+"/",os.path.basename(matches[2]))
data2=data2.Dataframe

data1['index']=data1.index
data2['index']=data2.index

# data3=pd.merge(data1,data2,how='outer')
# data3=data1.join(data2)
#            self.Dataframe=self.Dataframe.append(fileobject.Dataframe).sort_index(axis=0)
			#			 self.Dataframe=pd.merge(self.Dataframe,fileobject.Dataframe,left_index=True, right_index=True,how='outer')

data3=data1.append(data2).sort_index(axis=0)
data1.index
data2.index
data3.index

	def append_dataframe(self,fileobject):
		"""
		User input: A list of file path with the different files to merge
		Description	
			exemple: H05XXX240 will be merged with H05XXX245
		"""
		try:
			self.Dataframe=pd.concat([self.Dataframe,fileobject.Dataframe],axis=0)
			print("Merging dataframe "+fileobject.fname)
		except:
			print('It cant merge dataframe')
	def write_dataframe(self,OutPath,fname):
		self.Dataframe.to_csv(OutPath+fname)
		print('Writing dataframe')
		

# Test
IP="/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C08clear_merge.TXTclear"
mm=LCB_station(IP)
mm.Data['index']=mm.Data.index
newindexed=mm.Data.drop_duplicates(cols='index', take_last=True)
Final=newindexed.reindex(index=sorted(newindexed.index))
mm.Data=Final
mm.Data['Ta C'].plot()
plt.savefig('chatte.png')

mm.Data.to_csv('testc06.csv')

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
# Find all the clima and Hydro
Files=glob.glob(InPath+"/*48.TXTclear")
net=LCB_net()

# add the stations 13
Files.append('/home/thomas/PhD/obs-lcb/LCBData/obs/Extrema14260/Clean/C/CleanC1314247.TXTclear')
# create the net
network=[]
for i in Files:
	rr=LCB_station(i)
	if rr.InPath =='/home/thomas/PhD/obs-lcb/LCBData/obs/Extrema14260/Clean/C/CleanC1314247.TXTclear':
		rr.Data.index=rr.Data.index+ pd.DateOffset(days=1)+ pd.DateOffset(hours=1)
	rr.Set_From('2014-09-8 00:00:00')
	rr.Set_To('2014-09-9 00:00:00')
	rr.Data.index
	net.add(rr)
	network.append(rr)

for rr in network:
	rr.Set_By('H')
	rr.Set_From('2014-09-8 00:00:00')
	rr.Set_To('2014-09-9 00:00:00')
	oo=Vector([rr.DailyDiffnet()['Ua g/kg'],rr.DailyDiffnet()['Ta C']],rr.daily_h()['Dm G'],rr.daily_h()['Sm m/s'])
	#oo.SetType('AnomalieH')
	#oo.SetType('type1')
	oo.SetTypeTwin('AnomalieT')
	oo.SetOption('twin',True)
	oo.SetType('AnomalieH')
	oo.plot()
	print(i+ '  '+rr.From+' '+rr.To)
	plt.savefig(out+rr.get_InPath()[-18:]+'.png', transparent=True,bbox_inches='tight')





