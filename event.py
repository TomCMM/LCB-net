#======= Import module
from __future__ import division
import os
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch
import os
import copy
from LCBnet_lib import *
from scipy import interpolate


#===============================================================================
# Hovmoller Station - interpolation + selection synoptic condition + Normalisation 
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/Hovermollerinterp/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")

Files=[
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C08clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C07clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C06clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C05clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C11clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C12clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C13clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C14clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C15clear_merge.TXT'
 ]

#Files=['/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C05clear_merge.TXT']
network=[]
net=LCB_net()
for i in Files:
    print(i)
    rr=LCB_station(i)
    network.append(rr)
    net.add(rr)

#===============================================================================
#  Determine the events
#===============================================================================
class event():
    """
    Object to handle raining event for precipitation time serie
    """
    def __init__(self,rain):
        self.rain = rain
        self.para={}
        self.paradef={'IntMin':0.001,# minimum 4 station detect in the network 0.001 * min of accumu(see notebook)
                      'AccMin':0.01,# mm
                      'Acc/IntMax':10,# def value 100 pour eviter probleme de longue accumulation et faible intensité
                      'Acctime':'10Min',
                      'IntMax':5
                      }
    def setpara(self,parameter,value):
        try:
            self.para[parameter]=value
        except:
            pass
    def getpara(self,parameter):
        try:
            value=self.para[parameter]
            return value
        except:
            print('Parameter by default used')
            try:
                value=self.paradef[parameter]
                return value
            except:
                print('this parameter dosent exist- please set it up')
    def defineEvents(self):
        rain=self.rain #original data
        RainAcc=rain.groupby(pd.TimeGrouper(self.getpara('Acctime'))).max() # grouped data
        RainAcc=RainAcc[~np.isnan(RainAcc)]# select only where the network is complete
        Rain= RainAcc> self.getpara('IntMin') # Boolean Data
        Initime=np.array([])
        Endtime=np.array([])
        for ini,end,indexini,indexend in zip(Rain[:-1],Rain[1:],Rain[:-1].index,Rain[1:].index):
            if ini == False and end == True:
                Initime=np.append(Initime,indexend)
            if ini == True and end == False:
                Endtime=np.append(Endtime,indexini)
        events=pd.DataFrame({'Initime':Initime,'Endtime':Endtime})
        Intensity=np.array([])
        Accum=np.array([])
        for i in events.index:
            Intensity=np.append(Intensity,RainAcc[events.iloc[i]['Initime'] : events.iloc[i]['Endtime']].max())
            Accum=np.append(Accum,RainAcc[events.iloc[i]['Initime'] : events.iloc[i]['Endtime']].sum())
        events['Intensity']=pd.Series(Intensity, index=events.index)
        events['Accumulation']=pd.Series(Accum, index=events.index)
        events=events[events['Accumulation'] > self.getpara('AccMin')] # Threshold accumulation 
        events=events[events['Intensity'] < self.getpara('IntMax')]
        events=events[events['Accumulation']/events['Intensity'] < self.getpara('Acc/IntMax')]
        return events


Event=event(net.Data['Rc mm'])
events=Event.defineEvents()
# histogramme intensity events
hist=np.histogram(events['Intensity'],bins=[0.15,0.75,2,4])
#hist=np.histogram(events['Intensity'],bins=[0,0.05,0.1,0.15,0.20,0.25])
#hist=np.histogram(events['Intensity'],bins=[0,0.25,4])
#hist=np.histogram(events['Intensity'],bins=np.logspace(-2.2,0.9,6))

# test
#classhist=classhist[:2]


#===============================================================================
# Distribution rainfall 
#===============================================================================
MeanRainNorma=pd.DataFrame()
nbcompleteday=np.array([])# stock the number of fail per event
for ini,end,nbevent in zip(hist[1][:-1],hist[1][1:],hist[0]):
    IntensityEvent=events[events['Intensity']> ini]
    IntensityEvent=IntensityEvent[IntensityEvent['Intensity'] < end] # must exist a better way to do this
    maxRainNorma=np.array([[]])
    for event in IntensityEvent.index:
        RainNorma=np.array([])
        for rr in network:
            starain=(rr.getvar('Rc mm')[IntensityEvent['Initime'][event] : IntensityEvent['Endtime'][event]])
            lenevent=starain.shape[0]# durée de l evenement
            if lenevent ==0:
                RainNorma=np.append(RainNorma,starain.sum())
            else:
                if (np.count_nonzero(np.isnan(starain))/lenevent)> 0.20 or (np.count_nonzero(np.isnan(starain))/lenevent)==0: # si moins de 10% de Nan j'interpole
                    RainNorma=np.append(RainNorma,starain.sum())
                    print('Il a beaucoup ou pas derreur, je laisse comme ca :'+str((np.count_nonzero(np.isnan(starain))/lenevent)))
                else:
                    print('Il y a peu d erreur alors j Interpolate')
                    #---interpolation
                    ok = -np.isnan(starain)
                    xp = ok.ravel().nonzero()[0]
                    fp = starain[-np.isnan(starain)]
                    x  = np.isnan(starain).ravel().nonzero()[0]
                    starain[np.isnan(starain)] = np.interp(x, xp, fp)
                    #---
                    RainNorma=np.append(RainNorma,starain.sum())
        print(np.count_nonzero(np.isnan(RainNorma)))
        nbcompleteday=np.append(nbcompleteday,np.count_nonzero(np.isnan(RainNorma)))
        if np.count_nonzero(np.isnan(RainNorma))<7 :
                #---interpolation
            ok = -np.isnan(RainNorma)
            xp = ok.ravel().nonzero()[0]
            fp = RainNorma[-np.isnan(RainNorma)]
            x  = np.isnan(RainNorma).ravel().nonzero()[0]
            RainNorma[np.isnan(RainNorma)] = np.interp(x, xp, fp)
            #---
            maxRainNorma=np.append(maxRainNorma,RainNorma)
#             print('complete data')
#             print(maxRainNorma)
    maxRainNorma=maxRainNorma.reshape(maxRainNorma.shape[0]/12,12)
    maskedData = np.ma.masked_array(maxRainNorma,np.isnan(maxRainNorma))
    MeanRainNorma[str(round(ini,2))+'_'+str(round(end,2))]=pd.Series(np.sum(maskedData,axis=0).data)
#     MeanRainNorma=np.append(MeanRainNorma,np.mean(maskedData,axis=0).data)
# MeanRainNorma=MeanRainNorma.reshape(hist[0].shape[0],12)

np.histogram(nbcompleteday)# Histogramm of the day with fail

plt.figure(figsize=(21,12))
plt.plot(MeanRainNorma)
plt.legend(MeanRainNorma.columns)
plt.savefig('Intensity_normalised.png')
plt.close()

#Files=reversed(Files)
position=[]
staname=[]
stations=pos_sta().sortsta('Lon')
for i in stations:
    position.append(i[1])
    staname.append(i[0])

#position=position[::-1]
nbsta=len(Files)


time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30




#===============================================================================
#  Selection events
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/synoptic/SyntheseSynopticCPTEC/synoptic_condition.csv'
Eventsynoptic=pd.read_csv(InPath,index_col=0,parse_dates=True)
Eventsynoptic=Eventsynoptic['2015-02-01'::]# avoid september

#===============================================================================
# Hovermoller
#===============================================================================
for event in ['ZCAS']:
    IndexEvent=Eventsynoptic.index[Eventsynoptic[event]==True]
    if event == 'Front':# specify which prefrontal or postfrontal
        IndexEvent=IndexEvent+ pd.DateOffset(-1)
        IndexEvent2=IndexEvent+ pd.DateOffset(-2)
        IndexEvent3=IndexEvent+ pd.DateOffset(-3)
        IndexEvent=IndexEvent+IndexEvent2+IndexEvent3
    var=np.array([])
    Wind_speed=np.array([])
    Wind_dir=np.array([])
    Norm=np.array([])
    Theta=np.array([])
    
    for rr in network:
        variable=rr.getvar('Rc mm')
        vel_10min=rr.getvar('Sm m/s').groupby(pd.TimeGrouper('20Min')).mean()
        dir_10min=rr.getvar('Dm G').groupby(pd.TimeGrouper('20Min')).mean()
        
        newvar=pd.Series()# select Index of Event (Should exist a better way)
        newvel=pd.Series()
        newdir=pd.Series()
        for i in IndexEvent.dayofyear:
            newvar=newvar.append(variable[variable.index.dayofyear==i])
            newvel=newvel.append(vel_10min[vel_10min.index.dayofyear==i])
            newdir=newdir.append(dir_10min[dir_10min.index.dayofyear==i])
        
        newvar=newvar.groupby(lambda t: (t.hour,t.minute)).sum()
        newvel=newvel.groupby(lambda t: (t.hour,t.minute)).mean()
        newdir=newdir.groupby(lambda t: (t.hour,t.minute)).mean()
        var=np.append(var,newvar.tolist())
        print var.shape
        Norm=np.append(Norm,newvel.tolist())
        Theta=np.append(Theta,newdir.tolist())

    FIG=LCBplot(rr)
    plt.figure(figsize=(FIG.getpara('wfig'),FIG.getpara('hfig')))
    plt.suptitle(FIG.getpara('subtitle'),fontsize=20)
    
    var=var.reshape(nbsta,720)
    V=np.cos(map(math.radians,Theta+180))*Norm
    U=np.sin(map(math.radians,Theta+180))*Norm

    U=U.reshape(nbsta,72)
    V=V.reshape(nbsta,72)
    
    var=var.transpose()
    U=U.transpose()
    V=V.transpose()

#  Interpolation

    newvar=np.array([[]])
    for i in np.arange(var.shape[0]):
        data=var[i,:]
        x=np.array(position)
        mask=~np.isnan(data)
        datamask=data[mask]
        positionmask=x[mask]
        try:
            f=interpolate.InterpolatedUnivariateSpline(positionmask,datamask,k=1)
            newvar=np.append(newvar,f(x))
        except:
            print('Cant interpolate - Therfore let NAN data')
            newvar=np.append(newvar,data)
    
    newvar=newvar.reshape(720,nbsta)
    var=newvar


    U.shape
    V.shape
    var.shape
    Position.shape
    Time.shape
    Levels=np.linspace(0,5,100)
    #Levels=np.linspace(-0.1,0.1,30)
    cmap = plt.cm.get_cmap("RdBu_r")
    plt.contourf(Position,Time,var,levels=Levels,cmap=cmap)    
    plt.colorbar()
    a=plt.quiver(Position[::10,::],Time[::10,::],U[:,:],V[:,:],scale=35)
    #plt.gca().invert_xaxis()    


    l,r,b,t = plt.axis()
    dx, dy = r-l, t-b
    plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

    plt.savefig(str(out)+str(event)+'-hovermoler.png')
    plt.close()



