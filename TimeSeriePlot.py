#===============================================================================
# DESCRIPTION
#    Simple plot of the overall Serie
# INPUT
#    InPath: Path of the WXT data
#    OutPath: Path for the graphic Output
#    Stations
#    From: INitial time of the period to plot
#    To: End time of the period to plot
#    EXAMPLE
#    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
# out='/home/thomas/'
# From='2013-11-01 00:00:00'
# To='2014-11-10 00:00:00'
# var='Ta C'
#===============================================================================
import glob
import LCBnet_lib
from LCBnet_lib import *


def TimeSeriePlot(self, Files, stations = None, network = None):
    if stations == True:
        for i in Files:
            sta=LCB_station(i)
            print(sta.getpara('staname'))
            plot=LCBplot(sta)
            plot.setpara('OutPath',OutPath)
            plot.TimePlot(['Ta C'])
    if network == True:
        
        
if __name__=='__main__':
    Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
    OutPath='/home/thomas/'
    Files=glob.glob(Path+"*")
    TimeSeriePLot(Files, stations = True)









