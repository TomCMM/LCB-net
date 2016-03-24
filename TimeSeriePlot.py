#===============================================================================
# DESCRIPTION
#    Simple plot of the overall Serie of the stations
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


def TimeSeriePlot(Files, var = 'Ta C', OutPath = '/home/thomas/'):
    if not isinstance(var, list):
        var = [var]

    for v in var:
        for i in Files:
            print i
            sta=LCB_station(i)
            sta.TimePlot(var = v)


if __name__=='__main__':
    Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     Path='/home/thomas/MergeDataThreeshold/'
    OutPath='/home/thomas/'
    Files=glob.glob(Path+"*")
    TimeSeriePlot(Files,var=[ 'Ta C'], OutPath = OutPath)
 








