#===============================================================================
# Thomas July 2015
# DESCRIPTION
#     Estimation of the Nocturnal PBL height in the Ribeirao Das Posses
#     Based on the station observations
#===============================================================================

# Library
import glob
from LCBnet_lib import *






if __name__=='__main__':
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    OutPath='/home/thomas/hovermoler.png'
    Files=glob.glob(InPath+"*")
    net=LCB_net()
    net.AddFilesSta(Files)