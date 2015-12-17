#===============================================================================
#     DESCRIPTION
#        Code to represent the fraction of available data
#        To be used in article
#    EXAMPLE
#        
#===============================================================================

# Import library
from __future__ import division
import glob
from LCBnet_lib import *



if __name__=='__main__':
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)
    Files =AttSta.getatt(AttSta.stations(['Head']),'InPath')

    net=LCB_net()
    net.AddFilesSta(Files)
    
    df= net.validfraction()
    From='2014-10-01 00:00:00' 
    To='2015-11-01 00:00:00'
    df =df[From:To]
    df['fraction'].resample('M').plot(kind='bar')
    plt.show()
