#===============================================================================
# DESCRIPTION
#    Simple plot of the mean network variable
#    To be used in the article

#===============================================================================

import glob
from LCBnet_lib import *

if __name__=='__main__':
    Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    OutPath='/home/thomas/MergeDataThreeshold/'
    Files=glob.glob(Path+"*")
    net=LCB_net()
    net.AddFilesSta(Files)

    net.TimePlot(var = ['Ta C','Ua g/kg','Pa H','Rc mm', 'Sm m/s'],
                  subplots = True, From = "2014-11-01 00:00:00", To = '2015-08-01 00:00:00')


