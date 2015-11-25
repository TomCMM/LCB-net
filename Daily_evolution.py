#===============================================================================
# DESCRIPTION
#    plot the daily evolution of the variable in the Ribeirao Das Posses
#    Plot used in the artic
#===============================================================================
from LCBnet_lib import *


if __name__ == "__main__":
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)
    
    net = LCB_net()
    net.AddFilesSta(AttSta.getatt(AttSta.stations(['Ribeirao']),'InPath'))
    
    From = ["2014-11-01 00:00:00","2015-04-01 00:00:00" ]
    To = ["2015-04-01 00:00:00 ","2015-10-01 00:00:00 "]

    net.dailyplot(var = ['Ta C', 'Ua g/kg', 'Rc mm'], From=From, To=To, group= "H", labels =['Summer', 'Winter'] , save = True)
