#===============================================================================
# DESCRIPTION
#    plot the daily evolution of the variable in the Ribeirao Das Posses
#    Plot used in the article
#===============================================================================
from LCBnet_lib import *


if __name__ == "__main__":
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)
    
    net = LCB_net()
    net.AddFilesSta(AttSta.getatt(AttSta.stations(['Ribeirao']),'InPath'))
    
    From = ["2014-11-01 00:00:00","2015-04-01 00:00:00" ]
    To = ["2015-04-01 00:00:00 ","2015-11-01 00:00:00 "]
    
    From2 = ["2015-11-01 00:00:00 ",None]
    To2 = ["2016-01-01 00:00:00 ",None]


    net.dailyplot(var = ['Ta C', 'Ev hpa'], From=From, To=To, From2 = From2, To2=To2, group= "H", labels =['Summer', 'Winter'] , save = True)
    net.dailyplot(var = ['Rc mm'],how='sum', From=From, To=To, From2 = From2, To2=To2, group= "H", labels =['Summer', 'Winter'] , save = True)
