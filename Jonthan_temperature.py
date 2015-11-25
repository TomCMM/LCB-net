import glob
from LCBnet_lib import *


InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
Files=glob.glob(InPath+"*")

net=LCB_net()
AttSta = att_sta()
AttSta.setInPaths(InPath)
AttSta.showatt()

stanames = AttSta.stations(['Head'])
staPaths = AttSta.getatt(stanames , 'InPath')
net.AddFilesSta(staPaths)

data = net.Data["2014-10-01 00:00:00" : "2015-10-01 00:00:00"] 

min_day = data['Ta C'].resample('D',how='mean')
month_min_mean = min_day.groupby(lambda x: x.month).mean()

month_min_mean.to_csv('/home/thomas/Monthlymean_dailymean_Full.csv')
month_min_mean.plot()


