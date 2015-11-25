from __future__ import division
from LCBnet_lib import *
from Irradiance import  Irradiance_sim_obs as Irr


def plot_south_north_freq(dir_ridges):
    """
    DESCRIPTION
        plot the occurence of the wind in the South direction and North direction
    INPUT
        dir_ridges: list of series of wind direction
    """
    
    color = ['k','r','b']
    
    for c, dir_ridge in zip(color,dir_ridges):
        Southwind = dir_ridge[(dir_ridge > 135) & (dir_ridge < 225)].dropna()
        Northwind = dir_ridge[(dir_ridge < 45) | (dir_ridge > 315)].dropna()
        
        Southfreq = np.bincount(Southwind.index.hour) / np.bincount(dir_ridge.index.hour)
        Northfreq = np.bincount(Northwind.index.hour) / np.bincount(dir_ridge.index.hour)

        plt.plot(Southfreq,'-', c=c)
        plt.plot(Northfreq,'--', c=c)
        
    plt.show()

if __name__=='__main__':
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)
    station_names =AttSta.stations(['West','ridge'])
    Files =AttSta.getatt(station_names,'InPath')
    
    sta_ridge = LCB_station(Files[0])

    dir_ridge = sta_ridge.getData(var=["Dm G"], From='2014-11-01 00:00:00', To='2015-10-01 00:00:00', by='H')
    dir_ridge_summer = sta_ridge.getData(var=["Dm G"], From='2014-11-01 00:00:00', To='2015-05-01 00:00:00', by='H')
    dir_ridge_winter = sta_ridge.getData(var=["Dm G"], From='2015-05-01 00:00:00', To='2015-10-01 00:00:00', by='H')
    
    plot_south_north_freq([dir_ridge,dir_ridge_summer,dir_ridge_winter])

    
    

    
    






