from __future__ import division
from LCBnet_lib import *
from Irradiance import  Irradiance_sim_obs as Irr
import matplotlib


def plot_south_north_freq(dir_ridges):
    """
    DESCRIPTION
        plot the occurence of the wind in the South direction and North direction
    INPUT
        dir_ridges: list of series of wind direction
    """
    lcbplot = LCBplot() # get the plot object
    argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
    arglabel = lcbplot.getarg('label')
    argticks = lcbplot.getarg('ticks')
    argfig = lcbplot.getarg('figure')
    arglegend = lcbplot.getarg('legend')
    
    fig = plt.figure(**argfig)
    color = ['r','b']
    for c, dir_ridge in zip(color,dir_ridges):
        dir_ridge = dir_ridge.dropna()
        Southwind = dir_ridge[(dir_ridge > 135) & (dir_ridge < 225)].dropna()
        Northwind = dir_ridge[(dir_ridge < 45) | (dir_ridge > 315)].dropna()

        Southfreq = np.bincount(Southwind.index.hour) / np.bincount(dir_ridge.index.hour)
        Northfreq = np.bincount(Northwind.index.hour) / np.bincount(dir_ridge.index.hour)
        
        plt.plot(Southfreq*100,'-', c=c,linewidth = 10)
        plt.plot(Northfreq*100,'--', c=c,linewidth = 10)
        plt.xticks(**arglabel)
        plt.yticks( **arglabel)
        
        plt.tick_params(axis='both', which='major', **argticks)
        plt.xlabel('hours (h)', **arglabel)
        plt.ylabel('Frequency (%)', **arglabel)
    plt.grid(True)
#     plt.show()
    plt.savefig("/home/thomas/wind_ridge.svg", transparent=True)

if __name__=='__main__':
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)
    station_names =AttSta.stations(['West','ridge'])
    Files =AttSta.getatt(station_names,'InPath')
     
    print Files
     
    sta_ridge = LCB_station(Files[0])
 
 
    dir_ridge_summer = sta_ridge.getData(var=["Dm G"], From='2014-10-01 00:00:00', To='2015-04-01 00:00:00', From2='2015-10-01 00:00:00', To2='2016-01-01 00:00:00' )
    dir_ridge_winter = sta_ridge.getData(var=["Dm G"], From='2015-04-01 00:00:00', To='2015-10-01 00:00:00')
    print dir_ridge_summer
    print dir_ridge_winter
 
    plot_south_north_freq([dir_ridge_summer,dir_ridge_winter])
   





