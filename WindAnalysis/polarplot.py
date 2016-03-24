#------------------------------------------------------------------------------ 
# DESCRIPTION
#    module with contain class and function necessary 
#    for the analysis of the wind pattern in the Ribeirao 
#    das Posses, Extrema, Minas Gerais
#------------------------------------------------------------------------------ 

# Library import
from __future__ import division
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from LCBnet_lib import *



if __name__ == '__main__':
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)
    
    Files = AttSta.getatt(AttSta.stations(['Head','West']),'InPath')
    
    #===============================================================================
    # Polar plot distribution
    #===============================================================================
    #------------------------------------------------------------------------------ 
    # station
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
    out='/home/thomas/'
    
    maxwind = 5
    
    Hours=[4,10,15]
    # Find all the clima and Hydro
    Files=glob.glob(InPath+"*")
    import matplotlib 
    
    matplotlib.rc('xtick', labelsize=50)
    matplotlib.rc('ytick', labelsize=50)
    # 
    Files=[
#             '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
#             '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT',
            '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C07clear_merge.TXT',
            '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
            '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C13clear_merge.TXT',
#             '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C17clear_merge.TXT'
            ]
    
    for Hour in Hours:
        print Hour
        for File in sorted(Files):
            print(File)
            station = LCB_station(File)
            wind = station.getvar('Sm m/s')
            dirwind = station.getvar('Dm G')
            wind = wind[wind.index.hour==Hour]
            dirwind = dirwind[dirwind.index.hour==Hour]
    

            hist, bin_edges = np.histogram(dirwind.values, bins=np.arange(0,360,24))
            print hist
            print hist.sum()
            hist_norm=(hist/hist.sum())*100

            Wind=[]
    
            for i,e in zip(bin_edges[:-1],bin_edges[1:]):
                Wind.append(wind[(dirwind < e) & (dirwind>i)].mean())
    
            width=np.repeat(23*np.pi/180,len(hist_norm))
            bin_rad=(bin_edges[:-1]+10)*(np.pi/180)

            plt.figure(figsize=(21,12))

            
            ax = plt.subplot(111, polar=True)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=plt.normalize(vmin=0, vmax=maxwind)) # vmin and vmax value maximum and minimum for the color
            sm._A = []
            plt.colorbar(sm)
    
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

            bars = ax.bar(bin_rad, hist_norm, width=width, bottom=0.0,  linewidth=2)
            # Use custom colors and opacity
            for r, bar in zip(Wind, bars):
                bar.set_facecolor(plt.cm.Greys(r/maxwind))
    
            ax.set_rmax(50)
            ax.set_rgrids([25,50], labels=[r'25$\%$',r'50$\%$'], angle=315)
            ax.set_xticklabels([r'0$^{\circ}$',"" , r'90$^{\circ}$', '', r'180$^{\circ}$', '', r'270$^{\circ}$', ''])


            print "print"
            plt.savefig("/home/thomas/" + station.getpara('stanames')+'__'+str(Hour)+'__'+'-polarplot.svg',bbox_inches='tight') # reduire l'espace entre les grafiques
























