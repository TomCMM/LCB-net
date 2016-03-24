import glob
import LCBnet_lib
from LCBnet_lib import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import statsmodels.api as sm
import matplotlib
from scipy.interpolate import interp1d
from collections import Counter
from scipy import interpolate
from LapseRate import AltitudeAnalysis

if __name__=='__main__':
    #===========================================================================
    #  Get input Files
    #===========================================================================
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)

    station_names =AttSta.stations(['Head','West','valley'])
#     station_names.append('C17')

    Files =AttSta.getatt(station_names,'InPath')
    Files = Files + AttSta.getatt(AttSta.stations(['Head','West','slope']),'InPath')
    
    altanal = AltitudeAnalysis(Files)

    #===========================================================================
    # Plot var in function of Altitude
    #===========================================================================

#     hours = np.arange(15,24,1)
#     hours = [16, 18, 20,22,0,2, 4]
    altanal.VarVsAlt(vars= ['Ta C'], by= 'H',  dates = hours, From='2015-03-01 00:00:00', To='2015-08-01 00:00:00')
    altanal.plot(analysis = 'var_vs_alt', marker_side = True, annotate = True, print_= True)

    #===========================================================================
    # Plot var in function of Altitude - Mean summer and Winter
    #===========================================================================

    hours = [10,12,14]
    altanal.VarVsAlt(vars= ['Ta C', 'Ua g/kg', 'Theta C', 'Sm m/s'], by= 'H',  dates = hours, From='2014-10-01 00:00:00', To='2015-08-01 00:00:00')
    altanal.plot(analysis = 'var_vs_alt', annotate = True, print_= True)
 
 
#     plt.figure()
#     East =AttSta.getatt(AttSta.stations(['Head','slope']),'InPath')
#     East = East + AttSta.getatt(AttSta.stations(['Head','ridge']),'InPath')
#     
#     West =AttSta.getatt(AttSta.stations(['Head','slope']),'InPath')
#     West = West + AttSta.getatt(AttSta.stations(['Head','ridge']),'InPath')
