#===============================================================================
# DESCRIPTION
#    Maniuplate the irradiance observed 
#    by the pyranometers in the Ribeirao Das Posses
#    make a plot of the irradiance simulated and observed
#
# AUTHOR
#    Thomas Martin 11 August 2015
#===============================================================================
from __future__ import division
import pandas as pd
import glob
import datetime
import matplotlib.pyplot as plt
from clima_lib.LCBnet_lib import *
from grad_stations import Gradient
from divergence import Divergence 
from LapseRate import AltitudeAnalysis

from LCBnet_lib import *
from divergence import Divergence
from grad_stations import Gradient
#------------------------------------------------------------------------------ 


if __name__=='__main__':
#     #===========================================================================
#     # read
#     #===========================================================================
    irr = LCB_Irr()
    inpath_obs = '/home/thomas/PhD/obs-lcb/LCBData/obs/Irradiance/data/'
    files_obs = glob.glob(inpath_obs+"*")
    print files_obs
    irr.read_obs(files_obs)
    data = irr.data_obs
    data.to_csv('/home/thomas/irr.csv')
        
    inpath_sim='/home/thomas/Irradiance_rsun_lin2_lag0_glob_df.csv'
    irr.read_sim(inpath_sim)
#       
# #     #===========================================================================
# #     # Quantiles plot
# #     #===========================================================================
# #  
# #     irr.plot_quantiles(kind ='Irr', save=True)
# #      
# #     #===========================================================================
# #     # var vs irradiation ratio
# #     #===========================================================================
# #     # Get the gradient data
    dir_inpath = '/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    outpath = '/home/thomas/'
#    
    grad = Gradient(dir_inpath)
    grad.couples_net([['West', 'East'],['valley','slope'], ['Medio', 'Head','valley']])
    data_grad = grad.grad(var=['Ta C'], by = "H", From ='2014-11-01 00:00:00', To = '2016-01-01 00:00:00' , return_=True)
     
    print data_grad
     
    valley_slope= data_grad['valley_slope2014-11-01 00:00:00']
    irr.var_vs_ratio(data_grad=valley_slope,kind = "irr",ratio_class = True, remove='30_70', save=True, name="VS")
      
    west_east= data_grad['West_East2014-11-01 00:00:00']
    irr.var_vs_ratio(data_grad=west_east,kind = "irr",ratio_class = True, remove='30_70', save=True, name="WE")
      
    medio_head= data_grad['Medio_Head_valley2014-11-01 00:00:00']
    irr.var_vs_ratio(data_grad=medio_head,kind = "irr",ratio_class = True, remove='30_70', save=True, name="MH")
#   
#   
# #===========================================================================
# # var vs Longwave radiation desvio by class of ratio
# #===========================================================================
#     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT'
#     station9 = LCB_station(Path)
#     data9 = station9.getData(['Sm m/s','Ta C', 'Ev hpa'], by='H')
#     
#     From='2014-11-01 00:00:00'
#     To='2016-01-01 00:00:00'
#  
#     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT'
#     station = LCB_station(Path)
#     data = station.getData(['Sm m/s','Ta C', 'Ev hpa'],From=From, To=To, by='H')
#     data = data.between_time('03:00','05:00')
#     data = data.resample('D', how='mean')
# #     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C08.TXT'
# #     station8 = LCB_station(Path)
# #     data = station.getData(['Sm m/s','Ta C', 'Ev hpa'], by='H')
# #        
# #     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C15.TXT'
# #     station7 = LCB_station(Path)
# #     data = station.getData(['Sm m/s','Ta C', 'Ev hpa'], by='H')
# #        
# #     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C15.TXT'
# #     station_15 = LCB_station(Path)
# #     data_15 = station_15.getData(['Ta C', 'Ev hpa'], by='H')
#       
#       
# #     dir_inpath = '/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
# #     grad = Gradient(dir_inpath)
# #     grad.couples_net([['valley','slope']])
# #     data_grad = grad.grad(var=['Ta C'], by = "H", From ='2014-11-01 00:00:00', To = '2015-10-01 00:00:00' , return_=True)
# #     
# #     valley_slope= data_grad['valley_slope2014-11-01 00:00:00']
# #     valley_slope = valley_slope.between_time('03:00','04:00')
# #     valley_slope = valley_slope.resample('D', how='mean')
# #        
# #     irr.calc_ILw(data['Ev hpa'],data['Ta C'])
#     ratio =  irr.ratio()
#     ratio = ratio[np.isfinite(ratio)]
#     ratio = ratio.between_time('08:00','10:00')
#     ratio = ratio.resample('D', how='mean')
#     ratio.name = 'ratio'
#     data = pd.concat([data, ratio], axis=1, join ='inner')
#      
# #===============================================================================
# # Scatter plot of variable vs Longwave incoming radiation filtered by wind
# # #===============================================================================
# #  
# #        
# #     wind = station.getData(['Sm m/s'])
# #     wind8 = station8.getData(['Sm m/s'])
# #     wind7 = station7.getData(['Sm m/s'])
# #       
# #     V =  station9.getData(['V m/s'])
# #     speed =  station9.getData(['Sm m/s'])
# # #  
# # #     wind = wind['Sm m/s'][V[(V['V m/s']<0) & (speed['Sm m/s']<15)].index] # select only wind from the NOrth
# #     wind = pd.concat([wind,wind8,wind7],axis=1, join ='inner').max(axis=1)
# #     wind.name = 'Sm m/s'
# #     print V<0
# #     wind = wind.between_time('03:00','05:00')
# #     wind = wind.resample('D', how='mean')
# #        
# #     valley_slope = pd.concat([valley_slope,wind],axis=1, join = 'inner')
# #     print valley_slope.columns
# # #     
# # #     wind_count = station.getData(['Sm m/s'])
# # #     wind_count = wind_count.resample('H', how='mean')
# # #     wind_count = wind_count.between_time('18:00','05:00')
# # #     wind_count.index = wind_count.index+pd.offsets.Hour(6)
# # #     wind_count = wind_count[wind_count<4]
# # #     wind_count = wind_count.resample('D', how='count')
# # #     print wind_count
# #    
# #   
# #   
# #     Ilw = irr.Ilw.between_time('03:00','05:00')
# #     Ilw = Ilw.resample('D', how='mean')
# # #     Ilw = Ilw.multiply(wind_count['Sm m/s'], axis=0) # POWER
# # #
# # #     Ta9 = station.getData(['Ta C'])
# # #     Ta9 = Ta9.between_time('04:00','05:00')
# # #     Ilw = Ta9.resample('D', how='mean')
# #     Ilw.columns = ['longwave']
# #     data = pd.concat([valley_slope, Ilw], axis=1, join ='inner')
# 
# 
# #===============================================================================
# # nigth time lapse rate article
# #===============================================================================
# 
#     dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     AttSta = att_sta()
#     AttSta.setInPaths(dirInPath)
#          
#     station_names = AttSta.stations(['Head','slope'])
#     station_names = station_names + AttSta.stations(['Head','valley'])
#     station_names.remove('C11')
# 
#     Files =AttSta.getatt(station_names,'InPath')
#     altanal = AltitudeAnalysis(Files, net="LCB")
#      
# #     lp = altanal.Lapserate(var='Ta C',return_=True,delta=True,hasconst=False,  From='2014-11-01 00:00:00', To='2016-01-01 00:00:00')
# #     print lp
# #     
# #     lapserate_West=  altanal._lapserate(var='Ua g/kg').mean(axis=1)
# #===============================================================================
# # Lapse rate
# #===============================================================================
#         
# #     dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
# #     AttSta = att_sta()
# #     AttSta.setInPaths(dirInPath)
# #       
# #     # West
# #     station_names =AttSta.stations(['Head','East','valley'])
# #     station_names.extend(AttSta.stations(['Head','East','slope']))
# # #     station_names.remove('C08')
# #          
# #           
# #     Files =AttSta.getatt(station_names,'InPath')
# #     altanal = AltitudeAnalysis(Files, net="LCB")
# #          
#     lapserate_West=  altanal._lapserate(var='Ta C').mean(axis=1)
#     print lapserate_West
#     lapserate_West = lapserate_West['2014-11-01 00:00:00': '2016--01 00:00:00']
#     lapserate_West = lapserate_West.between_time('03:00','05:00')
#     lapserate_West = lapserate_West.resample('D').mean()
#     print lapserate_West
#     lapserate_West.name = 'lapserate'
# #      
# #     # East
# #     station_names =AttSta.stations(['Head','East','valley'])
# #     station_names.extend(AttSta.stations(['Head','East','slope']))
# #     station_names.remove('C14')
# # #     station_names.remove('C10')
# #       
# #     Files =AttSta.getatt(station_names,'InPath')
# #     altanal = AltitudeAnalysis(Files)
#           
# #     lapserate_East=  altanal._lapserate(var='Theta C').mean(axis=1)
# #     lapserate_East = lapserate_East.between_time('03:00','05:00')
# #     lapserate_East = lapserate_East.resample('D')
# #     lapserate_East.name = 'lapserate'
# #      
#           
# #     lapserate = pd.concat([lapserate_West], axis=1, join='inner').mean(axis=1)
# #     print lapserate
# #     lapserate.name = 'lapserate'
#           
#     data = pd.concat([data, lapserate_West], axis=1, join = 'inner')
#   
# #===============================================================================
# # Shear
# #===============================================================================
#          
# #     dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
# #     AttSta = att_sta()
# #     AttSta.setInPaths(dirInPath)
# #         
# #     station_names =AttSta.stations(['Head','West','slope'])
# #     station_names.extend(AttSta.stations(['Head','East','slope']))
# #     station_names.append('C09')
# #     station_names.append('C15')
# #     Files =AttSta.getatt(station_names,'InPath')
# #     altanal = AltitudeAnalysis(Files)
# #     shear=  altanal._lapserate(var='Sm m/s').mean(axis=1)
# #     shear = shear.between_time('04:00','05:00')
# #     shear = shear.resample('D')
# #     shear.name = 'shear'
# #     print shear
# #     data = pd.concat([data, shear], axis=1, join = 'inner')
# #     print data
#     
# #===============================================================================
# # wind sta9 vs DT
# #===============================================================================
# #     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT'
# #     station09 = LCB_station(Path)
# #     wind_valley = station09.getData('Ta C')
# #     wind_valley = wind_valley.between_time('03:00','05:00')
# #     wind_valley = wind_valley.resample('D')
# #     wind_valley.columns = ["valley"]
#        
# #     data = pd.concat([data, wind_valley], axis=1, join = 'inner')
# #     print data.columns
#       
# #     lcbplot = LCBplot() # get the plot object
# #     argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
# #     arglabel = lcbplot.getarg('label')
# #     argticks = lcbplot.getarg('ticks')
# #     argfig = lcbplot.getarg('figure')
# #     arglegend = lcbplot.getarg('legend')
#       
#     plt.close()
#     fig = plt.figure()
#       
#     plt.axvline(x=0,color='k', linewidth=5, alpha=0.5)
#     plt.axhline(y=5,color='k',linestyle='--', linewidth=5, alpha=0.5)
#     cmap = plt.get_cmap('RdBu_r')
#     plt.scatter(data['lapserate']*100, data['Sm m/s'], c = data['ratio'],cmap=cmap, s=50)
#     print data['lapserate']
#     cbar = plt.colorbar()
#     cbar.ax.tick_params(labelsize=40) 
#     plt.xticks()
#     plt.yticks()
#     plt.xlabel('lapse rate',  fontsize=30)
#     plt.ylabel('wind speed',  fontsize=30)
#     plt.tick_params(axis='both', which='major', labelsize=30)
#     cbar.ax.tick_params(axis='both', which='major', labelsize=30)
# #     plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#     plt.grid(True)
#     plt.savefig("/home/thomas/lapserate_ratio_night.svg", transparent=True)
#  
#  
 
 
#===============================================================================
# Divergence
#===============================================================================
#     dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     outpath="/home/thomas/"
#     AttSta = att_sta()
#     AttSta.setInPaths(dirInPath)
#       
#     net_West = LCB_net()
#     net_East = LCB_net()
#       
#       
#     files_west = AttSta.getatt(AttSta.stations(['Head','West','valley']),'InPath')
#     files_west = files_west + AttSta.getatt(AttSta.stations(['Head','West','slope']),'InPath')
#       
#     files_east = AttSta.getatt(AttSta.stations(['Head','East','valley']),'InPath')
#     files_east = files_east + AttSta.getatt(AttSta.stations(['Head','East','slope']),'InPath')
#   
#       
#     net_West.AddFilesSta(files_west)
#     net_East.AddFilesSta(files_east)
#   
#     Div = Divergence(net_West, net_East)
#       
#     conv_spring = Div.div( From='2014-10-01 00:00:00', To='2016-01-01 00:00:00')
#     print "spring"
#     print conv_spring.index
# #    
# #    
# #  
# # #     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT'
# # #     station9 = LCB_station(Path)
# # #     data = station9.getData(['Sm m/s','Ta C', 'Ev hpa'], by='H')
# # #     irr.calc_ILw(data['Ev hpa'],data['Ta C'])
# #  
#     irr.var_vs_ratio(data_grad=conv_spring,kind = "irr",ratio_class = True, remove='30_70',save=True, sci=True, name="Divergence")



