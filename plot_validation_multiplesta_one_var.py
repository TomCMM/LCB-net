

import pandas as pd
import clima_lib.LCBnet_lib as lcb
import matplotlib.pyplot as plt
import glob
from arps_lib.netcdf_lib import *
from clima_lib.Irradiance.irr_lib import LCB_Irr 
import matplotlib 
from clima_lib.LCBnet_lib import *


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)
 
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['axes.labelsize'] = 8
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['legend.fontsize'] = 8
# plt.rc('legend',fontsize=20)


if __name__ == '__main__':

    attributes = pd.DataFrame(columns=['Lat', 'Lon'])
# # # #===============================================================================
# # # # Attribute
# # # #===============================================================================
    obs_path = "/home/thomas/phd/obs/staClim/inmet/obs_clean/INMET/"
    stanames = ["A509",'A706','A728', 'A701']
        
    AttSta = lcb.att_sta()
    AttSta.setInPaths(obs_path)
    stalat = AttSta.getatt(stanames, 'Lat')
    stalon = AttSta.getatt(stanames, 'Lon')
#     attributes[stanames] = [stalat, stalon]
    files =AttSta.getatt(stanames,'InPath')
    
    net = LCB_net()
    net.AddFilesSta(files, net='INMET')
    print stalon
    df_obs = net.getvarallsta(var='Ta C')
    print df_obs
#===============================================================================
# Get var ARPS
#=============================================================================== 
#     sims = ['out9kmk1','out9kmk2','out9kmk3','out9kmk4', 'out9kmk5', 'out9kmk6', 'out9kmk7', 'out9kmk8', 'out9kmk9']
  
    sims = ['out1kmk3']
    sim_path="/dados3/soc/s5oc/"
      
    for sim in sims:
        Files=glob.glob(sim_path+sim+"/*")
              
        Files.sort()
        Files = Files[:-1]
        varnames = ['Tc']
  
        ARPS = arps()
        BASE = BaseVars(Files[0],"ARPS")
                    
        SPEV = SpecVar()
        ARPS.load(BASE)
        ARPS.load(SPEV)
        ARPS.showvar()
      
        idx = ARPS.get_gridpoint_position(stalat, stalon)
        print idx
        Serie = netcdf_serie(Files,'ARPS')
            
        df_sim = pd.DataFrame()          
        for var in varnames:
            df = Serie.getdfmap(var,select_points=[[0]*len(idx),[0]*len(idx), idx['i'].values, idx['j'].values])
            print df
            df_sim = pd.concat([df_sim, df],join='outer', axis=1)
  
    df_sim.columns = AttSta.getatt(stanames, 'Nome')
    df_obs.columns = AttSta.getatt(stanames, 'Nome')

    fig, ax = plt.subplots()
    colors = ['r','b','g','c']
    df_obs.loc[df_sim.index, :].plot(ax =ax, linestyle='-', colors = colors)
    df_sim.plot(ax =ax, linestyle='--', colors = colors)
    plt.show()
         
         
         
         
         
         