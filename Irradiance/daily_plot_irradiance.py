from __future__ import division
import pandas as pd
import glob
import datetime
import matplotlib.pyplot as plt
from LCBnet_lib import *
from grad_stations import Gradient
from divergence import Divergence 
from LapseRate import AltitudeAnalysis
from Irradiance_sim_obs import LCB_Irr

if __name__=='__main__':
    #===========================================================================
    # read
    #===========================================================================
    irr = LCB_Irr()
    inpath_obs = '/home/thomas/PhD/obs-lcb/LCBData/obs/Irradiance/data/'
    files_obs = glob.glob(inpath_obs+"*")
    irr.read_obs(files_obs)
    
#     
#     irr.data_obs['Pira_397'].plot()
#     plt.show()
    inpath_sim='/home/thomas/Irradiance_rsun_lin2_lag0_glob_df.csv'
    irr.read_sim(inpath_sim)

    df = irr.concat()
    print df
    
    df.plot()
    plt.show()
#   
#     #===========================================================================
#     # Quantiles plot
#     #===========================================================================
#   
    irr.plot_quantiles(kind ='Irr', save=True)
