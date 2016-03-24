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
from LCBnet_lib import *
from grad_stations import Gradient
from divergence import Divergence 
from LapseRate import AltitudeAnalysis

from LCBnet_lib import *
from divergence import Divergence
from grad_stations import Gradient
#------------------------------------------------------------------------------ 
class LCB_Irr():
    """
    DESCRIPTION
        Read the data of Irradiance and return a dataframe
    INPUT
        inpath: list 
        file's Path from data logger
    EXAMPLE
        'ID','Year','Day','Hour','Pira_397','Pira_369','Tlogger','LoggerV'
        105;2015;71;2;0;0;18.72;12.34
    """
    def __init__(self):
        self.param_obs = {}
        self.param_sim = {}

    def read_obs(self, inpaths, interpolate =None):
        """
        DESCRIPTION
            Read the file of the irradaiance observed and save the data into a dataframe
        INPUT
            The paths of the files
        
        OUTPUT
            save the dataframe in self.data_obs
        """
        self.param_obs['inpaths'] = inpaths
        
        if not isinstance(inpaths,list):
            raise TypeError('Should be a list')
        for inpath in inpaths:
            try:
                newdata_obs = pd.read_csv(inpath, sep =None)
                newdata_obs.columns=['ID','Year','Day','Hour','Pira_397','Pira_369','Tlogger','LoggerV']
                data_obs = pd.concat([data_obs, newdata_obs], ignore_index=True)
            except UnboundLocalError:
                data_obs = pd.read_csv(inpath, sep =None)
                data_obs.columns=['ID','Year','Day','Hour','Pira_397','Pira_369','Tlogger','LoggerV']

        data_obs = self._newindex(data_obs)
        
        data_obs = data_obs[(data_obs['Pira_369']>0) & (data_obs['Pira_369']<1500)] # small filter
        
        data_obs = data_obs.resample('min').mean() # resample in minute
        print data_obs
        if interpolate:
    #         data_obs.groupby([data_obs.index.day]).transform(lambda x: x.fillna(x.mean()))
            data_obs = data_obs.interpolate(method='time') # interpolat
        
        
        self.data_obs = data_obs

    def read_sim(self,inpath):
        """
        DESCRIPTION
            Read the data from the Rsun model
        Return
            write the data into a dataframe and stock it at self.data_sim
        """

        Isim=pd.read_csv(inpath, index_col=0, parse_dates=True)

        self.data_sim = Isim

    def calc_ILw(self,em, T, return_=False):
        """
        DESCRIPTION
                Calculate the downard radiation in W m-2
                Brutsaert WH (1975) on a derivable formula for long-wave 
                long-wave radiation from clear skies.
                Water Ressourses
        INPUT
                em: capor pressure hpa
                T: temperature degree
        OUTPUT
                Longwave radiation
        """
        STEFAN= 5.67051 * 10 **-8
        em = em * 10**2 # convert in PA
        T= T + 273.15# convert in Kelvin
        E=0.643*(em/T)**(1/7)
        L = E*STEFAN*T**4
        
        L = pd.DataFrame(L)
        L.columns = ['longwave']
        
        if return_:
            return L
        else:
            self.Ilw = L

    def desvio_ILWmean(self, by='H'):
        """
        DESCRIPTION
            Calculate the desvio of the ILW mean in percent
        Return a serie
        """
        ilw = self.Ilw
        ilw = ilw.resample( by)
        ilw['Hours'] = ilw.index.hour
        self._meanilw = ilw.groupby(lambda x: x.hour).mean()
        ilw = ilw.apply(self._desvio, axis=1)
        desvio = ilw['longwave']
        desvio = pd.DataFrame(desvio)
        desvio.columns = ['longwave']
        return desvio

    def _desvio(self,x):
        mean_Ilw =  self._meanilw['longwave'][ self._meanilw['Hours'] == x['Hours']]
        res = ((x['longwave'] - mean_Ilw)/x['longwave']) * 100
        return res.values

    def plot_ratio(self,kind ='Irr', by='H',  save= False, outpath = "/home/thomas/"):
        lcbplot = LCBplot() # get the plot object
        argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
        arglabel = lcbplot.getarg('label')
        argticks = lcbplot.getarg('ticks')
        argfig = lcbplot.getarg('figure')
        arglegend = lcbplot.getarg('legend')
        
        plt.figure(**argfig)

        desvio = self.desvio_ILWmean()
        print desvio
        desvio = desvio.resample(by)
        desvio['hours'] = desvio.index.hour
        
        desvio.plot(kind='scatter', x='hours', y='longwave', alpha=0.05, c='k', s=30)
        plt.xlim(0,23)
        plt.ylim(-30,20)

        plt.ylabel("Incoming Longwave radiation (w/m2)", **arglabel)
        plt.xlabel( "Hours", **arglabel)
        plt.grid(True)
        plt.tick_2params(axis='both', which='major', **argticks)
        plt.tick_params(axis='both', which='major', **argticks)
        plt.legend(**arglegend)
        
        if not save:
            plt.show()
        else:
            plt.savefig(outpath+kind+"_quantile.png")

    def concat(self, pyrnometer="Pira_369", calc_irr_staname='C05'):
        """
        DESCRIPTION
            Concatenate the observation and the simulation on the same dataframe
            
        INPUT
            The name of the pyranometer to be used
        """

        Iobs = self.data_obs[pyrnometer]
        Isim = self.data_sim[calc_irr_staname]
  

        df = pd.concat([ Iobs, Isim], axis=1, join = "outer")
        df.columns = ['Obs','Sim']

        return df

    def ratio(self, by ="H", how='mean', interpolate=None):
        """
        DESCRIPTION
            Return the ratio between clear sky simulated irradiance and observed irradianec
        """
        Irr = self.concat()
        Irr = Irr.resample(by).mean()
        Irr = Irr.dropna(how='any')
        ratio = Irr['Obs'] / Irr['Sim']
        ratio.columns = ['Ratio']
        if interpolate:
            print "I have interpolated"
            old_index = ratio.index
            ratio = ratio[np.isfinite(ratio)]
            ratio = ratio.reindex(old_index)
            ratio = ratio.interpolate()

        return ratio

    def plot_quantiles(self,kind ='Irr', save= False, outpath = "/home/thomas/"):
        """
        DESCRITPION
            make a plot of the mean diurnal cycle of the irradiation simulated
            and observated in Ribeirao Das Posses
        """
        lcbplot = LCBplot() # get the plot object
        argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
        arglabel = lcbplot.getarg('label')
        argticks = lcbplot.getarg('ticks')
        argfig = lcbplot.getarg('figure')
        arglegend = lcbplot.getarg('legend')

        fig=plt.figure(**argfig)

        
        c_summer = 'r'
        c_winter = 'b'

        if kind == 'Irr':
            df = self.concat()
            summer = df["2014-10":"2015-04"]
            summer2 = df["2015-10":"2016-04"]
            summer = summer.append(summer2)

            winter = df["2015-04":"2015-10"]
            summer.columns = ['Obs_summer','Sim_summer']
            winter.columns = ['Obs_winter','Sim_winter']
            
        if kind == 'Ilw':
            df = self.Ilw
            summer = df[start_summer:start_winter]
            winter = df[start_winter:end_winter]
            summer.columns = ['Obs_summer']
            winter.columns = ['Obs_winter']

        summer = summer.groupby(summer.index).first()
        winter = winter.groupby(winter.index).first()
# 
#         summer.plot()
#         plt.show()
#         winter.plot()
#         plt.show()
        
        mean = pd.concat([summer, winter], axis=1)

        
        
        quartile1 = mean.groupby(lambda x: x.hour).quantile(q=0.10)
        quartile3 = mean.groupby(lambda x: x.hour).quantile(q=0.90)
        quartile2 = mean.groupby(lambda x: x.hour).quantile(q=0.50)
        mean = mean.groupby(lambda x: x.hour).mean()
        
        
        quartile1 = quartile1.fillna(0)
        quartile3 = quartile3.fillna(0)
        quartile2 = quartile2.fillna(0)
        mean = mean.fillna(0)



        plt.fill_between(quartile1['Obs_summer'].index.values, quartile1['Obs_summer'].values, quartile3['Obs_summer'].values, alpha=0.1,color=c_summer)
        plt.fill_between(quartile1['Obs_winter'].index.values, quartile1['Obs_winter'].values, quartile3['Obs_winter'].values, alpha=0.1,color=c_winter)
        plt.plot([], [], color='r', alpha=0.1,linewidth=10, label='Obs S q=0.90 & 0.10')
        plt.plot([], [], color='b',alpha=0.1, linewidth=10, label='Obs W q=0.90 & 0.10')
        

        plt.plot(quartile2['Obs_summer'].index.values, quartile2['Obs_summer'].values,linewidth = 10, linestyle='-', color=c_summer, alpha=0.7, label='Obs W median')
        plt.plot(quartile2['Obs_winter'].index.values, quartile2['Obs_winter'].values,linewidth =10, linestyle='-', color=c_winter, alpha=0.7, label='Obs W median')

#         plt.plot(mean['Obs_summer'].index.values, mean['Obs_summer'].values,linewidth = 2, linestyle='--', color=c_summer, alpha=0.7, label='Obs S mean')
#         plt.plot(mean['Obs_winter'].index.values, mean['Obs_winter'].values,linewidth = 2, linestyle='--', color=c_winter, alpha=0.7, label='Obs W mean')


        if kind == 'Irr':
                plt.plot(mean['Sim_summer'].index.values, mean['Sim_summer'].values,linewidth =10, linestyle='--', color=c_summer, alpha=0.7, label='Sim S median')
                plt.plot(mean['Sim_winter'].index.values, mean['Sim_winter'].values,linewidth = 10, linestyle='--', color=c_winter, alpha=0.7, label='Sim W median')
                plt.ylabel(r"Irradiance ($w.m^{-2}$)", **arglabel)

        if kind == 'Ilw':
                plt.ylabel("Incoming Longwave radiation (w/m2)", **arglabel)
        
        plt.xlabel( "Hours", **arglabel)
        plt.grid(True, color="0.5")
        plt.tick_params(axis='both', which='major', **argticks)
        plt.tick_params(axis='both', which='major', **argticks)
#         plt.legend(**arglegend)
        
        if not save:
            plt.show()
        else:
            plt.savefig(outpath+kind+"_quantile.svg", transparent=True)

    def var_vs_ratio(self,kind = "irr", data_grad=None, ratio_class = None, interpolate=None, save=None, remove=None, sci=None, name =None):
        """
        DESCRIPTION
            plot ratio in function of an other variable
        INPUT 
            A Serie or dataframe with the variable(s) to be plot in function of the ratio
            kind: irr: irradiance
                 ilw: incoming longwave radiation
            remove: to remove one label
            sci: To put the y labels in scinetific notation 
        """

        if not ratio_class:
            if kind == "ilw":
                ratio = self.desvio_ILWmean()
            if kind == "irr":

                ratio = self.ratio()

            df = pd.concat([ratio, data_grad],axis=1, join='inner')
            df.columns = ["ratio", "var_gradient"]
            
            df.plot(kind='scatter', x='ratio', y='var_gradient')
            plt.show()
        else:
            print "Class Ratio"
            if kind == "ilw":
                ratio = self.desvio_ILWmean()
                
                ratio_labels=['-10_5','-5_5','5_10'] # this should be dynamic
                ratio_class= [-10,-5,5,10] # this should be dynamic
    
                ratio_cut = pd.Series(pd.cut(ratio['longwave'], bins=ratio_class, labels=ratio_labels),index=ratio.index)
                df = pd.concat([ratio_cut, data_grad],axis=1, join='inner')
                df.columns = ['ratio', 'grad']
        
                for i,v in enumerate(ratio_labels):
                    select = df['grad'][df['ratio'] == v]
                    select = select.groupby(lambda t: (t.hour)).mean()
                    plt.plot(select.index, select, label=v)
                
                plt.axhline(y=0,color='k') # horizontal line at 0
                plt.xlim((6,18))
        #         plt.ylim((-2,2))
                
                plt.legend()


            if kind == "irr":
                print "Irradiance"
                ratio = self.ratio(interpolate=interpolate)
                ratio = ratio*100
                
                ratio_labels=['0_30','30_70','70_100']
                ratio_class= [0,30,70,100] # this should be dynamic

                ratio_cut = pd.Series(pd.cut(ratio, bins=ratio_class, labels=ratio_labels),index=ratio.index)
                df = pd.concat([ratio_cut, data_grad],axis=1, join='inner')
                df.columns = ['ratio', 'grad']
                df = df.between_time('07:00','18:00')

        
                lcbplot = LCBplot() # get the plot object
                argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
                arglabel = lcbplot.getarg('label')
                argticks = lcbplot.getarg('ticks')
                argfig = lcbplot.getarg('figure')
                arglegend = lcbplot.getarg('legend')
                
                plt.close()
                fig = plt.figure(**argfig)
                plt.xlim([7,18])
                color=iter(['b','r'])
                for i,v in enumerate(ratio_labels):
                    select = df['grad'][df['ratio'] == v]
                    select = select.groupby(lambda t: (t.hour)).mean()
                    print v
                    if v != remove:
                        ax = plt.plot(select.index, select, label=v,c=next(color), linewidth=10)
                plt.xticks(**arglabel)
                plt.yticks(**arglabel)
                plt.xlabel('hours (h)', **arglabel)
                plt.ylabel('Difference', **arglabel)
                
                plt.axhline(y=0,color='k') # horizontal line at 0
                
                plt.legend(prop={'size':20})
                if sci:
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                
            plt.grid(True)
            if save:
                if name:
                    plt.savefig(outpath+name+"_var_vs_ratio.svg", transparent=True)
                else:
                    plt.savefig(outpath+"var_vs_ratio.svg", transparent=True)
                
            else:
                plt.show()
                
    def _newindex(self,data_obs):
        """
        Convert the data_obs logger date into a datetime time serie index
        """
        # creating index
        newindex = [ ]
        for i in data_obs.index:
            hour = data_obs['Hour'][i]
            hour = str(hour).zfill(4)[0:2]
            if hour == '24':
                hour ='00'
            hour = int(hour)
            minute = data_obs['Hour'][i]
            minute = int(str(minute).zfill(4)[2:4])
            year = int(data_obs['Year'][i])
            day = int(data_obs['Day'][i])

            date=datetime.datetime(year,1,1,hour ,minute) + datetime.timedelta(day-1)
            newindex.append( date )

        data_obs['newindex']=newindex
        data_obs=data_obs.set_index('newindex')

        return data_obs



if __name__=='__main__':
#     #===========================================================================
#     # read
#     #===========================================================================
    irr = LCB_Irr()
    inpath_obs = '/home/thomas/PhD/obs-lcb/LCBData/obs/Irradiance/data/'
    files_obs = glob.glob(inpath_obs+"*")
    print files_obs
    irr.read_obs(files_obs)
        
    inpath_sim='/home/thomas/Irradiance_analysis/Irradiance_rsun_lin2_lag-0.2_glob_df.csv'
    irr.read_sim([inpath_sim])
     
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
#     dir_inpath = '/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     outpath = '/home/thomas/'
#    
#     grad = Gradient(dir_inpath)
#     grad.couples_net([['West', 'East'],['valley','slope'], ['Medio', 'Head','valley']])
#     data_grad = grad.grad(var=['Ua g/kg'], by = "H", From ='2014-11-01 00:00:00', To = '2016-01-01 00:00:00' , return_=True)
#    
#     valley_slope= data_grad['valley_slope2014-11-01 00:00:00']
#     irr.var_vs_ratio(data_grad=valley_slope,kind = "irr",ratio_class = True, remove='30_70', save=True, name="VS")
#     
#     west_east= data_grad['West_East2014-11-01 00:00:00']
#     irr.var_vs_ratio(data_grad=west_east,kind = "irr",ratio_class = True, remove='30_70', save=True, name="WE")
#     
#     medio_head= data_grad['Medio_Head_valley2014-11-01 00:00:00']
#     irr.var_vs_ratio(data_grad=medio_head,kind = "irr",ratio_class = True, remove='30_70', save=True, name="MH")
#   
#   
# #===========================================================================
# # var vs Longwave radiation desvio by class of ratio
# #===========================================================================
#     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT'
#     station9 = LCB_station(Path)
#     data9 = station9.getData(['Sm m/s','Ta C', 'Ev hpa'], by='H')
#     
    From='2014-11-01 00:00:00'
    To='2016-01-01 00:00:00'

    Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT'
    station = LCB_station(Path)
    data = station.getData(['Sm m/s','Ta C', 'Ev hpa'],From=From, To=To, by='H')
    data = data.between_time('03:00','05:00')
    data = data.resample('D', how='mean')
#     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C08.TXT'
#     station8 = LCB_station(Path)
#     data = station.getData(['Sm m/s','Ta C', 'Ev hpa'], by='H')
#        
#     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C15.TXT'
#     station7 = LCB_station(Path)
#     data = station.getData(['Sm m/s','Ta C', 'Ev hpa'], by='H')
#        
#     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C15.TXT'
#     station_15 = LCB_station(Path)
#     data_15 = station_15.getData(['Ta C', 'Ev hpa'], by='H')
     
     
#     dir_inpath = '/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     grad = Gradient(dir_inpath)
#     grad.couples_net([['valley','slope']])
#     data_grad = grad.grad(var=['Ta C'], by = "H", From ='2014-11-01 00:00:00', To = '2015-10-01 00:00:00' , return_=True)
#     
#     valley_slope= data_grad['valley_slope2014-11-01 00:00:00']
#     valley_slope = valley_slope.between_time('03:00','04:00')
#     valley_slope = valley_slope.resample('D', how='mean')
#        
#     irr.calc_ILw(data['Ev hpa'],data['Ta C'])
    ratio =  irr.ratio()
    ratio = ratio[np.isfinite(ratio)]
    ratio = ratio.between_time('08:00','10:00')
    ratio = ratio.resample('D', how='mean')
    ratio.name = 'ratio'
    data = pd.concat([data, ratio], axis=1, join ='inner')
    
#===============================================================================
# Scatter plot of variable vs Longwave incoming radiation filtered by wind
# #===============================================================================
#  
#        
#     wind = station.getData(['Sm m/s'])
#     wind8 = station8.getData(['Sm m/s'])
#     wind7 = station7.getData(['Sm m/s'])
#       
#     V =  station9.getData(['V m/s'])
#     speed =  station9.getData(['Sm m/s'])
# #  
# #     wind = wind['Sm m/s'][V[(V['V m/s']<0) & (speed['Sm m/s']<15)].index] # select only wind from the NOrth
#     wind = pd.concat([wind,wind8,wind7],axis=1, join ='inner').max(axis=1)
#     wind.name = 'Sm m/s'
#     print V<0
#     wind = wind.between_time('03:00','05:00')
#     wind = wind.resample('D', how='mean')
#        
#     valley_slope = pd.concat([valley_slope,wind],axis=1, join = 'inner')
#     print valley_slope.columns
# #     
# #     wind_count = station.getData(['Sm m/s'])
# #     wind_count = wind_count.resample('H', how='mean')
# #     wind_count = wind_count.between_time('18:00','05:00')
# #     wind_count.index = wind_count.index+pd.offsets.Hour(6)
# #     wind_count = wind_count[wind_count<4]
# #     wind_count = wind_count.resample('D', how='count')
# #     print wind_count
#    
#   
#   
#     Ilw = irr.Ilw.between_time('03:00','05:00')
#     Ilw = Ilw.resample('D', how='mean')
# #     Ilw = Ilw.multiply(wind_count['Sm m/s'], axis=0) # POWER
# #
# #     Ta9 = station.getData(['Ta C'])
# #     Ta9 = Ta9.between_time('04:00','05:00')
# #     Ilw = Ta9.resample('D', how='mean')
#     Ilw.columns = ['longwave']
#     data = pd.concat([valley_slope, Ilw], axis=1, join ='inner')
       
#===============================================================================
# Lapse rate
#===============================================================================
       
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)
     
    # West
    station_names =AttSta.stations(['Head','East','valley'])
    station_names.extend(AttSta.stations(['Head','East','slope']))
#     station_names.remove('C08')
        
         
    Files =AttSta.getatt(station_names,'InPath')
    altanal = AltitudeAnalysis(Files)
        
    lapserate_West=  altanal._lapserate(var='Ua g/kg').mean(axis=1)
    lapserate_West = lapserate_West.between_time('03:00','05:00')
    lapserate_West = lapserate_West.resample('D')
    lapserate_West.name = 'lapserate'
#      
#     # East
#     station_names =AttSta.stations(['Head','East','valley'])
#     station_names.extend(AttSta.stations(['Head','East','slope']))
#     station_names.remove('C14')
# #     station_names.remove('C10')
#       
#     Files =AttSta.getatt(station_names,'InPath')
#     altanal = AltitudeAnalysis(Files)
        
#     lapserate_East=  altanal._lapserate(var='Theta C').mean(axis=1)
#     lapserate_East = lapserate_East.between_time('03:00','05:00')
#     lapserate_East = lapserate_East.resample('D')
#     lapserate_East.name = 'lapserate'
#      
        
#     lapserate = pd.concat([lapserate_West], axis=1, join='inner').mean(axis=1)
#     print lapserate
#     lapserate.name = 'lapserate'
        
    data = pd.concat([data, lapserate_West], axis=1, join = 'inner')
 
#===============================================================================
# Shear
#===============================================================================
        
#     dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     AttSta = att_sta()
#     AttSta.setInPaths(dirInPath)
#         
#     station_names =AttSta.stations(['Head','West','slope'])
#     station_names.extend(AttSta.stations(['Head','East','slope']))
#     station_names.append('C09')
#     station_names.append('C15')
#     Files =AttSta.getatt(station_names,'InPath')
#     altanal = AltitudeAnalysis(Files)
#     shear=  altanal._lapserate(var='Sm m/s').mean(axis=1)
#     shear = shear.between_time('04:00','05:00')
#     shear = shear.resample('D')
#     shear.name = 'shear'
#     print shear
#     data = pd.concat([data, shear], axis=1, join = 'inner')
#     print data
   
#===============================================================================
# wind sta9 vs DT
#===============================================================================
#     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT'
#     station09 = LCB_station(Path)
#     wind_valley = station09.getData('Ta C')
#     wind_valley = wind_valley.between_time('03:00','05:00')
#     wind_valley = wind_valley.resample('D')
#     wind_valley.columns = ["valley"]
     
#     data = pd.concat([data, wind_valley], axis=1, join = 'inner')
#     print data.columns
    
    lcbplot = LCBplot() # get the plot object
    argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
    arglabel = lcbplot.getarg('label')
    argticks = lcbplot.getarg('ticks')
    argfig = lcbplot.getarg('figure')
    arglegend = lcbplot.getarg('legend')
    
    plt.close()
    fig = plt.figure(**argfig)
    
    plt.axvline(x=0,color='k', linewidth=5, alpha=0.5)
    plt.axhline(y=5,color='k',linestyle='--', linewidth=5, alpha=0.5)
    cmap = plt.get_cmap('RdBu_r')
    plt.scatter(data['lapserate']*100, data['Sm m/s'], c = data['ratio'],cmap=cmap, s=100)
    print data['lapserate']
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40) 
    plt.xticks(**arglabel)
    plt.yticks(**arglabel)
    plt.xlabel('lapse rate', **arglabel)
    plt.ylabel('wind speed', **arglabel)
    plt.tick_params(axis='both', which='major', **argticks)
    cbar.ax.tick_params(axis='both', which='major', **argticks)
#     plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.grid(True)
    plt.savefig("/home/thomas/lapserate_ratio_night.svg", transparent=True)




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
#    
#    
#  
# #     Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT'
# #     station9 = LCB_station(Path)
# #     data = station9.getData(['Sm m/s','Ta C', 'Ev hpa'], by='H')
# #     irr.calc_ILw(data['Ev hpa'],data['Ta C'])
#  
#     irr.var_vs_ratio(data_grad=conv_spring,kind = "irr",ratio_class = True, remove='30_70',save=True, sci=True, name="Divergence")



