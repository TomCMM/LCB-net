#===============================================================================
# DESCRIPTION
#    Analysis of the lapse rate in the Ribeirao das Posses
#===============================================================================

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


def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

class AltitudeAnalysis():
    """
    Methods to analysis and plot variable in function of the alitude
    """
    def __init__(self,Files, **kwargs):
        self.kwargs = kwargs
        self.sunrise = "21:00"
        self.sunset = "20:00"
        self.net=LCB_net()
        self.net.AddFilesSta(Files)
        self.attsta = att_sta() # when I am using it from somewhere elese I canot use it that it why I put it there
        
        self.elev = self.attsta.sortsta(self.net.getpara('stanames'), 'Altitude')['metadata']  # This dosen't seems to be dynamic
        self.stations = self.attsta.sortsta(self.net.getpara('stanames'), 'Altitude')['stanames']  # This dosen't seems to be dynamic
        self.stas_by_elev = self.net.getsta([],all = True, sorted = 'Altitude')['stanames']

    def PblHeight(self, option ="sbl", altref = "sealevel", plot =False):
        """
        DESCRIPTION:
            Calculate the PBL height using different method
            
            option = "sblpt" calculate the PBL height using the SBLpT method 
            described on the article M. Collaud Coen et al 2014.
            It detect when the gradient of potential temperature vanish on the space.
            I modified it to take the minimum gradient observed

            option = "sbl" calculate the PBL height using the SBI method 
            described on the article M. Collaud Coen et al 2014. and that I modified a bit
            It check when the temperature start to decrease with height.
            The value observed are smooth to avoid the natural spatial variation of the temperature
            
            plot == True: return a daily plot 
            plot= "boxplot": return an annual boxplot 
        """

        pbl_height = self._gradient(option)
        if plot == True:
            pbl_height = pbl_height[pbl_height['sbl_height'].apply(np.isreal)]
            df = pd.DataFrame(pbl_height['sbl_height'], index =pbl_height.index)
            grouped = df.groupby(lambda x: x.hour)
            grouped.mean().plot()
            plt.show()

        elif plot == "boxplot":
            params = {'legend.fontsize': 30,'legend.linewidth': 3}
            plt.rcParams.update(params)
            pbl_height = pbl_height.resample("D")
            pbl_height = pbl_height.drop(['d_theta_d_z'], axis=1)
            pbl_height['Months'] = pbl_height.index.month
            pbl_height.boxplot(by=['Months'])
            plt.xlabel("Month",fontsize=30)
            plt.ylabel('Stable boundary layer height above valley center (m)',fontsize=30)
            plt.tick_params(axis='both', which='major', labelsize=30)
            plt.tick_params(axis='both', which='major', labelsize=30)
            plt.show()

        elif plot=="boxplot_state":

            pbl_height['Months'] = pbl_height.index.month
            pbl_height['values']  = 1
            pbl_height['sbl_height']=pbl_height['sbl_height'].apply(self._replace)
            pivoted = pd.pivot_table(pbl_height, values= 'values', index='Months', columns='sbl_height', aggfunc=np.sum)
            pivoted.plot(kind='bar')
            plt.show()

        elif plot=="boxplot_height":

            pbl_height = pbl_height.resample('D')
            pbl_height['Months'] = pbl_height.index.month

            pbl_height.boxplot(column='sbl_height', by=['Months'])
            plt.show()
        else:
            return pbl_height

    def VarVsAlt(self, From=None, To=None, desvio_padrao=None, desvio_sta9= None, dates=None, vars=['Ta C'], by = False, every = False, return_data = False, **kwargs):
        """
        DECRIPTION
            dates: list of integer representing the date to be ploted.
                it should be concordant with the option "by" which determine the frequency
            var: list of variables
            by: aggregate the data. 
                'M', month
                'D', day
                'H', hour
                'T', minute
            every: if true, return  a panel every <value>. 
                    if false return the overall period mean
                'M', month
                'D', day
                'H', hour
            return_data: if True return the data, 
                        if False create the attribut self.var_vs_alt
        NOTE
            I should clean this methods 
        """
        Ink('VarVsAlt', 0, kwargs)

        var_vs_alt = {}
        for var in vars:
            data_panel = None
            for staname in self.stas_by_elev:
                station= self.net.getsta([staname])[0]
                if desvio_sta9:
                    sta9 = self.net.getsta(['C15'])[0]
                if every:
                    dataframe  = station.getData(var=var, by=by, every=every, From=From, To=To)
                    if desvio_padrao:
                        dataframe = dataframe - self.net.getData(var=var, by=by, every=every, From=From, To=To)
                    if desvio_sta9:
                        dataframe = dataframe - sta9.getData(var=var, by=by, every=every, From=From, To=To)
                else:
                    dataframe  = station.getData(var=var, group = by, From=From, To=To)
                    if desvio_padrao:
                        dataframe = dataframe - self.net.getData(var=var, group = by, From=From, To=To)
                    if desvio_sta9:
                        dataframe = dataframe - sta9.getData(var=var, group = by, From=From, To=To)
                if dates != None:
                    dataframe = dataframe.iloc[dates,:]
                else:
                    dataframe = dataframe.iloc[:,:]
#                 data_dic[staname] = dataframe
                if isinstance(data_panel, pd.Panel):
#                     print "="*80
#                     print ("feeding the panel")
#                     dataframe
                    data_panel[staname] = dataframe
#                     print data_panel[staname]
                else:
#                     print "="*80
#                     print ('Initialising')
#                     print dataframe
                    data_panel = pd.Panel({staname:dataframe})
#                     print data_panel[staname]

#             data_panel = pd.Panel(data_dic)
                print data_panel
            data_transposed = data_panel.transpose(2,1,0)
            
            var_vs_alt[var] = data_transposed
        self.var_vs_alt = var_vs_alt

        if return_data:
            return self.var_vs_alt

    def _annotate_var_vs_elev(self, x, y, s):
        for i in range(len(x)):
            plt.annotate(s[i], 
                 xy=(x[i],y[i][0]),  
                 xycoords='data')

    def _markers(self, stanames):
        markers = []
        for s in stanames:
            if s in self.attsta.stations(['Head', 'West']):
                markers.append('s')
            else:
                markers.append('o')
        return markers

    def plot(self, analysis, data = None, annotate = False, desvio_padrao = None, join = False, marker_side = None, plot_mean_profile=None, **kwargs):
        """
        DESCRIPTION
            Plot an analysis
        INPUT
            analysis: "var_vs_alt"
            Join: True, make all the plot on the same figure
            kwargs 
                grey= None
                log = None 
                grey = None 
                xlabel = None 
                print_= None
            marker_side: TRUE, use different marker to separate the East and the West face
            mean_profile: TRue, draw the mean profile of the period selected
        """
        Ink('Altitude Analysis Plot', 0, kwargs)
        lcbplot = LCBplot() # get the plot object
        attvar = AttVar() # get the variable attribute object 
        kwargs = merge_dicts(self.kwargs,kwargs) # merge the kwargs to pass the option between the class and the method
        
        argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
        arglabel = lcbplot.getarg('label')
        argticks = lcbplot.getarg('ticks')
        argfig = lcbplot.getarg('figure')
        arglegend = lcbplot.getarg('legend')


        if analysis == "var_vs_alt":
            if not data:
                data = [self.var_vs_alt]

            elev = self.elev
            if join:
                argplot['linestyle'] = "-"
                argplot['linewidth'] = 0.6
                argplot['alpha'] = 0.7

            for var in data[0].keys():
                if join:
                    plt.figure(**argfig)
                    plt.legend(**arglegend)

                mean_profiles = []
                colo = ["b", "r"]
                for col,var_vs_alt in zip(colo, data):
                    print"0"*80
                    print len(data)
                    if kwargs.get('grey', False): # in the get the second value is returner if grey doss not exist
                        color = iter(["#990000", "#000080"])
                        marker = iter(["o",'s'])
    
                    if join:
                        color=iter(cm.RdBu(np.linspace(0,1,len(var_vs_alt[var].items)*len(hours))))

                    if plot_mean_profile:
                        mean_profiles.append(var_vs_alt[var].mean(axis='items'))
                        color=iter(col*len(var_vs_alt[var].items)*len(hours))
    
                    for item in var_vs_alt[var].items:
                        dataframe = var_vs_alt[var][item]
    
                        if not join:
                            plt.figure(**argfig)
                            color=iter(cm.RdBu(np.linspace(0,1,len(dataframe.index))))
    
                        for hour in dataframe.index:
                            nbsta = len(dataframe.loc[hour,:])
                            c = next(color)
                            Ink(elev,2,kwargs)
                            Ink(nbsta,2,kwargs)
    
                            if marker_side:
                                m = self._markers(dataframe.loc[hour,:].index)
                                for i,v in enumerate(dataframe.loc[hour,:]):
                                    plt.plot(v, elev[i], marker=m[i], c=c, label=str(hour), **argplot)
                            else:
                                plt.plot(dataframe.loc[hour,:], elev, c=c, label=str(hour), **argplot)

                            if desvio_padrao:
                                plt.axvline(0)
    
                            plt.grid(True)
                            plt.xlabel(attvar.getatt(var, 'longname')[0], **arglabel)
                            plt.ylabel("Altitude (m)", **arglabel)
                            plt.tick_params(axis='both', which='major', **argticks)
                            plt.tick_params(axis='both', which='major', **argticks)
                        
                        if annotate:
                            self._annotate_var_vs_elev(dataframe.loc[hour,:], elev, self.stations)
                        if kwargs.get('log', False):
                            plt.yscale('log')
    
                        if not join:
                            plt.legend(**arglegend)
                            if kwargs.get('print_', False):
                                Ink('Plot',2,kwargs)
                                outpath = lcbplot.getarg('OutPath')
                                plt.savefig(outpath+str(var[0:2])+"_"+str(item[1])+"_AltitudeAnalysis.png")
                                plt.close()
                            else:
                                plt.show()
                if plot_mean_profile:
                    colo = ["b", "r"]
                    for c,mean_profile in zip(colo, mean_profiles):
                        plt.plot(mean_profile.loc[hour,:], elev, c=c, label=str(hour),linewidth=4)

                if join:
                    if kwargs.get('print_', False):
                        Ink('Plot',2,kwargs)
                        outpath = lcbplot.getarg('OutPath')
                        plt.savefig(outpath+str(var[0:2])+"_"+str(item[1])+"_AltitudeAnalysis.png")
                        plt.close()
                    else:
                        plt.show()

    def Lapserate(self, var = 'Ta C', return_=False):
        """
        Plot lapserate of a variable
        """
        for f in Files:
            hours = range(0,24,2)
            lr = [ ]
            for hour in hours:
                elev = [ ]
                data = [ ]
                stations = self.net.getsta('', all=True, sorted='Altitude')
                for station,staname in zip(stations['stations'],stations['stanames']) :
                    elev.append(self.attsta.getatt(staname,'Altitude'))
                    d = station.getData(var = var, group = 'H')
                    data.append(d[var][hour])
                X = sm.add_constant(elev)
                est = sm.OLS(data, X).fit()
                lr.append(est.params[1])
            
            plt.plot(hours,lr,'o')
        
        plt.show()
        return np.array(lr).mean()

    def Lapserate_boxplot(self, var = 'Ta C', how = 'mean'):
        """
        Box plot monthly lapse rate
        """
        for f in [Files]:
            net=LCB_net()
            net.AddFilesSta(f)
            df = pd.DataFrame()
            elev = [ ]
            for staname in self.net.getsta([],all = True, sorted = 'Altitude').keys():
                elev.append(self.attsta.getatt(staname,'Altitude'))
                station= self.net.getsta([staname])[0]
                s = pd.Series(station.getData(var = "Ta C", by = 'D', how = how),name = staname)
                s.names = staname
                df = pd.concat([df,s], axis=1, )

        LR = df.apply(self.__f, axis=1)
        LR_month = pd.DataFrame(LR,columns=['LR'])
        LR_month['Months']= LR_month.index.month

        LR_month.boxplot(by='Months')
        plt.show()

    def __f(self,x):
        elev = self.elev
        stations = self.elevation
        X = sm.add_constant(elev)
        Y = [x[stations[0]],x[stations[1]],x[stations[2]],x[stations[3]],
             x[stations[4]],x[stations[5]],x[stations[6]],x[stations[7]],
             x[stations[8]],x[stations[9]],x[stations[10]],x[stations[11]]]
        est = sm.OLS(Y, X).fit()
        return est.params[1]



#===============================================================================
#  Main
#===============================================================================

    def _replace(self, value):
        if type(value) != str:
            return "sbl_in_basin"
        else:
            return value

    def _find_z_sbl(self, value, elev):
        """
        DESCRIPTION
            0) make a moving average 
            1) find the first negative lapserate and the last positive lapserate
            2) sort the lapse rate to be interpolated
            2) linear interpolation betweem the selected stations
            
        NOTE
            If their is two negative value of dt_dz dans la serie
            il va selectionner preferentiellement celle qui est la plus proche de zero
            l utilisation du genre "cubic" pour l interpolation donne des resultats incoherent
        """
        window = 2 # moving average window
        Threeshold_dt_dz = -0.0025 # C/m threeshold minimum to define a lapserate 
        
        print("----------------------------------------------------------------------")
        print("----------------------------------------------------------------------")
        print("------elevation------")
        print(elev)
        print("------values------")
        print(value)
        value_array = np.array(value)
#        value_array = np.insert(value_array, 0, value_array[0])# add value at the begining to perform the moving average
#        value_array = np.insert(value_array, -1, value_array[-1]) # add value at the begining to perfor the moving average windows = 3
#        value_array = movingaverage(value_array,(window)) # perform a moving average
#        value_array = movingaverage(value_array,window)# perform twice the something 
#        print "moving average" + str(value_array) 
        if all(v>Threeshold_dt_dz for v in value_array):
            print "over the ridge"
            return "over_ridge"
#             return 1400
        elif value_array[0] < 0 or value_array[1] < 0:
            print value_array[0]
            print "no sbl"
            return "no_sbl"
#             return np.nan

    
        else:
            print "inside the valley"
            index_first_negative = np.where( value_array < Threeshold_dt_dz)
            index_first_negative = index_first_negative[0][0]
            last_positive = int(index_first_negative -1)
            print "index_first_negative: " + str(index_first_negative)
            

            select_value = value_array[last_positive:index_first_negative+1]
            select_elev = elev[last_positive:index_first_negative+1]
            
            print "selected value" + str(select_value)
            while select_value[0] < 0:
                try:
                    print('last couple stations still in the cold pool')
                    index_first_negative = index_first_negative-1
                    last_positive = last_positive-1
                     
                    select_value = value_array[last_positive:index_first_negative+1]
                    select_elev = elev[last_positive:index_first_negative+1]
                    print "New index_first_negative: " + str(index_first_negative)
                    print "new selected value: " + str(select_value)
                except IndexError:
                    print('I have not found couple station lapserate with positive value -> no cold pool')
                    return np.nan
                    break


            print index_first_negative
            sorted_value, sorted_elev = zip(*sorted(zip(select_value, select_elev)))
            f = interp1d(sorted_value, sorted_elev, kind='linear')
            dt_dz=0

            print("------Zsbl------")
            print(f(dt_dz))
            
            
            return f(dt_dz)

    def _gradient(self,option):
        """
            Calculate the height of the PBL using the Gradient Method
        """
        print('Determining the height of the Nocturnal boundary layer')
        
        # Getting the basic variables necessary for the analysis
        dic = self.attsta.sortsta(self.net.getpara("stanames"), "Altitude")# I think this should be remove
        stanames = self.stations
        elev = self.elev

        # use the sblpt method
        if option =='sblpt':
            theta = self.net.getvarallsta(var=['Theta C'], stanames=stanames, by='H')
            d_theta = theta.transpose().diff()
            d_theta = d_theta.transpose()
    
            new_col_names = []
            alt_couple = {}
            for first, second in zip(d_theta.columns[0:-1], d_theta.columns[1::]):
                couplename = first+"_"+second
                new_col_names.append(couplename)
                alt_couple[couplename] = np.mean([self.attsta.getatt(first, "Altitude")[0], self.attsta.getatt(second, "Altitude")[0]]) 
    
            d_theta = d_theta.drop(d_theta.columns[0], axis=1) # drop the first columns which as nan now
            d_theta.columns = new_col_names
            d_z = np.diff(np.array(elev).flatten())
            lapserate = d_theta / d_z
    
            lapserate = lapserate.dropna(how='all')
            lapserate = lapserate.apply(np.abs)
            d_theta_d_z = lapserate.min(axis=1)
            station_couple = lapserate.idxmin(axis=1)
            couple_alt = pd.Series([alt_couple[sta] for sta in station_couple], index = station_couple.index)
            couple_alt = couple_alt - np.min(elev)

            df_results = pd.concat([d_theta_d_z, station_couple, couple_alt], axis=1)
            df_results.columns=['d_theta_d_z', "station_couple", "couple_alt"]
    
            mask = (df_results.index.hour > self.sunset) | (df_results.index.hour < self.sunrise )
            df_results = df_results[mask]
            return df_results

        # use the SBI method to detect the PBL
        if option == 'sbl':
            var = ['Ta C']
            lapserate = self._lapserate(var)
            lapserate = lapserate.dropna(how='any')

            couple_elev = (np.array(elev)[1:] + np.array(elev)[:-1])/2 # calculate the mean elevation of the couple where the lapse rate has been calculated
            couple_elev = couple_elev.flatten()
            pbl_height = lapserate.apply(self._find_z_sbl, elev=couple_elev, axis=1)
            data = np.array(pbl_height)
            df_pbl_height = pd.DataFrame(data, index = pbl_height.index, columns=['sbl_height'])
            df_pbl_height['hours'] = df_pbl_height.index.hour
#             df_pbl_height['sbl_height'] = df_pbl_height['sbl_height'].astype(float)
            df_pbl_height = df_pbl_height.between_time(self.sunset, self.sunrise)
#             mask = (df_pbl_height.index.hour > self.sunset) | (df_pbl_height.index.hour < self.sunrise )
#             df_pbl_height = df_pbl_height[mask]
            return df_pbl_height

    def _lapserate(self,var):
        """
        Return the lapse rate between each station 
        """
        # Getting the basic variables necessary for the analysis
#         dic = self.attsta.sortsta(self.net.getpara("stanames"), "Altitude")# I think this should be remove
        stanames = self.stations
        elev = self.elev
    
        variable = self.net.getvarallsta(var=var, stanames=stanames, by='H')
#         variable = variable.between_time('18:00','00:00')
#         print variable
#         variable = variable.resample('D')
        
        d_variable = variable.transpose().diff()
        d_variable = d_variable.transpose()
        
        new_col_names = []
        alt_couple = {}
        for first, second in zip(d_variable.columns[0:-1], d_variable.columns[1::]):
            couplename = first+"_"+second
            new_col_names.append(couplename)
            alt_couple[couplename] = np.mean([self.attsta.getatt(first, "Altitude")[0], self.attsta.getatt(second, "Altitude")[0]]) 

        d_variable = d_variable.drop(d_variable.columns[0], axis=1) # drop the first columns which as nan now
        d_variable.columns = new_col_names
        d_z = np.diff(np.array(elev).flatten())
        
        lapserate = d_variable / d_z
        
        return lapserate


if __name__=='__main__':
    #===========================================================================
    #  Get input Files
    #===========================================================================
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)
    
    
    station_names =AttSta.stations(['Head'])
#     station_names.append('C17')


    Files =AttSta.getatt(station_names,'InPath')
#     Files = Files + AttSta.getatt(AttSta.stations(['Medio','East']),'InPath')
    
    altanal = AltitudeAnalysis(Files)

    #===========================================================================
    # Plot var in function of Altitude
    #===========================================================================

    hours = np.arange(15,24,1)
    hours = [16, 18, 20,22,0,2, 4]
    altanal.VarVsAlt(vars= ['Ta C'], by= 'H',  dates = hours, From='2015-03-01 00:00:00', To='2015-08-01 00:00:00')
    altanal.plot(analysis = 'var_vs_alt', marker_side = True, annotate = True, print_= True)

    #===========================================================================
    # Plot var in function of Altitude - Mean summer and Winter
    #===========================================================================

#     hours = [10,12,14]
#     altanal.VarVsAlt(vars= ['Ta C', 'Ua g/kg', 'Theta C', 'Sm m/s'], by= 'H',  dates = hours, From='2014-10-01 00:00:00', To='2015-08-01 00:00:00')
#     altanal.plot(analysis = 'var_vs_alt', annotate = True, print_= True)


#     plt.figure()
#     East =AttSta.getatt(AttSta.stations(['Head','slope']),'InPath')
#     East = East + AttSta.getatt(AttSta.stations(['Head','ridge']),'InPath')
#     
#     West =AttSta.getatt(AttSta.stations(['Head','slope']),'InPath')
#     West = West + AttSta.getatt(AttSta.stations(['Head','ridge']),'InPath')


    #===========================================================================
    # Box plot
    #===========================================================================
#     altanal.Lapserate(var='Sm m/s')
    #Altanal.Lapserate_boxplot()

    #===========================================================================
    # PBL HEIGHT
    #===========================================================================
    # Graph Hourly
#     plt.close('all')
    #altanal.PblHeight(option='sblpt', plot="boxplot")
#     altanal.PblHeight(option='sbl', plot='boxplot_state')
#     altanal.PblHeight(option='sbl', plot="boxplot_height")
#     altanal.PblHeight(option='sbl', plot=True)

    #===========================================================================
    #Plot "trajectories"
    #===========================================================================

#     hours = [12]
#     varvsalt_winter = altanal.VarVsAlt(vars= ['Ta C', 'Ua g/kg', 'Theta C', 'Sm m/s'], desvio_sta9=True, by= 'H',every='D',dates=hours, From='2015-05-01 00:00:00', To='2015-08-01 00:00:00', return_data=True)
#     varvsalt_summer = altanal.VarVsAlt(vars= ['Ta C', 'Ua g/kg', 'Theta C', 'Sm m/s'], desvio_sta9=True, by= 'H',every='D',dates=hours, From='2014-11-01 00:00:00', To='2015-05-01 00:00:00', return_data=True)
# 
#     altanal.plot(analysis = 'var_vs_alt', data=[varvsalt_summer, varvsalt_winter], print_= True, desvio_padrao=True, join= True, plot_mean_profile=True)





























