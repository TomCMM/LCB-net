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

from scipy.optimize import curve_fit

def pol(x, a, b):
    """
    Polynomial function
    """
    return a*x**2 + b*x 

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
        self.net.AddFilesSta(Files, net = self.kwargs['net'])
        self.net.report()
        self.attsta = att_sta() # when I am using it from somewhere elese I canot use it that it why I put it there
        self.Files = Files
        self.elev = self.attsta.sortsta(self.net.getpara('stanames'), 'Alt')['metadata']  # This dosen't seems to be dynamic
        self.stations = self.attsta.sortsta(self.net.getpara('stanames'), 'Alt')['stanames']  # This dosen't seems to be dynamic
        self.stas_by_elev = self.net.getsta([],all = True, sorted = 'Alt')['stanames']

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

    def VarVsAlt(self, From=None, To=None, From2=None, To2=None, desvio_padrao=None, desvio_sta= None, 
                 dates=None, vars=['Ta C'], by = False, every = False, return_data = False,recalculate=False,quantile=False, **kwargs):
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
            desvio_sta: <str> name of the station to make the difference with. 
            return_data: if True return the data, 
                        if False create the attribut self.var_vs_alt
            recalculate: if True, recalculate the variable
        NOTE
            I should clean this methods 
        """
        Ink('VarVsAlt', 0, kwargs)

        var_vs_alt = {}
        for var in vars:
            data_panel = None
            for staname in self.stas_by_elev:
                station= self.net.getsta([staname])[0]
                if desvio_sta:
                    sta9 = self.net.getsta([desvio_sta])[0]

                if every:
                    dataframe  = station.getData(var=var, by=by, every=every, From=From, To=To, From2=From2, To2=To2, recalculate=recalculate)

                    if desvio_padrao:
                        dataframe = dataframe - self.net.getData(var=var, by=by, every=every, From=From, To=To, From2=From2, To2=To2, recalculate=recalculate)
                    if desvio_sta:
                        dataframe = dataframe - sta9.getData(var=var, by=by, every=every, From=From, To=To, From2=From2, To2=To2, recalculate=recalculate)
                elif quantile:
                    print 'Quantile'
                    dataframe  = station.getData(var=var, by = by, From=From, To=To, From2=From2, To2=To2, recalculate=recalculate)
                    
                    if desvio_padrao:
                        dataframe = dataframe - self.net.getData(var=var, by = by, From=From, To=To, From2=From2, To2=To2, recalculate=recalculate)
                    if desvio_sta:
                        dataframe = dataframe - sta9.getData(var=var, by = by, From=From, To=To, From2=From2, To2=To2, recalculate=recalculate)
                    dataframe = dataframe.groupby(lambda t: (t.hour)).quantile(quantile).unstack(level=-1)
                    print dataframe
                else:
                    dataframe  = station.getData(var=var, group = by, From=From, To=To, From2=From2, To2=To2, recalculate=recalculate)
                    
                    if desvio_padrao:
                        dataframe = dataframe - self.net.getData(var=var, group = by, From=From, To=To, From2=From2, To2=To2, recalculate=recalculate)
                    if desvio_sta:
                        dataframe = dataframe - sta9.getData(var=var, group = by, From=From, To=To, From2=From2, To2=To2, recalculate=recalculate)

#                 print dataframe                
                if dates != None:
                    try:
                        dataframe = dataframe.iloc[dates,:]
                    except IndexError:
                        dataframe = pd.DataFrame({var: pd.Series([np.nan]*len(dates),index=dates)})
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
#                 print data_panel
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

    def plot(self, analysis, data = None, annotate = False, desvio_padrao = None, join = False, profile=None, marker_side = None,
              plot_mean_marker=None, plot_mean_profile=None, delta=None, hasconst=True, polyfit=False, marker=False,  **kwargs):
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
            hasconstant, False, put the origina of the linear regression to zero
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

            # get len hours
            len_hours = data[0].itervalues().next().shape[1]
#             print "Nb of hours selected: " + str(len_hours)


            elev = self.elev
            if delta:
#                 mp = mp - mp[delta]
                low_elev = self.attsta.getatt(delta, 'Alt')
                elev = np.array(elev) - np.array(low_elev)
            if join:
                argplot['linestyle'] = "-"
                argplot['linewidth'] = 0.75
                argplot['alpha'] = 0.7

            for var in data[0].keys():
#                 if join:
#                     plt.figure(**argfig)
#                     plt.legend(**arglegend)

                mean_profiles = []
#                 colo = ["b", "r",'g','y']
                colo = ["b", "r",'g','y']
                marko =['^','o']

                for col,mar, var_vs_alt in zip(colo,marko, data):
#                     print"0"*80
#                     print len(data)
                    if kwargs.get('grey', False): # in the get the second value is returner if grey doss not exist
                        color = iter(["#990000", "#000080"])

                    if join:
                        color=iter(cm.RdBu(np.linspace(0,1,len(var_vs_alt[var].items)*len_hours)))
                        color = iter(['b','r','g'])

                    if plot_mean_profile or plot_mean_marker:
                        print "PLOT MEAN PROFILE"
                        mean_profiles.append(var_vs_alt[var].mean(axis='items'))
                        color=iter(col*len(var_vs_alt[var].items)*len_hours)
                    
                    if marker:
                        mark= iter(['o', '^','^','^'])
                        color = iter(['0.95','0.75','b','r'])
                    
                    for item in var_vs_alt[var].items:
                        dataframe = var_vs_alt[var][item]
                        
#                         if not join:
#                             plt.figure(**argfig)
#                             color=iter(cm.RdBu(np.linspace(0,1,len(dataframe.index))))
    
                        
                        for hour in dataframe.index:
                            nbsta = len(dataframe.loc[hour,:])
                            
                            if marker:
                                m = next(mark)
                            c = next(color)
                             
                            Ink(elev,2,kwargs)
                            Ink(nbsta,2,kwargs)
    
                            if marker_side:
                                m = self._markers(dataframe.loc[hour,:].index)
                                for i,v in enumerate(dataframe.loc[hour,:]):
                                    plt.plot(elev[i], v, marker=m[i], c=c, label=str(hour), **argplot)

                            if profile:
                                plt.plot(elev, dataframe.loc[hour,:] , c=c, label=str(hour), **argplot)

                            if marker:
                                plt.plot(elev, dataframe.loc[hour,:] ,marker=mar, c=c, label=str(hour),linestyle='None', markersize=10)
                
                            if desvio_padrao:
                                plt.axvline(0, c='0.8')
                            if delta:
                                plt.axhline(0, c='0.8')
    
                            if marker_side or profile:
                                plt.xlabel(attvar.getatt(var, 'longname')[0], **arglabel)
                                plt.ylabel("Altitude (m)", **arglabel)
                                plt.tick_params(axis='both', which='major', **argticks)
                                plt.tick_params(axis='both', which='major', **argticks)
                            
                            plt.grid(True)
                            plt.yticks(size=16)
                            plt.xticks(size=16)
#                             
#                             if hour == 12 and var =="Ta C":
#                                 plt.xlim([-1,6])
#                                 
#                             if hour == 12 and var =="Theta C":
#                                 plt.xlim([-4,5])
#                                 
#                             if hour == 12 and var =="Ua g/kg":
#                                 plt.xlim([-1.5,2.5])
#                                 
#                             if hour == 12 and var =="Sm m/s":
#                                 plt.xlim([-8,2])
#                                 
#                             if hour == 4 and var =="Ta C":
#                                 plt.xlim([-12,3])
#                                 
#                             if hour == 4 and var =="Theta C":
#                                 plt.xlim([-22,3])
#                                 
#                             if hour == 4 and var =="Ua g/kg":
#                                 plt.xlim([-2,2])
#                                 
#                             if hour == 4 and var =="Sm m/s":
#                                 plt.xlim([-8,5])
                                
                        if annotate:
                            self._annotate_var_vs_elev(elev, dataframe.loc[hour,:], self.stations)
                        if kwargs.get('log', False):
                            plt.yscale('log')
    
                        if not join:
#                             plt.legend(**arglegend)
                            if kwargs.get('print_', False):
                                Ink('Plot',2,kwargs)
                                outpath = lcbplot.getarg('OutPath')
                                plt.savefig(outpath+str(var[0:2])+"_"+str(item[1])+"_AltitudeAnalysis.png")
                                plt.close()
                            else:
                                plt.show()
                if plot_mean_profile:
                    colo = ["b", "r",'g','y']
                    for c,mean_profile in zip(colo, mean_profiles):
                        plt.plot(elev, mean_profile.loc[hour,:], c='0.90',alpha=0.80, label=str(hour),linewidth=15)
                        plt.plot(elev, mean_profile.loc[hour,:], c=c, label=str(hour),linewidth=10)

                if plot_mean_marker:
                    #                 
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
#                     print mean_profiles
                    colo = ["b", "r",'k','k']
                    pos_vertical = [0.95,0.90, 0.85]
                    linestyle=['-',':','--']
                    mark = ['o', '^', 's']
                    for s,m, c,mean_profile, t_v in zip(linestyle, mark, colo, mean_profiles, pos_vertical):
                        mp = mean_profile.loc[hour,:]
                        elev = np.array(self.elev).flatten()
                        if delta:
                            mp = mp - mp[delta]
                            low_elev = self.attsta.getatt(delta, 'Alt')
                            elev = np.array(elev) - np.array(low_elev)
 
                        ax.scatter(elev,mp, c=c,marker=m, label=str(hour),s=100)
                         
                        if polyfit:
                            print elev
                            print mp
                            print type(elev)
                            print type(mp)
                            est, pcov = curve_fit(polyfit, elev, mp)
                        else:
                            if hasconst==True:
                                X = sm.add_constant(elev)
                                est = sm.OLS(mp, X).fit()
                            else:
                                est = sm.OLS(mp, elev, hasconst=False).fit()
                             
 
                        X_plot = np.linspace(elev.min(),elev.max(),100)
                         
                        if polyfit:
                                ax.plot(X_plot, est[0]* X_plot**2 + est[1]* X_plot, c=c, linestyle=s)
#                                 ax.text(0.8, t_v,str(est[0])+'x2 + '+str(est[1])+"x", horizontalalignment='center',
#                                     verticalalignment='center',color=c,transform = ax.transAxes)
                                 
                                text = r'${%.2e}x^{2} + {%.2e}x$' %(est[0], est[1]) 
                                ax.text(0.1, t_v,text, horizontalalignment='left',verticalalignment='center',color=c,transform = ax.transAxes, fontsize=16)
 
                        else:
                            if hasconst==True:
                                ax.plot(X_plot, X_plot*est.params[1] + est.params[0], c=c, linestyle=s)
                                ax.text(0.8, t_v,str(est.params[1])+'x + '+str(est.params[0]), horizontalalignment='center',
                                        verticalalignment='center',color=c,transform = ax.transAxes)
                            else:
                                ax.plot(X_plot, X_plot*est.params[0], c=c, linestyle=s)
  
                            e = est.params[0]*100
                             
                            if var=='Ta C':
                                text = r'${%.2f} C.100m^{-1}$' %e
                            if var=='Sm m/s':
                                text = r'${%.2f} m.s^{-1}.100m^{-1}$' %e
                            if var=='Theta C':
                                text = r'${%.2f} C.100m^{-1}$' %e
                            if var =='Ua g/kg':
                                text = r'${%.2f} g.kg^{-1}.100m^{-1}$' %e
                            ax.text(0.1, t_v,text, horizontalalignment='left',verticalalignment='center',color=c,transform = ax.transAxes, fontsize=16)
                        plt.grid()
                        plt.yticks(size=16)
                        plt.xticks(size=16)


                if join:
                    if kwargs.get('print_', False):
                        Ink('Plot',2,kwargs)
                        outpath = lcbplot.getarg('OutPath')
                        plt.savefig(outpath+str(var[0:2])+"_"+str(item[1])+"_AltitudeAnalysis.svg")
                        plt.close()
                    else:
                        plt.show()

    def Lapserate(self, var = 'Ta C', return_=False, outpath=None, From = None, To = None,
                  From2=None,To2=None, filter=None, delta=False , hasconst=True, hours = None):
        """
        Calcul the lapserate of a variable with the altitude of the stations passed
        
        parameters:
            delta: True, return the lapse of the difference of a variable 
            with the measurement made at the lowest stations  
            
            hasconst: True, make the linear regression pass by zero 
        """

        for f in self.Files:
            hours = range(0,24,1)
            lr = [ ]
            for hour in hours:
                elev = [ ]
                data = [ ]
                stations = self.net.getsta('', all=True, sorted='Alt')
                if delta:
                    lowest_sta = stations['stations'][0]
                    lowest_staname = stations['stanames'][0]
                    lowest_d = lowest_sta.getData(var = var, group = 'H', From=From, To=To, From2=From2, To2=To2)
                    lowest_elev = self.attsta.getatt(lowest_staname,'Alt')[0]
                
                for station, staname in zip(stations['stations'],stations['stanames']) :
                    
                    if delta:
                        e = self.attsta.getatt(staname,'Alt')[0] - lowest_elev
                        elev.append(e)
                    else:
                        elev.append(self.attsta.getatt(staname,'Alt'))
                    
                    d = station.getData(var = var, group = 'H', From=From, To=To, From2=From2, To2=To2)
                    if delta:
                        d = d-lowest_d
                    
#                     print d
                    data.append(d[var][hour])
                
                if hasconst:
                    X = sm.add_constant(elev)
                    est = sm.OLS(data, X).fit()
                    lr.append(est.params[1])
                else:
                    est = sm.OLS(data, elev, hasconst=False).fit()
                    lr.append(est.params[0])
            
            
        plt.plot(hours,lr)
        if not outpath:
            pass
        else:
            plt.savefig(outpath+"lapserate.png")

        if not return_:
            return np.array(lr).mean()
        else:
            return np.array(lr)

    def Lapserate_boxplot(self, var = 'Ta C', how = 'mean', outpath=None):
        """
        Box plot monthly lapse rate
        """
        for f in [Files]:
            net=LCB_net()
            net.AddFilesSta(f)
            df = pd.DataFrame()
            elev = [ ]
            for staname in self.net.getsta([],all = True, sorted = 'Alt').keys():
                print staname
                elev.append(self.attsta.getatt(staname,'Alt'))
                station= self.net.getsta([staname])[0]
                s = pd.Series(station.getData(var = "Ta C", by = 'D', how = how),name = staname)
                s.names = staname
                df = pd.concat([df,s], axis=1, )

        LR = df.apply(self.__f, axis=1)
        LR_month = pd.DataFrame(LR,columns=['LR'])
        LR_month['Months']= LR_month.index.month

        LR_month.boxplot(by='Months')
        if not outpath:
            plt.show()
        else:
            plt.savefig(outpath+"lapserate.png")

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
        dic = self.attsta.sortsta(self.net.getpara("stanames"), "Alt")# I think this should be remove
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
                alt_couple[couplename] = np.mean([self.attsta.getatt(first, "Alt")[0], self.attsta.getatt(second, "Alt")[0]]) 
    
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
#         dic = self.attsta.sortsta(self.net.getpara("stanames"), "Alt")# I think this should be remove
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
            alt_couple[couplename] = np.mean([self.attsta.getatt(first, "Alt")[0], self.attsta.getatt(second, "Alt")[0]]) 

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

#     station_names =AttSta.stations(['Head','West','valley'])
# #     station_names.append('C17')
# 
#     Files =AttSta.getatt(station_names,'InPath')
#     Files = Files + AttSta.getatt(AttSta.stations(['Head','West','slope']),'InPath')
#     
#     altanal = AltAnalysis(Files)


    #===========================================================================
    # Box plot lapserate
    #===========================================================================
#     altanal.Lapserate(var='Ta C', outpath='/home/thomas/Z_article/')
#     altanal.Lapserate_boxplot()

    #===========================================================================
    # PBL HEIGHT
    #===========================================================================
    # Graph Hourly
#     plt.close('all')
    #altanal.PblHeight(option='sblpt', plot="boxplot")
#     altanal.PblHeight(option='sbl', plot='boxplot_state')
#     altanal.PblHeight(option='sbl', plot="boxplot_height")
#     altanal.PblHeight(option='sbl', plot=True)

#     #===========================================================================
#     #Plot "trajectories" - 
#     #===========================================================================
  
#     station_names =AttSta.stations(['Head','East'])
#     Files =AttSta.getatt(station_names,'InPath')
#        
#     altanal = AltitudeAnalysis(Files, net='LCB')
#        
#     hours = [12]
#     
#     
#     varvsalt_summer = altanal.VarVsAlt(vars= ['Ta C', 'Ua g/kg', 'Theta C','Sm m/s'], desvio_sta='C15', by= 'H',
#                                        every='D',dates=hours, From='2014-11-01 00:00:00', To='2015-04-01 00:00:00', From2='2015-10-01 00:00:00', To2='2016-01-01 00:00:00', return_data=True)
#      
#     varvsalt_winter = altanal.VarVsAlt(vars= ['Ta C', 'Ua g/kg', 'Theta C','Sm m/s'], desvio_sta='C15', by= 'H',
#                                        every='D',dates=hours, From='2015-05-01 00:00:00', To='2015-10-01 00:00:00', return_data=True)
#           
#   
#     print 'alo'
#     print varvsalt_winter
#     altanal.plot(analysis = 'var_vs_alt', data=[varvsalt_winter, varvsalt_summer], print_= True,profile=True, desvio_padrao=True, join= True, plot_mean_profile=True)
#       
#     #===========================================================================
#     #Plot "trajectories" - article
#     #===========================================================================
#  
#     station_names =AttSta.stations(['Head','East'])
#     Files =AttSta.getatt(station_names,'InPath')
#        
#     altanal = AltitudeAnalysis(Files, net='LCB')
#        
#     hours = [12]
#     
#     
#     varvsalt_summer = altanal.VarVsAlt(vars= ['Ta C', 'Ua g/kg', 'Theta C','Sm m/s'], desvio_sta='C15', by= 'H',quantile=[0.1,0.9],
#                                        dates=hours, From='2014-11-01 00:00:00', To='2015-04-01 00:00:00', From2='2015-10-01 00:00:00', To2='2016-01-01 00:00:00', return_data=True)
#      
#     varvsalt_winter = altanal.VarVsAlt(vars= ['Ta C', 'Ua g/kg', 'Theta C','Sm m/s'], desvio_sta='C15', by= 'H',quantile=[0.1,0.9],
#                                        dates=hours, From='2015-05-01 00:00:00', To='2015-10-01 00:00:00', return_data=True)
#           
#   
#     print 'alo'
#     print varvsalt_winter
#     print varvsalt_summer
#     altanal.plot(analysis = 'var_vs_alt', data=[varvsalt_winter, varvsalt_summer], shade=True,
#                   print_= True,profile=False, desvio_padrao=True, join= True, plot_mean_profile=True)
#       
#===============================================================================
# PRINT LAPSE RATE 
#===============================================================================
 
 
#     station_names = AttSta.stations(['Head','slope'])
#     station_names = station_names + AttSta.stations(['Head','valley'])
#  
#      
#     Files =AttSta.getatt(station_names,'InPath')
#   
#     altanal = AltitudeAnalysis(Files)
#       
#     lp = pd.DataFrame()
#       
#     # Winter
#     lp['lp_Ta_winter'] = altanal.Lapserate(var="Ta C",return_=True,  From='2015-05-01 00:00:00', To='2015-08-01 00:00:00')
# #     lp['lp_Ua_winter'] = altanal.Lapserate(var="Ua g/kg",return_=True,  From='2014-11-01 00:00:00', To='2015-05-01 00:00:00')
# #     lp['lp_Sm_winter'] = altanal.Lapserate(var="Sm m/s",return_=True,  From='2014-11-01 00:00:00', To='2015-05-01 00:00:00')
# #     lp['Ta C_winter'] =  altanal.net.getData(var="Ta C", From='2014-11-01 00:00:00', To='2015-05-01 00:00:00')
#  
#     # Summer
#     lp['lp_Ta_summer'] = altanal.Lapserate(var="Ta C",return_=True,  From='2015-01-01 00:00:00', To='2015-04-01 00:00:00')
# #     lp['lp_Ua_summer'] = altanal.Lapserate(var="Ua g/kg",return_=True,  From='2015-05-01 00:00:00', To='2015-11-01 00:00:00')
# #     lp['lp_Sm_summer'] = altanal.Lapserate(var="Sm m/s",return_=True,  From='2015-05-01 00:00:00', To='2015-11-01 00:00:00')
# #     lp['Ta C_summer'] =  altanal.net.getData(var="Ta C", From='2015-05-01 00:00:00', To='2015-08-01 00:00:00')
#     
#     
#     lp['lp_Ta_spring'] = altanal.Lapserate(var="Ta C",return_=True,  From='2015-10-01 00:00:00', To='2016-01-01 00:00:00')
#     lp['lp_Ta_spring'] = altanal.Lapserate(var="Ta C",return_=True,  From='2014-10-01 00:00:00', To='2015-01-01 00:00:00')
# 
# 
# 
#     plt.show()
    # Filter wind
     
    # Filter radiation
#     print lp
#     print lp.mean(axis=0)


#===============================================================================
# plot East/West lapse rate Article
#===============================================================================
 
 
 
 
#     vars=['Ta C',"Ua g/kg", "Sm m/s" ]
#     for var in vars:
#          
#          
#         station_names = AttSta.stations(['Head','West','slope'])
#         station_names = station_names + AttSta.stations(['Head','West','valley'])
#         station_names = station_names + ['C10']
#         Files =AttSta.getatt(station_names,'InPath')
#         altanal = AltitudeAnalysis(Files, net="LCB")
#          
#         lp = pd.DataFrame()
#         lp['lp_Ta_summer_west'] = altanal.Lapserate(var=var,return_=True,  From='2014-11-01 00:00:00', To='2015-04-01 00:00:00', 
#                                                From2='2015-11-01 00:00:00',To2='2016-01-01 00:00:00')
#         lp['lp_Ta_winter_west'] = altanal.Lapserate(var=var,return_=True,  From='2015-04-01 00:00:00', To='2015-11-01 00:00:00')
#         lp['lp_Ta_spring_west'] = altanal.Lapserate(var=var,return_=True,  From='2014-11-01 00:00:00', To='2015-01-01 00:00:00', 
#                                                     From2='2015-11-01 00:00:00',To2='2016-01-01 00:00:00')
#         
#        
#         station_names = AttSta.stations(['Head','East','slope'])
#         station_names = station_names + AttSta.stations(['Head','East','valley'])
#         station_names.remove('C11')
#         if var== "Ua g/kg":
#             station_names.remove('C13')
# 
#         Files =AttSta.getatt(station_names,'InPath')
#         altanal = AltitudeAnalysis(Files, net="LCB")
#                 
#       
#         lp['lp_Ta_summer_east'] = altanal.Lapserate(var=var,return_=True,  From='2014-11-01 00:00:00', To='2015-04-01 00:00:00', 
#                                                From2='2015-11-01 00:00:00',To2='2016-01-01 00:00:00')
#         lp['lp_Ta_winter_east'] = altanal.Lapserate(var=var,return_=True,  From='2015-04-01 00:00:00', To='2015-11-01 00:00:00')
#         lp['lp_Ta_spring_east'] = altanal.Lapserate(var=var,return_=True,  From='2014-11-01 00:00:00', To='2015-01-01 00:00:00',
#                                                     From2='2015-11-01 00:00:00',To2='2016-01-01 00:00:00')
#           
#         lp = lp*100
#         lcbplot = LCBplot() # get the plot object
#         attvar = AttVar() # get the variable attribute object 
#           
#         argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
#         arglabel = lcbplot.getarg('label')
#         argticks = lcbplot.getarg('ticks')
#         argfig = lcbplot.getarg('figure')
#         arglegend = lcbplot.getarg('legend')
#           
#         ls = ["-",'-','-','--','--','--']
#         cs=['r','b','k','r','b','k']
#         plt.figure(**argfig)
#           
#         for c,l,col in zip(cs,ls,lp.columns):
# #             print col
#             plt.plot(lp.index,lp[col],c=c, linestyle=l, linewidth=5)
#         plt.axhline(y=0, color='k', alpha=0.5,linewidth=4) # environnemental lapse rate
#         if var == "Ta C":
#             plt.axhline(y=-0.65, color='0.5', alpha=0.5,linewidth=4) # environnemental lapse rate
#             plt.axhline(y=-0.98, color='0.5', alpha=0.5,linewidth=4) # adiabatic lapse rate
#         plt.grid(True)
#         plt.xlabel("Hours (h)", **arglabel)
#         plt.ylabel("Lapse rate", **arglabel)
#         plt.xticks(range(0,24,6))
#         plt.xlim([0,24])
#         plt.tick_params(axis='both', which='major', **argticks)
#         plt.tick_params(axis='both', which='major', **argticks)
#         plt.savefig("/home/thomas/lapserate_"+var[0:2]+".svg", transparent=True)
#         plt.close()
#  





#===============================================================================
# plot East/West lapse rate - DeltaZ
#===============================================================================
#  
 
 
  
#     vars=['Ta C']
#     for var in vars:
#           
#           
#         station_names = AttSta.stations(['Head','West','slope'])
#         station_names = station_names + AttSta.stations(['Head','West','valley'])
#         station_names = station_names + ['C10']
#  
#         Files =AttSta.getatt(station_names,'InPath')
#         altanal = AltitudeAnalysis(Files, net="LCB")
#           
#         lp = pd.DataFrame()
#         lp['lp_Ta_summer_west'] = altanal.Lapserate(var=var,return_=True,delta=True,hasconst=False,  From='2014-11-01 00:00:00', To='2015-04-01 00:00:00', 
#                                                 From2='2015-11-01 00:00:00',To2='2016-01-01 00:00:00')
#         lp['lp_Ta_winter_west'] = altanal.Lapserate(var=var,return_=True, delta=True,hasconst=False, From='2015-04-01 00:00:00', To='2015-11-01 00:00:00')
#         lp['lp_Ta_spring_west'] = altanal.Lapserate(var=var,return_=True,delta=True,hasconst=False,  From='2014-11-01 00:00:00', To='2015-01-01 00:00:00')
#            
#            
#         station_names = AttSta.stations(['Head','East','slope'])
#         station_names = station_names + AttSta.stations(['Head','East','valley'])
#         station_names.remove('C11')
#         station_names.remove('C13')
#         Files =AttSta.getatt(station_names,'InPath')
#         altanal = AltitudeAnalysis(Files, net="LCB")
#                     
#           
#         lp['lp_Ta_summer_east'] = altanal.Lapserate(var=var,return_=True, delta=True,hasconst=False, From='2014-11-01 00:00:00', To='2015-04-01 00:00:00', 
#                                                From2='2015-11-01 00:00:00',To2='2016-01-01 00:00:00')
#         lp['lp_Ta_winter_east'] = altanal.Lapserate(var=var,return_=True,delta=True,hasconst=False,  From='2015-04-01 00:00:00', To='2015-11-01 00:00:00')
#         lp['lp_Ta_spring_east'] = altanal.Lapserate(var=var,return_=True,delta=True,hasconst=False,  From='2014-11-01 00:00:00', To='2015-01-01 00:00:00')
#               
#         lp = lp*100
#         lcbplot = LCBplot() # get the plot object
#         attvar = AttVar() # get the variable attribute object 
#               
#   
#         ls = ["-",'-','-','--','--','--']
#         cs=['r','b','k','r','b','k']
#         plt.figure()
#               
#         for c,l,col in zip(cs,ls,lp.columns):
#             plt.plot(lp.index,lp[col],c=c, linestyle=l, linewidth=5)
#         plt.axhline(y=0, color='k', alpha=0.5,linewidth=4) # environnemental lapse rate
#         if var == "Ta C":
#             plt.axhline(y=-0.65, color='0.5', alpha=0.5,linewidth=4) # environnemental lapse rate
#             plt.axhline(y=-0.98, color='0.5', alpha=0.5,linewidth=4) # adiabatic lapse rate
#         plt.grid(True)
#         plt.xlabel("Hours (h)", fontsize=20)
#         plt.ylabel("Lapse rate", fontsize=20)
#         plt.xticks(range(0,24,6))
#         plt.xlim([0,24])
#         plt.tick_params(axis='both', which='major', labelsize=20)
#         plt.tick_params(axis='both', which='major', labelsize=20)
#         plt.savefig("/home/thomas/lapserate_"+var[0:2]+".svg", transparent=True)
#         plt.close()
# 
#
# #     #===========================================================================
# #     #Plot linear regression and lapse rate - delta
# #     #===========================================================================
 
    station_names =AttSta.stations(['Head']) 
#     station_names =AttSta.stations(['Head', 'slope'])
#     station_names =station_names + AttSta.stations(['Head', 'valley'])
#     station_names.remove('C13') # humidity 12h
    station_names.remove('C11')
      
#     station_names.remove('C10') # Temperature 12h
#     station_names.remove('C11')
#     station_names.remove('C08')
     
#     station_names.remove('C12') # Temperature 04h
#     station_names.remove('C11')
     
     
    Files =AttSta.getatt(station_names,'InPath')
          
    altanal = AltitudeAnalysis(Files, net='LCB')
          
    hours = [12]
       
       
    varvsalt_summer = altanal.VarVsAlt(vars= ['Ta C','Sm m/s','Ev hpa','Ua g/kg','Theta C', 'Pa H'], desvio_sta='C10', by= 'H',
                                       every='D',dates=hours, From='2014-12-01 00:00:00', To='2015-03-01 00:00:00', 
                                       From2='2015-12-01 00:00:00', To2='2016-01-01 00:00:00', return_data=True, recalculate=False)
        
        
    varvsalt_winter = altanal.VarVsAlt(vars= ['Ta C','Sm m/s','Ev hpa','Ua g/kg','Theta C', 'Pa H'], desvio_sta='C10', by= 'H',
                                       every='D',dates=hours, From='2015-06-01 00:00:00', To='2015-09-01 00:00:00', return_data=True, recalculate=False)
             
    varvsalt_spring = altanal.VarVsAlt(vars= ['Ta C','Sm m/s','Ev hpa','Ua g/kg','Theta C', 'Pa H'], desvio_sta='C10', by= 'H',
                                       every='D',dates=hours, From='2014-10-15 00:00:00', To='2015-12-01 00:00:00', 
                                       From2='2015-09-01 00:00:00', To2='2015-12-01 00:00:00',
                                       return_data=True, recalculate=False)
       
    altanal.plot(analysis = 'var_vs_alt', data=[varvsalt_winter, varvsalt_summer, varvsalt_spring], print_= True, 
                 delta='C10', hasconst=False, desvio_padrao=True, join= True, plot_mean_marker=True)
#     altanal.plot(analysis = 'var_vs_alt', data=[varvsalt_winter, varvsalt_summer, varvsalt_spring], print_= True, 
#                  delta='C10', polyfit=pol, desvio_padrao=True, join= True, plot_mean_marker=True)
#          
#     #===========================================================================
#     #Plot "trajectories" - quantiles - dispersion 
#     #===========================================================================

#     station_names =AttSta.stations(['Head']) 
# #     station_names =AttSta.stations(['Head', 'slope'])
# #     station_names =station_names + AttSta.stations(['Head', 'valley'])
# #     station_names.remove('C13') # humidity 12h
# #     station_names.remove('C11')
# #    
# #     station_names.remove('C10') # Temperature 12h
# #     station_names.remove('C11')
# #     station_names.remove('C08')
#   
# #     station_names.remove('C12') # Temperature 04h
# #     station_names.remove('C11')
#   
#   
#     Files =AttSta.getatt(station_names,'InPath')
#        
#     altanal = AltitudeAnalysis(Files, net='LCB')
#        
#     hours = [04]
#     
#     
#     varvsalt_summer = altanal.VarVsAlt(vars= ['Ta C','Sm m/s','Ev hpa','Ua g/kg','Theta C', 'Pa H'], desvio_sta='C10', by= 'H', quantile=[0.10,0.5,0.90]
#                                        ,dates=hours, From='2014-12-01 00:00:00', To='2015-03-01 00:00:00', 
#                                        From2='2015-12-01 00:00:00', To2='2016-01-01 00:00:00', return_data=True, recalculate=False)
#      
#      
#     varvsalt_winter = altanal.VarVsAlt(vars= ['Ta C','Sm m/s','Ev hpa','Ua g/kg','Theta C', 'Pa H'], desvio_sta='C10', by= 'H', quantile=[0.10,0.5,0.90]
#                                        ,dates=hours, From='2015-06-01 00:00:00', To='2015-09-01 00:00:00', return_data=True, recalculate=False)
#           
#  
#     
#     altanal.plot(analysis = 'var_vs_alt', data=[varvsalt_winter, varvsalt_summer], print_= True, 
#                  delta='C10', hasconst=False, desvio_padrao=True, join= True, plot_mean_marker=False, marker=True)
# #     altanal.plot(analysis = 'var_vs_alt', data=[varvsalt_winter, varvsalt_summer, varvsalt_spring], print_= True, 
#                  delta='C10', polyfit=pol, desvio_padrao=True, join= True, plot_mean_marker=True)
 














