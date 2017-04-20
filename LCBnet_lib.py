#===============================================================================
# For the math/tex font be the same than matplotlib
#===============================================================================

from __future__ import division
import math
import matplotlib.pyplot as plt
import operator
import os
import glob
import numpy as np
import pandas as pd
# import fnmatch
# import copy
# from scipy import interpolate
# from scipy import stats
# import datetime
from geopy.distance import vincenty
import re
from scipy.spatial import distance
from toolbox.geo import * 

#===============================================================================
# Analysis - Parameters 
#===============================================================================

class man(object):
    """
    DESCRIPTION
        contain methods to analysis and compute the data of a station or a network
    """
    def __init(self):
        pass

    def module(self,var):
        module={
              'Es hpa':self.__Es,
              'Ws g/kg':self.__Ws,
              'Ua g/kg':self.__q,
              'Theta C':self.__Theta,
              'Ev hpa':self.__Ev,
              'U m/s':self.__U,
              'V m/s':self.__V,
              'Ta C':self.__T,
              'Sm m/s':self.__Sm,
              'Pa H':self.__Pa,
              'Ua %':self.__Ua,
              "Rad w/m2":self.__radw
              }
        return module[var]

    def __T(self):
        """
        need for recalculating
        """
        return self.getvar('Ta C')

    def __Ua(self):
        """
        need for recalculating
        """
        return self.getvar('Ua %')
    
    def __Sm(self):
        """
        need for recalculating
        """
        return self.getvar('Sm m/s')
    
    def __Pa(self):
        """
        need for recalculating
        """
#         return self.getvar('Pa H')

    def __Es(self):
        """
        Return the vapor pressure at saturation from the Temperature
        
        Bolton formula derivated from the clausius clapeyron equation 
        Find in the American Meteorological Glaussary
        T is given in Degree celsus
        es is given in Kpa (but transform in hPa in this code)
        """
        es=0.6112*np.exp((17.67*self.getvar('Ta C'))/(self.getvar('Ta C')+243.5))*10 #hPa
        #self.__setvar('Es Kpa',es)
        return es

    def __Ws(self):
        """
        Return Mixing ratio at saturation calculated from
        the vapor pressure at saturation and the total Pressure
        """
        ws=self.getpara('E')*(self.getvar('Es hpa')/(self.getvar('Pa H')-self.getvar('Es hpa')))
        #self.__setvar('Ws g/kg',ws)
        return ws

    def __Ev(self):
        """
        Vapor pressure 
        """
        w= self.getvar('Ua g/kg')* 10**-3
        e=self.getpara('E')
        p=self.getvar('Pa H')

        Ev = (w/(w+e))*p
        return Ev

    def __q(self):
        """
        Compute the specfic humidity from the relative humidity, Pressure and Temperature
        """
        q=((self.getvar('Ua %')/100)*self.getvar('Ws g/kg'))/(1-(self.getvar('Ua %')/100)*self.getvar('Ws g/kg'))*1000
        #self.__setvar('Ua g/kg',q)
        return q

    def __Theta(self): 
        """
        Compute the Potential temperature
        """
        
        theta=(self.getvar('Ta C'))*(self.getpara('P0')/self.getvar('Pa H'))**(self.getpara('Cp')/self.getpara('R'))
        #self.__setvar('Theta C',theta)
        return theta

    def __U(self):
        """
        Return the wind in the X direction (U) in m/s
        """
        U,V = PolarToCartesian(self.getvar('Sm m/s'),self.getvar('Dm G'))
        return U

    def __V(self):
        """
        Return the wind in the Y direction (V) in m/s
        """
        U,V = PolarToCartesian(self.getvar('Sm m/s'), self.getvar('Dm G'))
        return V

    def __radw(self):
        """
        DESCRIPTION
            Input the radiation in W kj/m2 (from inmet stations)
            and return in w/m2
        """
        data = self.getvar('Rad kJ/m2')
        data = (data * 1000) / 3600.
        return data


    def getvarRindex(self,varname,var):
        Initime=self.getpara('From')
        Endtime=self.getpara('To')
        Initime=pd.to_datetime(Initime)# be sure that the dateset is a Timestamp
        Endtime=pd.to_datetime(Endtime)# be sure that the dateset is a Timestamp

        Data=var
        newdata=Data.groupby(Data.index).first()# Only methods wich work

        idx=pd.date_range(Initime,Endtime,freq='2min')
        newdata=newdata.reindex(index=idx)
        return newdata

#     def FromToBy(self,How):
#         
#         data=self.Data[self.getpara('From'):self.getpara('To')]
#         data=data.resample(self.getpara('By'),how=How)
#         return data

    def reindex(self,data, From, To):
        """
        Reindex with a 2min created index
        """
#         Initime=self.getpara('From')
#         Endtime=self.getpara('To')
#         by = self.getpara('by')# need it to clean data hourly
        Initime=pd.to_datetime(From)# be sure that the dateset is a Timestamp
        Endtime=pd.to_datetime(To)# be sure that the dateset is a Timestamp

        newdata=data.groupby(data.index).first()# Only methods wich work
        idx=pd.date_range(Initime,Endtime,freq='2min') # need it to clean data hourly
        newdata=newdata.reindex(index=idx)
        return newdata

    def getData(self,var= None, every=None,net='mean', From=None,To=None,From2=None, To2=None, by=None, how = "mean" , 
                group = None ,rainfilter = None , reindex =None, recalculate =False):
        """
        DESCRIPTION
            More sophisticate methods to get the LCBobject data than "getvar"
        INPUT
            If no arguments passed, the methods will look on the user specified parameters of the station
            If their is no parameters passed, it will take the parameters by default
            
            var: list of variable name
            
            net: how to group the data from the network
    
            group: 'D':Day, 'H':hour , 'M':minute, 'MH': minutes and hours
            
            resample:
                    B       business day frequency
                    C       custom business day frequency (experimental)
                    D       calendar day frequency
                    W       weekly frequency
                    M       month end frequency
                    BM      business month end frequency
                    CBM     custom business month end frequency
                    MS      month start frequency
                    BMS     business month start frequency
                    CBMS    custom business month start frequency
                    Q       quarter end frequency
                    BQ      business quarter endfrequency
                    QS      quarter start frequency
                    BQS     business quarter start frequency
                    A       year end frequency
                    BA      business year end frequency
                    AS      year start frequency
                    BAS     business year start frequency
                    BH      business hour frequency
                    H       hourly frequency
                    T       minutely frequency
                    S       secondly frequency
                    L       milliseonds
                    U       microseconds
                    N       nanoseconds
            how: how to group the data : mean or sum
            
        """

        #=======================================================================
        # Get the default parameters
        #=======================================================================
        if From == None:
#             pass
            From=self.getpara('From')
        else:
#             pass
            self.setpara('From', From) # This cause problem to get module data 
            # but at the same time if it is commented it give a problem in the getallvardata
            
        if To == None:
#             pass
            To=self.getpara('To')        
        else:
#             pass
            self.setpara('To', To)
        
        if To2==None:
            pass
        else:
            pass
#             self.setpara('To2',To2)
        
        if From2==None:
            pass
        else:
            pass
#             self.setpara('From2',From2)

#         print self.Data
#         #=======================================================================
#         # Get data from the network
#         #=======================================================================
        if isinstance(self, LCB_net):
            "Print I am a network"
            if net=='mean':
                print('I am averaging over all the stations')
                panel = self.getpanel()
                self.Data = panel.mean(0)
                

        #=======================================================================
        # Select the variables
        #=======================================================================
        if var == None:
            data = self.Data
        else:
            if not isinstance(var, list):
                var = [var]
            for v in var:
                if v not in self.Data.columns or recalculate:
                    try: 
                        data = self.Data
                        data[v] = self.getvar(v, recalculate=recalculate) # calculate the variable
                    except KeyError:
                        print "Return empty time serie"

                else:
                    data = self.Data

#         print data
        #=======================================================================
        # Reindex if needed
        #=======================================================================
        if reindex == True:
            if not From2:
                data = self.reindex(data, From,To)
            else:
                raise 
                print("Need to implement the from2 to 2 reindexing")
#                 data = self.reindex(data.append(data[From2:To2]), From,To)
        else:
            if not From2:
                data = data[From:To]
            else:
                data = data[From:To].append(data[From2:To2])

        #=======================================================================
        # Apply a filter for the rain
        #=======================================================================
        if rainfilter == True: # need to be implemented in a method
            data=data[data['Rc mm'].resample("3H",how='mean').reindex(index=data.index,method='ffill')< 3]
            if data.empty:
                raise ValueError(" The rainfilter removed all the data -> ")



        #=======================================================================
        # Resample
        #=======================================================================


        if by != None:
#             data=data.resample(by, how = how)
            if how=='sum':
                data=data.resample(by).sum()
#                 data=data.resample(by,how=lambda x: x.values.sum()) # This method keep the NAN
            if how=='mean':
                data=data.resample(by).mean() # This method keep the NAN

#         print data
        #===============================================================================
        # Select the period
        #===============================================================================

        data = data[From:To] # I should remove what is before
#         print data
#         print "80"*80
#         print data['Ta C']

        #===============================================================================
        # Group
        #===============================================================================
        
        if group:
            if how == "sum":
                print "9"*100
                print "sum"
                if group == 'M':
                    data=data.groupby(lambda t: (t.month)).sum()
        
                if group == 'D':
                    data=data.groupby(lambda t: (t.day)).sum()
        
                if group == 'H':
                    data=data.groupby(lambda t: (t.hour)).sum()
#                     print data
        
                if group == 'T':
                    data=data.groupby(lambda t: (t.minute)).sum()
                if group == "TH":
                    data=data.groupby(lambda t: (t.hour,t.minute)).sum()
        
            elif how == 'mean': # IT was set as "none"
                if group == 'M':
                    data=data.groupby(lambda t: (t.month)).mean()
        
                if group == 'D':
                    data=data.groupby(lambda t: (t.day)).mean()
        
                if group == 'H':
                    
                    data=data.groupby(lambda t: (t.hour)).mean()
        
                if group == 'T':
                    data=data.groupby(lambda t: (t.minute)).mean()
        
                if group == "TH":
                    data=data.groupby(lambda t: (t.hour,t.minute)).mean()

        #=======================================================================
        # Resample by every
        #=======================================================================
        if every == "M" and var != None:
            if group:
                raise AttributeError('Option "every" can not work with the function group')
            else:
                data=data.resample(by, how = how) # I make it again just in case
                data['index'] = data.index
                data['index'] = data['index'].map(lambda x: x.strftime('%Y-%m'))

                data['days'] = data.index.day
                data = pd.pivot_table(data, index = ['days'], columns=['index'], values=var)

        if every == "D" and var != None:
            if group:
                raise AttributeError('Option "every" can not work with the function group')
            else:
                data=data.resample(by, how = how) # I make it again just in case
                data['index'] = data.index
                data['index'] = data['index'].map(lambda x: x.strftime('%Y-%m-%d'))

                data['hours'] = data.index.hour
                data = pd.pivot_table(data, index = ['hours'], columns=['index'], values=var)


        #=======================================================================
        # Return the needed variable
        #=======================================================================
        if var == None:
            return data
        else:
            try:
                return data[var]
            except:
                print "Return empty vector"
                df_ = pd.DataFrame(index=data.index, columns=[var])
                df_.fillna(np.nan)
                return df_[var]
    
    def getvar(self,varname,From=None,To=None,rainfilter=None, recalculate=False):
        """
        DESCRIPTION
            Get a variables of the station
        INPUT:
            varname: string
            Recalculate: if True, recalculate the variable
        """
        
        if From == None:
            From = self.getpara('From')
        if To == None:
            To = self.getpara('To')
        
        
        if recalculate:
            try:
                print('Calculate the variable: '+ varname)
                var_module=self.module(varname)
                var_module=var_module()[From:To]
                var_module=self.getvarRindex(varname,var_module)# Reindex everytime to ensure the continuity of the data
                return var_module
            except KeyError:
                varindex=self.Data[From:To].index
                df_ = pd.DataFrame(index=varindex, columns=[varname])
                df_.fillna(np.nan)
                var=self.getvarRindex(varname,df_[varname])
#                 print self.Data.columns
                print('This variable cant be calculated->  '+ varname)
                return  var
        else:
            try:
                var=self.Data[varname][From:To]
                var=self.getvarRindex(varname,var)# Reindex everytime to ensure the continuity of the data
                return var
            except KeyError:
                try:
                    print('Calculate the variable: '+ varname)
                    var_module=self.module(varname)
                    var_module=var_module()[From:To]
                    var_module=self.getvarRindex(varname,var_module)# Reindex everytime to ensure the continuity of the data
                    return var_module
                except KeyError:
                    varindex=self.Data[From:To].index
                    df_ = pd.DataFrame(index=varindex, columns=[varname])
                    df_.fillna(np.nan)
                    var=self.getvarRindex(varname,df_[varname])
#                     print self.Data.columns
                    print('This variable cant be calculated')
                    return  var

        if rainfilter == True:
            return self.__rainfilter(var)
    
#===============================================================================
# Station
#===============================================================================

class AttVar(object):
    """
    DESCRIPTION
        contain the attribut of the variable of the Ribeirao Das Posses
    NOTE
        In the futur when I create this kind of class to access the attribut
        I should make them general.
        Because I already created this kind of class called "att_sta" but it wasn't too general
    """
    def __init__(self):
        self._attname = ['longname']
        self._att = {'Ta C':{'longname': 'Temperature (C)','longname_latex':'Temperature $(C^{\circ})$'},
                     'Theta C':{'longname':'Potential temperature (C)'},
                     'Sm m/s':{'longname':'wind speed (m/s)'},
                     'Dm G':{'longname':'Wind direction (degree)'},
                     'U m/s':{'longname':'Zonal wind  (m/s)'},
                     'V m/s':{'longname':'Meridional wind  (m/s)'},
                     'Bat mV':{'longname':'Battery (mV)'},
                     'Rc mm':{'longname':'Accumulated precipitation (mm)'},
                     'Ua %':{'longname':'Relative humidity (%)'},
                     'Ua g/kg':{'longname':'Specific humidity (g/kg)','longname_latex':'Specific humidity $(g.kg^{-1})$'},
                     'Pa H':{'longname':'Pressure (H)'},
                     'Rad w/m2':{'longname':'Solar radiation (W/m2)'},
                     }

    def showatt(self):
        print(self._attname)
    
    def getatt(self,object, attnames):
        """
        INPUT
            staname : list. name of the stations 
            att : scalar. name of the attribut
        Return 
            return attributs in a list
        TODO
            just return a dictionnary
        """

        att = [ ]
        if type(attnames) != list:
            try:
                if type(attnames) == str:
                    attnames = [attnames]
            except:
                raise
        
        for attname in attnames:
            try:
                att.append(self._att[object][attname])
            except KeyError:
                
                raise KeyError('The parameter' + attname + ' do not exist')
        return att

class att_sta(object):
    """
    DESCRIPTION
        -> Class containing metadata informations about the climatic network in Extrema
        -> Contain methods to manipulate the network metadata
    params:
        Path_att: path of thecsv file with all the metadata
    """
    def __init__(self, Path_att="/home/thomas/phd/obs/staClim/metadata/metadata_allnet_select.csv"):

        print "**"*10
        print 'METADATA FILE USED:'
        print Path_att

        print "**"*10
        
        
        self.attributes =pd.read_csv(Path_att, index_col=0)

    def addatt(self, df = None, path_df=None):
        """
        DESCRIPTION
            add a dataframe of station attribut to the already existing 
        """
        attributes = self.attributes
        if path_df:
            df = pd.read_csv(path_df, index_col=0)
        

        newdf = pd.concat([attributes, df], join='outer', axis=1)
        self.attributes = newdf
      
    def sortsta(self,sta,latlon):
        """
        DESCRIPTION:
            Return a sorted list of the Latitude or longitude of the given station names
        INPUT: sta : name of the station
               latlon:  'Lat' or 'Lon'
        RETURN: {stations_name,positions} sorted dictionnary
        pos={}
        """
        metadata=[]
        staname=[]
        pos={}
        for i in sta:
            pos[i]=self.getatt(i, latlon)
        sorted_pos = sorted(pos.items(), key=operator.itemgetter(1))
        for i in sorted_pos:
            metadata.append(i[1])
            staname.append(i[0])
        return {'stanames':staname,'metadata':metadata}

    def stations(self,values=None, all=None, params=None):
        """
        Return the name of the stations corresponding to a particular parameter
        all: True, return all stations in parameters
        Params: Param_min, Param_max
                return the stations in between the param_min and params_max
        
        """
         
        if all:
            return self.attributes.index
        
        if type(values) is not list:
            raise TypeError('Must be a list')
    
        sta = [ ]
        if values: 
            for staname in self.attributes.index:
                l = []
                for k in values:
                    if k in self.attributes.loc[staname].values:
                        l.append(True)

                if len(l) == len(values): # I really dont know why using all does not work
                    sta.append(staname)

        if params:
            for param in params.keys():
#                 if sta !=[]:
                attribut = self.attributes.loc[sta,:]
#                 else:
#                     attribut = self.attributes
                if param in self.attributes.columns.values:

                    sta = attribut[(attribut[param]> params[param][0]) & (attribut[param]< params[param][1])]
                    sta = sta.index.values.tolist()      
        return sta

    def setatt(self,staname, newatt, newvalue):
        """
        DESCRIPTION
            insert a new value of a new attribut corresponding at the specific station
        """
        try:
            self.attributes.loc[staname, newatt] = newvalue
        except KeyError:
            pass
#             print ("station %s does not have attribut in the metadata table selected"% (staname))
    def showatt(self):
        print(self.attributes)

    def getatt(self,stanames,att):
        """
        INPUT
            staname : list. name of the stations 
            att : scalar. name of the attribut
            all: return the entire dataframe
        Return 
            return attributs in a list
        TODO
            just return a dictionnary
        """


        staatt = [ ]
        if type(stanames) != list:
            try:
                if type(stanames) == str:
                    stanames = [stanames]
            except:
                raise
#         print self.attributes['st68']
        
        for staname in stanames:
#             try:
            staatt.append(self.attributes.loc[staname,att])
#             except KeyError:
#                 print 'The parameter ' + att + ' do not exist for ' + staname
#                 raise

        # drop nan attribute
        return staatt

    def setInPaths(self,dirInPath):
        """
        INPUT
            String with the directory path of the station files
        OUTPUT
            
        EXAMPLE
            InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
            AttSta = att_sta()
            AttSta.setInPaths(InPath)
        """

        Files=glob.glob(dirInPath+"*")

        for f in Files:
#             print os.path.basename(f)
            staname = os.path.basename(f)[0:-4]
            InPath = f
            self.setatt(staname, 'InPath', InPath)

    def dist_sta(self,couples, formcouples = None):
        """
            DESCRIPTION
                Calculate the distance between two points
            INPUT
                list of couple of station names 
                Couple = TRUE => will form the list of couple from 
            OUTPUT
                return a list of the distance in kilometer (
            EXAMPLE
                couple = [['C15','C04']]
                dist_sta(couple)
                
                stationnames = ['C15','C04','C05']
                dist_sta()stationnames, formcouple=True)
        """

        def rearange(sortedlist):
            """
            DESCRIPTION
                Rearange the stations name list to form couple
                 which are the input of the att_sta.dist_sta() 
            """
            couple = [ ]
            for ini,end in zip(sortedlist[:-1],sortedlist[1:]):
                couple.append([ini,end])
            return couple
        
        if formcouples == True:
            if isinstance(couples,list):
                couples = rearange(couples)
            else:
                raise TypeError('Must be a list !')

        # make sure we have a list
        if isinstance(couples,list):
            pass
        else:
            couples = [couples]
    
        # make sure we have a list of list
        if all(isinstance(i, list) for i in couples):
            pass
        else:
            couples = [couples]
    
        dist = [ ]
        for sn in couples:
            print sn
            lat1 = self.getatt(sn[0],'Lat')[0]
            lon1 = self.getatt(sn[0],'Lon')[0]
            pos1 = (lat1,lon1)
            
            lat2 = self.getatt(sn[1],'Lat')[0]
            lon2 = self.getatt(sn[1],'Lon')[0]
            pos2 = (lat2,lon2)
            
            dist.append(vincenty(pos1,pos2).km)
    
        return dist

    def dist_matrix(self, stanames):
        """
        DESCRIPTION
            Return, dataframe with the distance matrix of the given stations
        IMPORTANT NOTE:
            This function return the euclidian distance based on latitude longitude
            It introduce error, locations should be converted in meters before
            but for the local scale I guess it is not a problem
        """
        lats = self.getatt(stanames, 'Lat')
        lons = self.getatt(stanames, 'Lon')
        coords = []
        coords  = [(lat, lon) for lat,lon in zip(lats, lons)]
        dist = distance.cdist(coords, coords, 'euclidean')
        df_dist = pd.DataFrame(dist, index=stanames,columns=stanames)
        
        return df_dist
        
    def to_csv(self, outpath, params_out=None, stanames=None):
        """
        Save the dictionnary of parameter as a dataframe
        """
        df = pd.DataFrame(self.attributes)
        df = df.T
        if params_out:
            df = df[params_out]
        if stanames:
            df = df.loc[stanames,:]
        
#         print df
        df.to_csv(outpath)

class Plots():
    """
    Class container
    Contain all the plot which can be applied to a station
    """

    def __init__(self):
        pass

    def TimePlot(self,var='Ta C', by = None, group = None, subplots = None, From=None, To=None, outpath='/home/thomas/'):

            if not isinstance(var, list):
                var = [var]
            for v in var:
                data=self.getData(var=v,From=From,To=To,by=by, group = group)

                data = data.dropna()
                try:
                    data.plot(subplots = subplots, marker='o')
                except TypeError:
                    print "No numeric data"
                    
                objectname = self.getpara('stanames')
                print objectname
                if isinstance(objectname, list):
                    objectname = "network" # should be implemented somewhere else

                if outpath:
                    plt.savefig(outpath+objectname+"_TimePlot.png")
                    print('Saved at -> ' +outpath)
                    plt.close()
                else:
                    plt.show()

    def dailyplot(self,var = None,how=None, From=None, To=None,From2=None, To2=None, group= None, save= False, outpath = "/home/thomas/", labels = None):
        """
        Make a daily plot of the variable indicated
        """
        lcbplot = LCBplot() # get the plot object
#         argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
#         arglabel = lcbplot.getarg('label')
#         argticks = lcbplot.getarg('ticks')
#         argfig = lcbplot.getarg('figure')
#         arglegend = lcbplot.getarg('legend')

        for v in var:
            fig = plt.figure()
            color = iter(["r", "b",'g','y'])
            for from_ , to_, from2_, to2_, label in zip(From, To, From2, To2, labels):
                c = color.next()
                if how == None:
                    data = self.getData(var = v, From = from_, To=to_, From2=from2_, To2=to2_)
                    quartile1 = data.groupby(lambda x: x.hour).quantile(q=0.10)
                    quartile3 = data.groupby(lambda x: x.hour).quantile(q=0.90)
                    mean = data.groupby(lambda x: x.hour).mean()
                if how =='sum':
                    mean = self.getData(var = v,group=group, how=how, From = from_, To=to_, From2=from2_, To2=to2_)
                
                if how ==None:
                    print "-->" + str(quartile1.columns)
     
                    plt.fill_between(quartile1[v].index.values, quartile1[v].values, quartile3[v].values, alpha=0.1,color=c)

                    plt.plot([], [], color=c, alpha=0.1,linewidth=8, label=(label+' q=0.90  0.10'))
      
    
                plt.plot(mean[v].index.values, mean[v].values,linewidth = 8, linestyle='-', color=c, alpha=0.7, label=(label+' mean'))

            plt.xlim((0,24))
            
            if v=='Ta C':
                plt.ylabel('Temperature', fontsize=30)
            elif v=='Ua g/kg':
                plt.ylabel('Specific humidity ',fontsize=30)
            elif v=='Rc mm':
                plt.ylabel('Accumulated Precipitation',fontsize=30)
            elif var=='Ev hpa':
                plt.ylabel('Vapor Pressure',fontsize=30)
            else:
                plt.ylabel(v,fontsize=30)
                
            plt.xlabel( "Hours",fontsize=30)
            plt.grid(True, color="0.5")
            plt.tick_params(axis='both', which='major',labelsize=30, width=2,length=7)
            plt.legend()
    
            if not save:
                plt.show()
            else:
                plt.savefig(outpath+v[0:2]+"_dailyPlot.svg", transparent=True)

class LCB_station(man, Plots):
    """
    DESCRIPTION
        Contain the data of ONE WXT! and the different module to transform the data in some specific way.
    PROBLEM
        SHould calculate the specific variables only when asked
    """

    def __init__(self,InPath, net='LCB', clean=True):
        """
        PROBLEM
            on raw file the option "Header=True is needed"
        SOLUTION
            self.rawData=pd.read_csv(InPath,sep=',',index_col=0,parse_dates=True)
        """
        self.para={
        }
        print InPath
        self.Data = self.__read(InPath, net=net, clean=clean)
        

        self.paradef={
                    'Cp':1004, # Specific heat at constant pressure J/(kg.K)
                    'P0':1000,#Standard pressure (Hpa)
                    'R':287,#constant ....
                    'Kelvin':272.15, # constant to transform degree in kelvin
                    'E':0.622,
#                     'By':'2min',
                    'To':self.Data.index[-1],
                    'From':self.Data.index[0],
#                     'group':'1H',
        }
        self.__poschar()

    def __read(self,InPath,net, clean=True):
        """
        read the different type of network
        and load the different parameters
        params:
            net: the network than you wnat to read
            clean if the data are clean or not
        """
        self.setpara("InPath",InPath)
        self.setpara("dirname", os.path.dirname(InPath))
        
        self.setpara("filename", os.path.basename(InPath))

        
        if net =='LCB':
            df = pd.read_csv(InPath,sep=',',index_col=0,parse_dates=True)
            self.setpara("stanames", os.path.basename(InPath)[0:3])
            self.setpara('By','2min')
            
        if net == "Sinda":
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            else:
                df = pd.read_csv(InPath, error_bad_lines=False,skiprows=1, index_col=0, parse_dates=True)
                
                mapping = {
                            'Pluvio (mm)':'Rc mm',
                            'TempAr (oC)': 'Ta C',
                            'TempMax (oC)': 'Tamin C',
                            'TempMin (oC)':'Tamax C',
                            'UmidRel (%)': 'Ua %',
                            'VelVento10m (m/s)': 'Sm m/s',
                            'DirVento (oNV)': 'Dm G'
                            }
                
                cols = []
                for f in df.columns:
                    col = re.sub(r'[^\x00-\x7F]+',' ', f)
                    if col in mapping.keys():
                        col = mapping[col]
                    cols.append(col)
                df.columns = cols
            df.index = df.index - pd.Timedelta(hours=3) # UTC
#             df.index = df.index - pd.Timedelta(hours=3) # become it make the mean the hours after
            self.setpara("stanames", os.path.basename(InPath)[-9:-4])
            self.setpara('by','3H')


        if net =='INMET':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            else:
                df = pd.read_csv(InPath,parse_dates=[[1,2,3,4]],index_col=0, keep_date_col=False, delim_whitespace=True, header=None, error_bad_lines=False)
                df.columns = ['ID', "Ta C", 'Tamax C', 'Tamin C', 'Ua %','Uamax %','Uamin %', 'Td C',
                               'Td max', 'Td min', 'Pa H', 'Pamax H', 'Pamin H', 'Sm m/s', 'Dm G',
                               'Smgust m/s', 'Rad kJ/m2', 'Rc mm' ]
                del df['ID']
            df.index = df.index - pd.Timedelta(hours=3) # UTC Once it is clean it is 
            self.setpara("stanames", os.path.basename(InPath)[-8:-4])
            self.setpara('by','1H')

        if net == 'IAC':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)

            else:
                def parse(yr, yearday, hrmn):
                    # transform 24 into 00 of the next day
                    if hrmn[0:2] == "24":
                        yearday = str(int(yearday)+1)# this might give a problem at the end of the year
                        hrmn ="00"+ hrmn[2:]
    
                    if hrmn == '100':
                        hrmn = '0100'
                    
                    if hrmn =='200':
                        hrmn = '0200'
                    date_string = ' '.join([yr, yearday, hrmn])
    #                 print(date_string)
    #                 print pd.datetime.strptime(date_string,"%Y %j %H%M")
                    return pd.datetime.strptime(date_string,"%Y %j %H%M")
    
                df = pd.read_csv(InPath, parse_dates={'datetime':[1,2,3]},date_parser=parse, index_col='datetime', header=None, error_bad_lines=False)

                if len(df.columns) == 8:
                    print "type2"
                    df.columns = ['type', 'Sm m/s', "Dm G", 'Rad2 W/m2', 'Ta C', 'Ua %','???', 'Rc mm']
                else:
                    df.columns = ['type', 'Sm m/s', 'Sm10m m/s', "Dm G", 'Rad W/m2', 'Rad2 W/m2', 'FG W/m2', 'Ua %',
                               'Ta C', 'Tasoil1 C', 'Tasoil2 C', 'Tasoil3 C', 'Pa H', 'SH %', 'Rc mm']
                
                # replace missing value with Nan
                df.replace(-6999, np.NAN, inplace=True)
                df.replace(6999, np.NAN, inplace=True)
                df.replace('null', np.NAN, inplace=True)
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1H')

        if net == 'svg':
            df = pd.read_csv(InPath, index_col=0, parse_dates=True )
#             df.index = df.index + pd.Timedelta(hours=2) # UTC Once it is clean it is 
            df.columns = ['Ta C', "Ua %", 'Pa kpa', 'Sm m/s', 'Dm G', 'Rc mm', 'Rad Wm/2']
            self.setpara("stanames", 'svg')
            self.setpara('by','30min')
        
        if net == 'peg':
            df = pd.read_csv(InPath, index_col=1, header=None, parse_dates=True)
#             df.index = df.index + pd.Timedelta(hours=2) # UTC Once it is clean it is 
            df.columns = ['i','Ta C']
            del df['i']
            self.setpara("stanames", 'peg')
            self.setpara('by','30min')
        
        return df
            
    def __poschar(self):
        AttSta = att_sta()
        attributes = AttSta.attributes
        try:
            for att in attributes:
                self.setpara(att,AttSta.getatt(self.getpara('stanames'), att))
        except KeyError:
            print('This stations doesnt have characteristics')

    def __stationname(self):
        InPath=self.getpara("InPath")

    def showvar(self):
        for i in self.Data.columns:
            print i


    def showpara(self):
        print self.para

    def __rainfilter(self,var):
        """
        Return the index of the series whithout daily rain
        Valleynet.Data['Rc mm'].resample("D",how='mean').reindex(index=Valleynet.Data.index,method='ffill')
        """
        Rain=self.getvar(self,'Rc mm').resample("D",how='mean')
        return var[Rain<0.1]

    def __setvar(self,varname,data):
        try:
            self.Data[varname]=data
        except:
            print('Couldnt add variable in the data'+ varname)

    def getpara(self,paraname):
        try:
            val=self.para[paraname]
            return val
        except KeyError:
#             print('Parameter by default used '+ paraname)
            try:
                val=self.paradef[paraname]
                return val
            
            except KeyError:
                print('this parameter dosenot exist '+ paraname)
                raise

    def setpara(self,name,value,key = None, append = None):
        """
        DESCRIPTION
            Set the parameter of the LCB object
        INPUT
            name : name of the parameter
            value: value of the parameter
            keys: if Keys != None then it will be the key of a dictionnary
            append: if == None then the newvalue overwrite the old one
        """

        if name == 'To' or name == 'From': # convert in datetime format
            value=pd.to_datetime(value)


        if append == None:
            if key == None:
                self.para[name]=value
            else:
                self.para[name] = {key:value}
        else:
            try:
                if key == None:
                    oldpara = self.para[name]
                    oldpara.append(value)
                else:
                    oldpara = self.para[name]
                    oldpara[key] = value
                self.para[name] = oldpara
            except KeyError:
                # To initialise the parameter
                if key == None:
                    self.para[name]=value
                else:
                    self.para[name] = {key:value}

    def __deregister(self, net):
        self.my_net = None

    def __register(self, net):
        self.my_net = net
        print(self.my_net)

    def report(self):
        print "Myself is:", self.getpara['InPath'], self
        if self.my_net:
            print "My net: ",self.my_net
        else:
            print "i dont belong to any net"

class LCB_net(LCB_station):
    def __init__(self):
        self.min = None
        self.max = None
        self.mean = None
        self.Data=None # Which is actually the mean 
        self.para={
        }
        self.paradef={
                    'Cp':1004,# could be set somwhere else # Specific heat at constant pressure J/(kg.K)
                    'P0':1000,#Standard pressure (Hpa)
                    'R':287,#constant ....
                    'Kelvin':272.15, # constant to transform degree in kelvin
                    'E':0.622,
                    'By':'H',
                    'To':None,
                    'From':None
        }
        
        self.setpara('stanames',[])
        self.setpara('guys', {}) # ask Marcelo is it good to put the LCB object in the parameters?

    def getvarallsta(self, var = None, stanames = None, by = None, how = 'mean', From=None,To=None, From2=None, To2=None, group=None):
        """
        DESCRIPTION
            return a dataframe with the selected variable from all the stations
        TODO 
            UTLISER arg* et kwarg pour passer les argument a 
            getData sans avoir besoin de tous les recrires
        """
        if not stanames:
            stanames = self.getpara('stanames')
        
        df = pd.DataFrame()
        for staname in stanames:
            station = self.getsta(staname)[0]
            s = station.getData(var = var, by = by, From=From, To=To, From2=None, To2=None, reindex=True, group=group, how=how)
            s.columns = [staname]
            
            df = pd.concat([df,s], axis=1)

        return df

    def getpanel(self, stanames=None, var = None):
        """
        Get a panel constituted by the dataframe of all the stations constituing the network
        """
        if not stanames:
            stanames = self.getpara('stanames')
        
        dict_panel = {}
        for staname in stanames:
            station = self.getsta(staname)[0]
            data = station.getData(reindex=True)
            dict_panel[staname] = data
        panel = pd.Panel(dict_panel)
        return panel
        
    def getsta(self,staname, all=None, sorted=None, filter=None):
        """
        Description
            Input the name of the station and give you the station object of the network
            staname : List
            if "all==True" then return a dictionnary containing 
            all the stations names and their respective object
            
            if sorted != None. Le paramtere fourni sera utilise pour ordonner en ordre croissant la liste de stations 
            dans le reseau
            In this case it return a dictionnary with the key "staname" with the stanames sorted
            and a key stations with the coresponding stations sorted
            
            if filter != None. list of parameter Seulement les stations contenant le parametre donner seront selectionner.
        Example
            net.getsta('C04')
            net.getsta('',all=True)
            net.getsta('',all=True,sorted='Lon')
            net.getsta('',all=True,sorted='Lon', filter =['West'])
        """
        
        
        
        if type(staname) != list:
            try:
                if type(staname) == str:
                    staname = [staname]
            except:
                raise

        if all==True:
            try:
                staname = self.getpara('stanames')
                sta = self.getpara('guys')
            except:
                print('COULDNT GIVE THE NAME OF ALL THE STATIONS')

        else:
            try:
                sta = [ ]
                for s in staname:
                    sta.append( self.getpara('guys')[s])
            except KeyError:
                print(s)
                raise KeyError('This stations is not in the network')

        if filter != None:
            staname=att_sta().stations(filter)
            sta = [ ]
            for i in staname:
                sta.append(self.getpara('guys')[i])

        if sorted != None:
            sortednames=att_sta().sortsta(staname,sorted)
            sortednames=sortednames['stanames']
            print sortednames
            sta['stanames'] =  sortednames
            s = []
            for i in sortednames:
                s.append(self.getpara('guys')[i])
            sta['stations'] = s
            print(sta)
        return sta 

    def report(self):
        guys = self.getpara('guys')
        stanames = self.getpara('stanames')
        print "Their is %d stations in the network. which are:" % len(guys)
        print stanames

    def dropstanan(self,var = "Ta C",by='H', perc=0, From=None,To=None):
        """
        Drop a station wich has more than <perc> percent of Nan value 
        in the the network period 
        """
        
        df = self.getvarallsta(var=var, by=by, From=From,To=To)
        sumnan = df.isnull().sum(axis=0)

        nans = (sumnan/len(df)) *100
        for sta in nans.index:
            if nans[sta] > perc:
                print "drop station "+ sta
                self.remove(self.getsta(sta)[0])
            
    def remove(self, item):
        """
        DESCRIPTION
            Remove a station from the network
        
        """
        item._LCB_station__deregister(self)# _Com_sta - why  ? ask Marcelo
        guys = self.getpara('guys')
        stanames = self.getpara('stanames')
        del guys[item.getpara('stanames')]
        if item.getpara('stanames') in stanames: stanames.remove(item.getpara('stanames'))
        self.setpara('guys',guys)
        self.setpara('stanames', stanames)
        self.min = None
        self.max = None
        self.Data = None
#         for guy in self.getpara('guys'):
#             self.__update(guy)

    def AddFilesSta(self,files, net='LCB', clean=True):
        """
        Add stations to the network
        input: list of data path
        
        """
        print('Adding ')
        
        if files ==[]:
            raise "The list of files is empty"

        for file in files:

            try:
                sta=LCB_station(file, net=net, clean=clean)
                self.add(sta)
            except AttributeError:
                print file
                print "Could not add station to the network"
        print "#"*80
        print "Network created wit sucess!"
        print "#"*80
        
    def add(self, station):
        """
            DESCRIPTION
                Add an LCB_object
        """
        self.__add_item(station)

    def validfraction(self, plot = None):
        """
        DESCRIPTION
            Calculate the percentage of the available data in the network

        OUTPUT
            dataframe
            column "fraction" is the porcentage of available data in the network
            the other columns reprensent the availability of each stations

        EXAMPLE
            dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
            AttSta = att_sta()
            AttSta.setInPaths(dirInPath)
            Files =AttSta.getatt(AttSta.stations([]),'InPath')
        
            net=LCB_net()
            net.AddFilesSta(Files)
            
            df= net.validfraction()
            df['fraction'].resample('M').plot(kind='bar')
            plt.show()
        """
        nbsta = len(self.getpara('guys'))
        
        stations = self.getpara('guys')
        
        df = pd.DataFrame()
        for sta in stations.keys():
            data = stations[sta].Data
            data = data.sum(axis=1)
            data = data.dropna(axis=0)
            index = data.index
            s = pd.Series([1]*len(index), index = index)
            ndf = pd.DataFrame(s,columns =[sta])
            df = df.join(ndf, how = 'outer')
    
        df['fraction'] = (df.sum(axis=1)/nbsta)*100
        if plot == True:
            df['fraction'].plot()
            plt.show()
        return df
 
    def write_clean(self, outpath):
        """
        Apply threshold to the dataframe and write them 
        """ 
        threshold={
                        'Pa H':{'Min':800,'Max':1050},
                        'Ta C':{'Min':-5,'Max':40,'gradient_2min':4},#
                        'Ua %':{'Min':0.0001,'Max':100,'gradient_2min':15},
                        'Rc mm':{'Min':0,'Max':80},
                        'Sm m/s':{'Min':0,'Max':30},
                        'Dm G':{'Min':0,'Max':360},
                        'Bat mV':{'Min':0.0001,'Max':10000},
                        'Vs V':{'Min':9,'Max':9.40}#'Vs V':{'Min':8.5,'Max':9.5}
                    }

        stanames = self.getpara('stanames')
 
        
        for staname in stanames:

            station = self.getsta(staname)[0]
            filename = station.getpara('filename')
            
            newdata = station.getData(reindex=False)
#             data = station.getData(reindex=True) # old version
#             newdata = data.copy() # old version
            for var in threshold.keys():
                try:
                    index = newdata[(newdata[var]<threshold[var]['Min']) | (newdata[var]>threshold[var]['Max']) ].index
                    newdata[var][index] = np.NAN
                    newdata.to_csv(outpath+filename)
                except KeyError:
                    print "No var: -> " + var

    def __object_name(self):
        """
        DESCRIPTION
            Create a default name for the LCB object
        """
        try:
            nbobject = len(self.getpara('stanames'))
        except:
            nbobject = 0
        return "LCB_object"+str(nbobject+1)

    def __add_item(self,item):
        
        
        try:
            staname=item.getpara('stanames')
        except KeyError:
            print("Give a default name to the LCB_object")
            staname = self.__object_name()

        print('========================================================================================')
        print('Adding To the network/group -> '+ str(staname))
        self.setpara('stanames',staname, append =True)
        self.setpara('guys',item, key= staname, append = True)
        self.getpara('guys')

        item._LCB_station__register(self)

#     def __update(self, item):
#         """
#         DESCIPTION 
#             Calculate the mean, Min and Max of the network
#         PROBLEM
#             If their is to much data the update cause memory problems
#         """
# 
#         try:
#             print('Updating network')
#             Data=item.reindex(item.Data)
#             self.max=pd.concat((self.max,Data))
#             self.max=self.max.groupby(self.max.index).max()
#             #------------------------------------------------------------------------------ 
#             self.min=pd.concat((self.min,Data))
#             self.min=self.min.groupby(self.min.index).min()
#             #------------------------------------------------------------------------------
#             net_data=[self.Data]*(len(self.getpara('guys'))-1)# Cette technique de concatenation utilise beaucoup de mermoire
#             net_data.append(Data)
#             merged_data=pd.concat(net_data)
#             self.Data=merged_data.groupby(merged_data.index).mean()
#             self.setpara('From',self.Data.index[0])
#             self.setpara('To',self.Data.index[-1])
# 
#         except TypeError:
#             print('Initiating data network')
#             self.Data=Data
#             self.min=Data
#             self.max=Data

class LCBplot():
    def __init__(self,):
        """
        DESCRIPTION
            this class contains the methods to manage plot object
            It contains also the basin parameters that I need to make my beautiful plots
        NOTE
            This class need to be very basic to be use in all the plot of the package beeing
            developped
        
        BEFORE MODIFICATION
            LCB_object: can be either a station or a network
        This class could made to be used both on the ARPS and LCBnet program
        """

        self.argdef= {
            'OutPath':'/home/thomas/',
            'screen_width':1920,
            'screen_height':1080,
            'dpi':96,
            }
        

        self.arg = {
                 'legend':{
                           "prop":{"size":20}, # legend size
                           "loc":"best"
                           },
                 'figure':{
                        },
                 'plot':{
                         'markersize':10,
                         'alpha':1
                         },
                 'label':{
                          "fontsize":30
                         },
                'ticks':{
                         'labelsize':30,
                         'width':2, 
                         'length':7
                         
                         }
                     
                }
        self.__figwidth()

    def __figwidth(self):
        width=self.getarg('screen_width')
        height=self.getarg('screen_height')
        DPI=self.getarg('dpi')
        wfig=width/DPI #size in inches 
        hfig=height/DPI
        self.setarg('wfig',wfig)
        self.setarg('hfig',hfig)
        self.arg['figure']['figsize'] = (wfig, hfig)
        self.arg['figure']['dpi'] = DPI
        

#     def __subtitle(self,LCB_station):
#         """
#         Write the subtitle of the plot
#         """
#         sub=LCB_station.getarg('From')
#         self.setarg('subtitle', sub)

    def setarg(self,argmeter,value):
        self.arg[argmeter]=value
        print(str(argmeter)+' has been set to -> '+ str(value))

    def getarg(self,argmeter):
        try:
            return self.arg[argmeter]
        except KeyError:
            print(argmeter + ' has been not set -> Default value used ['+str(self.argdef[argmeter])+']')
            try:
                return self.argdef[argmeter]
            except KeyError:
                print(argmeter+ ' dont exist')

    def delarg(self,varname):
        try:
            del self.arg[varname]
            print('Deleted argmeter-> ',varname)
        except KeyError:
            print('This argmeter dont exist')

    def __setargdef(self,argmeter,value):
        self.argdef[argmeter]=value
        print(str(argmeter)+' has been set by [default] to -> '+ str(value))

    def __getargdef(self,argmeter):
        try:
            return self.argdef[argmeter]
        except KeyError:
                print(argmeter+ ' by [default] dont exist')

    def __levels(self,varname):
        self.argdef['nlevel']=10# number of discrete variabel level
        self.argdef['varmax']=int(self.arps.get(varname).data.max())
        self.argdef['varmin']=int(self.arps.get(varname).data.min())
        varmax=self.getarg('varmax')
        varmin=self.getarg('varmin')
        nlevel=self.getarg('nlevel')
        levels=np.linspace(varmin,varmax,nlevel)
        return levels


 
# 
# 
# 
# 
# 
# class Vector(object):
#     def __init__(self,data,Theta,Norm):
#         print('Plotting Vector')
#         self.data=data
#         self.Theta=Theta
#         self.Norm=Norm
#         self.type='default'
#         self.arg=dict()
#         self.__Types()#Create an initial library of differents option defining different type of grafics
#         self.__Arg()# Final library of option
#     def __Types(self):
#         self.Types={
#         'default': {
#             'colorvar':['k','b'],
#             'Poswind':0,
#             'colorfig':'k',
#             'colorwind':'k',
#             'vectorlength':40,
#             'Poswindscale':[5,3],
#             'ScaleLength':5,
#             'fontsize':40,
#             'twin':False,# to plot on two axis
#             'linewidth':4,
#             'y_lim':[-4,4]},
#         'AnomalieT': {
#             'Poswind':-3.7,
#             'colorvar':['k','b'],
#             'colorfig':'b',
#             'colorwind':'k',
#             'vectorlength':40,
#             'Poswindscale':[5,3],
#             'ScaleLength':5,
#             'fontsize':40,
#             'y_lim':[-4,4],
#             'linewidth':4},
#         'AnomalieH':{
#             'Poswind':-1.3,
#             'colorvar':['k','b'],
#             'colorfig':'k',
#             'colorwind':'k',
#             'vectorlength':40,
#             'Poswindscale':[5,1.2],
#             'ScaleLength':5,
#             'fontsize':40,
#             'linewidth':4,
#             'y_lim':[-2,2]},
#         'AbsolueT':{
#             'Poswind':14,
#             'colorvar':['k','b'],
#             'colorfig':'b',
#             'colorwind':'k',
#             'vectorlength':40,
#             'Poswindscale':[5,25],
#             'ScaleLength':5,
#             'fontsize':40,
#             'linewidth':4,
#             'y_lim':[10,30]},
#         'AbsolueH':{
#             'Poswind':7.5,
#             'colorvar':['k','b'],
#             'colorfig':'k',
#             'colorwind':'k',
#             'vectorlength':40,
#             'Poswindscale':[5,13],
#             'ScaleLength':5,
#             'fontsize':40,
#             'linewidth':4,
#             'y_lim':[7,14]},
#             }
#     def __Extras(self):
#         if self.type=='AnomalieH' or self.type=='AnomalieT':
#             self.ax.plot(self.Theta.index,[0]*len(self.Theta.index),color=self.arg['colorfig'],linestyle='-', linewidth=self.arg['linewidth']-2)
#         if self.arg['twin']==True:
#             self.ax2 = self.ax.twinx()
#             self.ax2.plot([x[0]+x[1]/60 for x in self.data[1].index],self.data[1],color=self.argTwin['colorvar'][1],linestyle='-', linewidth=self.argTwin['linewidth'])
#             self.Properties_twin(self.ax2)
#     def __Arg(self):
#         self.arg=dict(self.arg.items()+self.Types[self.type].items())
#     def SetOption(self,option,var):
#         "Change the value of a default option. type-> library. option-> graphical option, var-> new value to collocate"
#         self.arg[option]=var
#     def SetType(self,Type):
#         "Change the defaut option with another library of option"
#         self.type=Type
#         self.__Arg()
#     def SetTypeTwin(self,Type):
#         self.argTwin=self.Types[Type]
#     def report(self):
#         print('current option choosen: '+str(self.type))
#         print('The parameters of this option are: '+str(self.arg))
#         print('To change the Type please use .SetType(''''option'''')')
#         print('To change an option in the current Type please use .SetOption')
#     def plot(self):
#         print("Plot")
#         self.Main()
#         self.__Extras()
#         self.Properties(self.ax)
#     def Main(self):
#         print("Main")
#         V=np.cos(map(math.radians,self.Theta+180))*self.Norm
#         U=np.sin(map(math.radians,self.Theta+180))*self.Norm
#         X=self.Theta.index
#         Y=[self.arg['Poswind']]*len(X)
#         Fig=plt.figure()
#         ax=plt.gca()
#         q=ax.quiver(X,Y,U,V,scale=self.arg['vectorlength'],color=self.arg['colorwind'])
#         p = plt.quiverkey(q,self.arg['Poswindscale'][0],self.arg['Poswindscale'][1],self.arg['ScaleLength'],str(self.arg['ScaleLength'])+" m/s",coordinates='data',color=self.arg['colorwind'],labelcolor=self.arg['colorwind'],fontproperties={'size': 30})
#         for idx,p in enumerate(self.data):
#             if self.arg['twin']==False or idx<1:
#                 print("is printing :  "+str(idx) )
#                 ax.plot([x[0]+x[1]/60 for x in p.index],p,color=self.arg['colorvar'][idx],linestyle='-', linewidth=self.arg['linewidth'])
#                 print("finish :  "+str(idx) )
#         self.ax=ax
#     def Properties(self,ax):
#         print("Properties")
#         ax.grid(True, which='both', color=self.arg['colorfig'], linestyle='--', linewidth=0.5)
#         ax.set_ylim(self.arg['y_lim']) # modify the Y axis length to colocate the vector field at zero
#         ax.spines['bottom'].set_color(self.arg['colorfig'])
#         ax.spines['top'].set_color(self.arg['colorfig'])
#         ax.spines['left'].set_color(self.arg['colorfig'])
#         ax.spines['bottom'].set_linewidth(self.arg['linewidth'])
#         ax.spines['top'].set_linewidth(self.arg['linewidth'])
#         ax.spines['left'].set_linewidth(self.arg['linewidth'])
#         ax.yaxis.label.set_color(self.arg['colorfig'])
#         ax.tick_params(axis='x', colors=self.arg['colorfig'], labelsize=self.arg['fontsize'],width=self.arg['linewidth'])
#         ax.tick_params(axis='y', colors=self.arg['colorfig'], labelsize=self.arg['fontsize'],width=self.arg['linewidth'])
#         ax.set_xticks(np.arange(0,25,4))
#         plt.draw()
#     def Properties_twin(self,ax):
#         print("Properties")
#         ax.grid(True, which='both', color=self.argTwin['colorfig'], linestyle='--', linewidth=0.5)
#         ax.set_ylim(self.argTwin['y_lim']) # modify the Y axis length to colocate the vector field at zero
#         ax.spines['bottom'].set_color(self.argTwin['colorfig'])
#         ax.spines['top'].set_color(self.argTwin['colorfig'])
#         ax.spines['left'].set_color(self.argTwin['colorfig'])
#         ax.spines['bottom'].set_linewidth(self.argTwin['linewidth'])
#         ax.spines['top'].set_linewidth(self.argTwin['linewidth'])
#         ax.spines['left'].set_linewidth(self.argTwin['linewidth'])
#         ax.yaxis.label.set_color(self.argTwin['colorfig'])
#         ax.tick_params(axis='x', colors=self.argTwin['colorfig'], labelsize=self.argTwin['fontsize'],width=self.argTwin['linewidth'])
#         ax.tick_params(axis='y', colors=self.argTwin['colorfig'], labelsize=self.argTwin['fontsize'],width=self.argTwin['linewidth'])
#         ax.set_xticks(np.arange(0,25,4))
#         plt.draw()
#  
# class PolarPlot(object):
#         """
#         Plot a polar plot with rectangle representing the different characteristics of the wind and a climatic variable
#         """
#         plt.figure()
#         ax = plt.subplot(111, polar=True)
#         ax.patch.set_facecolor('None')
#         ax.patch.set_visible(False)
#         ax.set_theta_zero_location("N")
#         bars = ax.bar(self.theta, self.radii, width=self.width, bottom=0.0)
#         for r, bar in zip(self.color, bars):
#             bar.set_facecolor(plt.cm.jet(r))
#             bar.set_alpha(0.7)
#         for h,r,t in zip([6,9,13,15,18],self.radii,self.theta):
#             ax.annotate(h,xy = (t,r), fontsize=30,color='b')
#         ax.set_yticks(range(0,6,1)) 