from __future__ import division
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch
import copy
from scipy import interpolate
import seaborn as sns
from scipy import stats
import datetime
from geopy.distance import vincenty


def PolarToCartesian(norm,theta):
    """
    Transform polar to Cartesian where 0 = North, East =90 ....
    """
    U=norm*np.cos(map(math.radians,-theta+270))
    V=norm*np.sin(map(math.radians,-theta+270))
    return U,V


class Ink():
    """
    DESCRIPTION
        function to display a string in the monitor with specified format
    INPUT
        level: 0, MAIN TITLE should be use for class
                1, Title, should be use for methods
                2, sub title should be use for loops
                'all', print all type of level
    """
    def __init__(self,*args):
        nblevel = 4
        string = args[0]

        if len(args)>1:
            level = args[1]
        else:
            level = nblevel

        if len(args)>2:
            kwargs = args[2]
            
        else:
            kwargs = "donotexist"

        if kwargs:
            if 'v' in kwargs:
                if level <= kwargs['v'] or kwargs['v']=='all':
                    self._print(string, level)
                else:
                    pass
        elif kwargs == "donotexist":
            self._print(string, level)


    def _print(self,string, level):
        string = str(string)
        if level == 0:
            print "0"*120
            print "0"*120
            print " "*60 + string.upper()
            print "0"*120
            print "0"*120
    
        elif level == 1:
            print "o"*60
            print " "*20 + string.upper()
            print "o"*60
    
        elif level == 2:
            print "="*60
            print " "*20 + string.upper()
            print "="*60
    
        elif level == 3:
            print "-"*60
            print " "*20 + string
            print "-"*60
    
        elif level == 4:
            print "-"*60
            print " "*30 + string
            print "-"*60


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

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
              'V m/s':self.__V
              }
        
        return module[var]

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

    def FromToBy(self,How):
        
        data=self.Data[self.getpara('From'):self.getpara('To')]
        data=data.resample(self.getpara('By'),how=How)
        return data

    def reindex(self,data):
        """
        Reindex with a 2min created index
        """
        Initime=self.getpara('From')
        Endtime=self.getpara('To')
        Initime=pd.to_datetime(Initime)# be sure that the dateset is a Timestamp
        Endtime=pd.to_datetime(Endtime)# be sure that the dateset is a Timestamp

        newdata=data.groupby(data.index).first()# Only methods wich work
        idx=pd.date_range(Initime,Endtime,freq='2min')
        newdata=newdata.reindex(index=idx)
        return newdata

    def getData(self,var= None, every=None, From=None,To=None,by=None, how = None , group = None ,rainfilter = None , reindex =None):
        """
        DESCRIPTION
            More sophisticate methods to get the LCBobject data than "getvar"
        INPUT
            If no arguments passed, the methods will look on the user specified parameters of the station
            If their is no parameters passed, it will take the parameters by default
            
            var: list of variable name
            
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
        """
        if From == None:
            From=self.getpara('From')
        if To == None:
            To=self.getpara('To')
#         if by == None:
#             by=self.getpara('By')


        if var == None:
            print('Bite')
            data = self.Data

        else:
            if not isinstance(var, list):
                var = [var]
            for v in var:
                if v not in self.Data.columns:
                    try: 
                        data = self.Data
                        data[v] = self.getvar(v) # calculate the variable
                    except KeyError:
                        raise('This variable do not exist and cannot be calculated')
                else:
                    data = self.Data

        if reindex == True:
            data = self.reindex(data[From:To])
        else:
            data = data[From:To]

        if rainfilter == True: # need to be implemented in a method
            data=data[data['Rc mm'].resample("3H",how='mean').reindex(index=data.index,method='ffill')< 3]
            if data.empty:
                raise ValueError(" The rainfilter removed all the data -> ")

        if by != None:
            data=data.resample(by, how = how)

#        SHOULD BE WORKING -> I don't remeneber why I did this
#         if group == True:
#             if data.index.hour.sum() == 0:
#                 data=data.groupby(lambda t: (t.day)).mean()
#             else:
#                 if data.index.minute.sum() == 0:
#                     data=data.groupby(lambda t: (t.hour)).mean()
#                 else:
#                     data=data.groupby(lambda t: (t.hour,t.minute)).mean()
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

#         data = data[From:To] # I should remove what is before

        if var == None:
            return data
        else:
            return data[var]

    def getvar(self,varname,From=None,To=None,rainfilter=None):
        """
        DESCRIPTION
            Get a variables of the station
        INPUT:
            varname: string
        """
        
        if From == None:
            From = self.getpara('From')
        if To == None:
            To = self.getpara('To')
        
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
                print('This variable cant be calculated')
        if rainfilter == True:
            return self.__rainfilter(var)

    def DailyDiffnet(self):
  
        """
        return a dataframe which contain the difference between the station and the network
        """
        try:
            Ndata=self.daily()-self.my_net.Mean()
        except NameError:
            self.report()
        return Ndata

    def daily_h(self):
        data=self.Data[self.getpara('From'):self.getpara('To')].groupby(lambda t: t.hour).mean()
        return data

    def daily(self):
        """
        Could be replace to getData
        But I keept it as some module depend on it
        """
        data=self.Data[self.getpara('From'):self.getpara('To')].groupby(lambda t: (t.hour, t.minute)).mean()
        return data



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
        self._att = {'Ta C':{'longname': 'Temperature (C)'},
                     'Theta C':{'longname':'Potential temperature (C)'},
                     'Sm m/s':{'longname':'wind speed (m/s)'},
                     'Dm G':{'longname':'Wind direction (degree)'},
                     'U m/s':{'longname':'Zonal wind  (m/s)'},
                     'V m/s':{'longname':'Meridional wind  (m/s)'},
                     'Bat mV':{'longname':'Battery (mV)'},
                     'Rc mm':{'longname':'Accumulated precipitation (mm)'},
                     'Ua %':{'longname':'Relative humidity (%)'},
                     'Ua g/kg':{'longname':'Specific humidity (g/kg)'},
                     'Pa H':{'longname':'Pressure (H)'},
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
    """
    def __init__(self):
        self.__attributes=['Lon','Lat','network','side','Altitude','position']
        self.__stapos={
                       'C15':{'Lon':-46.237139,'Lat':-22.889639,'network':'Head','side':'East','Altitude':1342,'position':'ridge','watershed':'Ribeirao','d_river':1350},
                       'C14':{'Lon':-46.238472,'Lat':-22.889139,'network':'Head','side':'East','Altitude':1279,'position':'slope','watershed':'Ribeirao','d_river':1220},
                       'C13':{'Lon':-46.241278,'Lat':-22.888389,'network':'Head','side':'East','Altitude':1206,'position':'slope','watershed':'Ribeirao','d_river':880},
                       'C12':{'Lon':-46.243694,'Lat':-22.886278,'network':'Head','side':'East','Altitude':1127,'position':'slope','watershed':'Ribeirao','d_river':580},
                       'C11':{'Lon':-46.245861,'Lat':-22.883444,'network':'Head','side':'East','Altitude':1077,'position':'valley','watershed':'Ribeirao','d_river':120},
                       'C10':{'Lon':-46.246944,'Lat':-22.883306,'network':'Head','side':'East','Altitude':1031,'position':'valley','watershed':'Ribeirao','d_river':30},
                       'C09':{'Lon':-46.258833,'Lat':-22.870194,'network':'Head','side':'West','Altitude':1356,'position':'ridge','watershed':'Ribeirao','d_river':1640},
                       'C08':{'Lon':-46.256667,'Lat':-22.874111,'network':'Head','side':'West','Altitude':1225,'position':'slope','watershed':'Ribeirao','d_river':1290},
                       'C07':{'Lon':-46.254528,'Lat':-22.876861,'network':'Head','side':'West','Altitude':1186,'position':'slope','watershed':'Ribeirao','d_river':860},
                       'C06':{'Lon':-46.252861,'Lat':-22.877917,'network':'Head','side':'West','Altitude':1140,'position':'slope','watershed':'Ribeirao','d_river':700},
                       'C05':{'Lon':-46.251667,'Lat':-22.881167,'network':'Head','side':'West','Altitude':1075,'position':'valley','watershed':'Ribeirao','d_river':260},
                       'C04':{'Lon':-46.249083,'Lat':-22.880972,'network':'Head','side':'West','Altitude':1061,'position':'valley','watershed':'Ribeirao','d_river':200},
                       'C16':{'Lon':-46.247306,'Lat':-22.863028,'network':'Medio','side':'West','Altitude':1078,'position':'slope','watershed':'Ribeirao','d_river':350},
                       'C17':{'Lon':-46.243944,'Lat':-22.864778,'network':'Medio','side':'East','Altitude':1005,'position':'valley','watershed':'Ribeirao','d_river':60},
                       'C18':{'Lon':-46.238611,'Lat':-22.864139,'network':'Medio','side':'East','Altitude':1069,'position':'slope','watershed':'Ribeirao','d_river':560},
                       'C19':{'Lon':-46.236139,'Lat':-22.864389,'network':'Medio','side':'East','Altitude':1113,'position':'slope','watershed':'Ribeirao','d_river':790}
                       }
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

    def stations(self,values):
        """
        Return the name of the stations corresponding to a particular parameter
        """
        sta=[]
        if type(values) is not list:
            raise TypeError('Must be a list')
        for staname in self.__stapos:
            if all (k in self.__stapos[staname].values() for k in values):
                sta.append(staname)
        return sta

    def setatt(self,staname, newatt, newvalue):
        """
        DESCRIPTION
            insert a new value of a new attribut corresponding at the specific station
        """
        try:
            self.__stapos[staname][newatt] = newvalue
            self.__attributes.append(newatt)
        except KeyError:
            print staname
            raise KeyError('this station do not exist')

    def showatt(self):
        print(self.__attributes)

    def getatt(self,stanames,att):
        """
        INPUT
            staname : list. name of the stations 
            att : scalar. name of the attribut
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
        
        for staname in stanames:
            try:
                staatt.append(self.__stapos[staname][att])
            except KeyError:
                
                raise KeyError('The parameter' + att + ' do not exist')
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
            staname = os.path.basename(f)[0:3]
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
                data.plot(subplots = subplots)
                
                objectname = self.getpara('stanames')
                if isinstance(objectname, list):
                    objectname = "network" # should be implemented somewhere else

                plt.savefig(outpath+v[0:2]+objectname+"_TimePlot.png")
                print('Saved at -> ' +outpath)
                plt.close()

    def dailyplot(self,var = None, From=None, To=None, group= None, save= False, outpath = "/home/thomas/", labels = None):
        """
        Make a daily plot of the variable indicated
        """
        lcbplot = LCBplot() # get the plot object
        argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
        arglabel = lcbplot.getarg('label')
        argticks = lcbplot.getarg('ticks')
        argfig = lcbplot.getarg('figure')
        arglegend = lcbplot.getarg('legend')
        
        plt.figure(**argfig)
        print From
        print To
        
        

        for v in var:
            plt.close()
            color = iter(["r", "b"])
            for from_ , to_, label in zip(From, To, labels):
                c = color.next()
                data = self.getData(var = v, From = from_, To=to_)
                quartile1 = data.groupby(lambda x: x.hour).quantile(q=0.10)
                quartile3 = data.groupby(lambda x: x.hour).quantile(q=0.90)
                mean = data.groupby(lambda x: x.hour).mean()
                print "-->" + str(quartile1.columns)
                
                plt.fill_between(quartile1[v].index.values, quartile1[v].values, quartile3[v].values, alpha=0.1,color=c)
    
                plt.plot([], [], color=c, alpha=0.1,linewidth=10, label=(label+' q=0.90 & 0.10'))
      
    
                plt.plot(mean[v].index.values, mean[v].values,linewidth = 2, linestyle='--', color=c, alpha=0.7, label=(label+' mean'))
                
    
    
            plt.ylabel(v, **arglabel)
            plt.xlabel( "Hours", **arglabel)
            plt.grid(True)
            plt.tick_params(axis='both', which='major', **argticks)
            plt.tick_params(axis='both', which='major', **argticks)
            plt.legend(**arglegend)
    
            if not save:
                plt.show()
            else:
                plt.savefig(outpath+v[0:2]+"_dailyPlot.png")


class LCB_station(man, Plots):
    """
    DESCRIPTION
        Contain the data of ONE WXT! and the different module to transform the data in some specific way.
    PROBLEM
        SHould calculate the specific variables only when asked
    """

    def __init__(self,InPath):
        """
        PROBLEM
            on raw file the option "Header=True is needed"
        SOLUTION
            self.rawData=pd.read_csv(InPath,sep=',',index_col=0,parse_dates=True)
        """
        
        self.Data=pd.read_csv(InPath,sep=',',index_col=0,parse_dates=True)
        self.para={
        }
        self.paradef={
                    'Cp':1004, # Specific heat at constant pressure J/(kg.K)
                    'P0':1000,#Standard pressure (Hpa)
                    'R':287,#constant ....
                    'Kelvin':272.15, # constant to transform degree in kelvin
                    'E':0.622,
                    'InPath':InPath,
                    'By':'2min',
                    'To':self.Data.index[-1],
                    'From':self.Data.index[0],
                    'group':'1H',
                    "dirname":os.path.dirname(InPath),
                    "filename":os.path.basename(InPath),
                    "stanames":os.path.basename(InPath)[0:3]
        }
        self.__poschar()

    def __poschar(self):
        AttSta = att_sta()
        attributes = AttSta._att_sta__attributes
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
        for i in self.module:
            print i

    def showpara(self):
        for i in self.para:
            print i

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
                    print oldpara
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


class LCB_net(LCB_station, man):
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

    def getvarallsta(self, var = None,stanames = None, all = None, by = None, how = None, From=None,To=None):
        """
        DESCRIPTION
            return a dataframe with the selected variable from all the stations
        TODO 
            UTLISER arg* et kwarg pour passer les argument a 
            getData sans avoir besoin de tous les recrires
        """
        if all == True:
            stanames = self.getpara('stanames')
        
        df = pd.DataFrame()
        for staname in stanames:
            station = self.getsta(staname)[0]
            s = station.getData(var = var, by = by, how = how, From=From, To=To)
            s.columns = [staname]
            df = pd.concat([df,s], axis=1)

        return df

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

    def remove(self, item):
        item._LCB_station__deregister(self)# _Com_sta - why  ? ask Marcelo
        guys = self.getpara('guys')
        del guys[item.getpara('stanames')]
        self.setpara('guys',guys)
        self.min = None
        self.max = None
        self.Data = None
        for guy in self.getpara('guys'):
            self.__update(guy)

    def AddFilesSta(self,Files):
        print('Adding ')
        for i in Files:
            print(i)
            sta=LCB_station(i)
            self.add(sta)

    def add(self, items):
        """
            DESCRIPTION
                Add an LCB_object
        """

        if isinstance(items, list):
            for item in items:
                self.__add_item(item)
        else:
            print('one network')
            print items
            self.__add_item(items)

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
            print data
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

        self.__update(item)
        item._LCB_station__register(self)

    def __update(self, item):
        """
        DESCIPTION 
            Calculate the mean, Min and Max of the network
        PROBLEM
            If their is to much data the update cause memory problems
        """

        try:
            print('Updating network')
            Data=item.reindex(item.Data)
            self.max=pd.concat((self.max,Data))
            self.max=self.max.groupby(self.max.index).max()
            #------------------------------------------------------------------------------ 
            self.min=pd.concat((self.min,Data))
            self.min=self.min.groupby(self.min.index).min()
            #------------------------------------------------------------------------------
            net_data=[self.Data]*(len(self.getpara('guys'))-1)# Cette technique de concatenation utilise beaucoup de mermoire
            net_data.append(Data)
            merged_data=pd.concat(net_data)
            self.Data=merged_data.groupby(merged_data.index).mean()
            self.setpara('From',self.Data.index[0])
            self.setpara('To',self.Data.index[-1])

        except TypeError:
            print('Initiating data network')
            self.Data=Data
            self.min=Data
            self.max=Data


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
                         'labelsize':30
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