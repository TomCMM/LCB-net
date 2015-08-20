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

#===============================================================================
# Clean - Merge -LCB Data
#===============================================================================
def PolarToCartesian(norm,theta):
    """
    Transform polar to Cartesian where 0 = North, East =90 ....
    """
    U=norm*np.cos(map(math.radians,-theta+270))
    V=norm*np.sin(map(math.radians,-theta+270))
    return U,V

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

class ManageDataLCB(object):
    """Read data LCB observations, clean the possible error and write a new file"""
    Header_info='Modulo XBee Unknown  Adress SL: Unknown\r\n'
    NbLineHeader=2
    def __init__(self,InPath,fname):
        """Take The Path and the filname of the raw files folder """
        self.InPath=InPath
        self.fname=fname
        self.__read(InPath,fname)
        self.Header()
        self.clear()
        self.Threshold={'Pa H':{'Min':850,'Max':920},
                        'Ta C':{'Min':0,'Max':50},#,'MavgA':6,'MavgB':6,'MavgC':11
                        'Ua %':{'Min':0,'Max':100},
                        'Rc mm':{'Min':0,'Max':10000},
                        'Sm m/s':{'Min':0,'Max':100},
                        'Dm G':{'Min':0,'Max':360},
                        'Bat mV':{'Min':0,'Max':10000},
                        'Vs V':{'Min':8.5,'Max':9.5}
                        }
    def Header(self,type=None):
#        headers = {
#            'C': 'T_St,Bat mV,Dm G,Sm m/s,Ta C,Tint C,Ua %,Pa H,Rc mm\r\n',
#           'c': 'T_St,Bat mV,Dm G,Sm m/s,Ta C,Tint C,Ua %,Pa H,Rc mm\r\n',
#            'H': 'T_S t,Bat mV,S_10cm,S_20cm,S_30cm,S_40cm,S_60cm,S_100cm,,\r\n',
#           'h': 'T_S t,Bat mV,S_10cm,S_20cm,S_30cm,S_40cm,S_60cm,S_100cm,,\r\n',
#          }
#     try:   
#         self.Header_var = headers[self.fname[0]]
#     except KeyError:
#        print('Their is not such Header') 
#       self.Header_var = None

        if type ==1:
            var='T_St,Bat mV,Dm G,Sm m/s,Ta C,Tint C,Ua %,Pa H,Rc mm,Vs V\r\n'
            return var

        if type ==2:
            var='T_St,Bat mV,Dm G,Sm m/s,Ta C,Ua %,Pa H,Rc mm,Vs V\r\n'
            return var

        if self.fname[0] =='C' or self.fname[0] =='c':
            self.Header_var='T_St,Bat mV,Dm G,Sm m/s,Ta C,Tint C,Ua %,Pa H,Rc mm\r\n'

        if self.fname[0] == 'H' or self.fname[0] == 'h' :
            self.Header_var='T_St,Bat mV,S_10cm,S_20cm,S_30cm,S_40cm,S_60cm,S_100cm,,\r\n'

    def nbcol(self):
        nbcol=len(self.Header_var.split(','))
        return nbcol

    def __read(self, InPath, fname):
        with open(InPath+fname) as f:
            content = f.readlines()
        self.content=content

        print("======================================================================================")
        print('OPEN THE FILE: '+self.fname)
        try:
            self.Dataframe=pd.read_csv(self.InPath+self.fname, sep=',',header=True,index_col=0,parse_dates=True)
            print('The file is Clean - A dataframe has been created')
        except:
            print('The file is Dirty! - I cant create a dataframe - Clean before and import again')

    def clear(self):
        LineToDel=[]
        content_clear=self.content

        # Check if the file is empty
        if not content_clear:
            print("-> The File is empty")
            content_clear.insert(0,self.Header_info)
            content_clear.insert(1,self.Header(type=1))
            print('New title',content_clear[0])
        
        if content_clear[0][0:11] != self.Header_info[0:11]:
            print("-> Rewrite title info")
            content_clear.insert(0,self.Header_info)
            print('New title',content_clear[0])

        if content_clear[1][0:4] != self.Header_var[0:4]:
            print("-> No Header -> Rewrite Header columns")
            content_clear.insert(1,self.Header_var)

        for idx,line in enumerate(content_clear[self.NbLineHeader::]):
            if len(line.split(',')) != self.nbcol() or line[0:1].isdigit() == False or line[-2:]!='\r\n' or len(line.split(',')[0])!=16:
                print('-> Deleting the line ',idx+self.NbLineHeader," :",line)
                LineToDel.append(self.NbLineHeader+idx)
        for i in sorted(LineToDel, reverse=True):
            del content_clear[i]
        self.content_clear=content_clear
        if len(self.content_clear) >2:
            if  len(content_clear[1].split(',')) > len(content_clear[2].split(',')):
                print("-> Less data columns than header _> Rewrite header")
                del content_clear[1]
                content_clear.insert(1,self.Header(type=2))
            if len(content_clear[1].split(',')) < len(content_clear[2].split(',')):
                print("-> More data columns than header -> Rewrite header")
                del content_clear[1]
                content_clear.insert(1,self.Header(type=1))

    def write_clean(self,OutPath,fname):
        self.clear()
        fname = fname.upper()
        f = open(OutPath+fname+'clear',"w")
        for line in self.content_clear:
            f.write(line)
        f.close

    def append_dataframe(self,fileobject):
        """
        User input: A list of file path with the different files to merge
                Description    
            exemple: H05XXX240 will be merged with H05XXX245
        """
        try:
            self.Dataframe=self.Dataframe.append(fileobject.Dataframe).sort_index(axis=0)
            print("Merging dataframe "+fileobject.fname)
        except:
            print('It cant merge dataframe')

    def write_dataframe(self,OutPath, fname):
        self.Dataframe.to_csv(OutPath+fname)
        print('--------------------')
        print('Writing dataframe')
        print('--------------------')
    def clean_dataframe(self):
        print('000000000000000000000000000000000000000000000000000000000000')
        for var in self.Dataframe.columns:
            try:
                if var != 'Vs V':
                    print('Filtering ->  ', var)
                    self.Dataframe=self.Dataframe[(self.Dataframe[var]>=self.Threshold[var]['Min']) & (self.Dataframe[var]<=self.Threshold[var]['Max']) ]# Threshold
                else:
                    print('Filtering battery')
                    self.Dataframe=self.Dataframe[((self.Dataframe['Vs V']>=self.Threshold['Vs V']['Min']) & (self.Dataframe['Vs V']<=self.Threshold['Vs V']['Max'])) | (self.Dataframe['Vs V'].isnull()) ]# Threshold
            except KeyError:
                print('no Threshold for '+var)
            try:
                self.Dataframe=self.Dataframe[(np.abs(pd.rolling_mean(self.Dataframe[var],30)-self.Dataframe[var])<self.Threshold[var]['MavgB'])]# inter-daily
                self.Dataframe=self.Dataframe[(np.abs(pd.rolling_mean(self.Dataframe[var],5)-self.Dataframe[var])<self.Threshold[var]['MavgA'])]# inter-daily
                self.Dataframe=self.Dataframe[(np.abs(pd.rolling_mean(self.Dataframe[var],180)-self.Dataframe[var])<self.Threshold[var]['MavgC'])]# inter-daily
                print('The running mean filter on',[var],' as removed  |---> [',
                      len(self.Dataframe[(np.abs(pd.rolling_mean(self.Dataframe[var],30)-self.Dataframe[var])>self.Threshold[var]['MavgB'])]),
                      ' and ',
                      len(self.Dataframe[(np.abs(pd.rolling_mean(self.Dataframe[var],5)-self.Dataframe[var])>self.Threshold[var]['MavgA'])]),
                      'and',
                      len(self.Dataframe[(np.abs(pd.rolling_mean(self.Dataframe[var],180)-self.Dataframe[var])>self.Threshold[var]['MavgC'])])
                      ,'] data')
            except KeyError:
                print('no Mavg threshold for this variable:->  ',var)
        # drop duplicates
        self.Dataframe = self.Dataframe.drop_duplicates()
        # remove null colomns (e.g. Unnamed)
        self.Dataframe = self.Dataframe.dropna(axis = 1, how ='all')
        print('000000000000000000000000000000000000000000000000000000000000')


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

    def getData(self,var= None,From=None,To=None,by=None , group = None ,rainfilter = None , reindex ="None"):
        """
        DESCRIPTION
            Get the data of the station and resample them
        INPUT
            If no arguments passed, the methods will look on the user specified parameters of the station
            If their is no parameters passed, it will take the parameters by default
            
            group: 'D':Day, 'H':hour , 'M':minute
        """
        if From == None:
            From=self.getpara('From')
        if To == None:
            To=self.getpara('To')
        if by == None:
            by=self.getpara('By')

        if var == None:
            print('Bite')
            data = self.Data
        else:
            if var not in self.Data.columns:
                try: 
                    data = self.Data
                    data[var] = self.getvar(var)
                except KeyError:
                    raise('This variable do not exist and cannot be calculated')
            else:
                data = self.Data


#------------------------------------------------------------------------------ 
        if reindex == True:
            data = self.reindex(data[From:To])
        else:
            data = data[From:To]

#------------------------------------------------------------------------------ 
        if rainfilter == True: # need to be implemented in a method
            data=data[data['Rc mm'].resample("3H",how='mean').reindex(index=data.index,method='ffill')< 3]
            if data.empty:
                raise ValueError(" The rainfilter removed all the data -> ")

#------------------------------------------------------------------------------ 
        if by != None:
            data=data.resample(by)

#------------------------------------------------------------------------------ 
        if group == True:
            if data.index.hour.sum() == 0:
                data=data.groupby(lambda t: (t.day)).mean()
            else:
                if data.index.minute.sum() == 0:
                    data=data.groupby(lambda t: (t.hour)).mean()
                else:
                    data=data.groupby(lambda t: (t.hour,t.minute)).mean()
#------------------------------------------------------------------------------ 
        if var == None:
            return data
        else:
            return data[var]

    def getvar(self,varname,From=None,To=None,rainfilter=None):
        """
        Get a variables of the station
        
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

#             except IndexError:
#                 print('Cant calculate because no data')
#                 var=pd.DataFrame(self.Data,index=self.Data.index,columns=[varname])
#                 var=var[varname][self.getpara('From'):self.getpara('To')]
#                 return var
#             except KeyError:
#                 print('Impossible to calculate this variable')
#             except IndexError:
#                 var=self.getvarRindex(varname)
#                 print('Reindexing')
#                 return var
#         except IndexError:
#             var=self.getvarRindex(varname)
#             print('Reindexing')
#             return var

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

class att_sta(object):
    """
    Class containing metadata informations about the climatic network in Extrema 
    """
    def __init__(self):
        self.__attributes=['Lon','Lat','network','side','Altitude','position']
        self.__stapos={
                       'C15':{'Lon':-46.237139,'Lat':-22.889639,'network':'Head','side':'East','Altitude':1342,'position':'ridge','watershed':'Ribeirao'},
                       'C14':{'Lon':-46.238472,'Lat':-22.889139,'network':'Head','side':'East','Altitude':1279,'position':'slope','watershed':'Ribeirao'},
                       'C13':{'Lon':-46.241278,'Lat':-22.888389,'network':'Head','side':'East','Altitude':1206,'position':'slope','watershed':'Ribeirao'},
                       'C12':{'Lon':-46.243694,'Lat':-22.886278,'network':'Head','side':'East','Altitude':1127,'position':'slope','watershed':'Ribeirao'},
                       'C11':{'Lon':-46.245861,'Lat':-22.883444,'network':'Head','side':'East','Altitude':1077,'position':'valley','watershed':'Ribeirao'},
                       'C10':{'Lon':-46.246944,'Lat':-22.883306,'network':'Head','side':'East','Altitude':1031,'position':'valley','watershed':'Ribeirao'},
                       'C09':{'Lon':-46.258833,'Lat':-22.870194,'network':'Head','side':'West','Altitude':1356,'position':'ridge','watershed':'Ribeirao'},
                       'C08':{'Lon':-46.256667,'Lat':-22.874111,'network':'Head','side':'West','Altitude':1225,'position':'slope','watershed':'Ribeirao'},
                       'C07':{'Lon':-46.254528,'Lat':-22.876861,'network':'Head','side':'West','Altitude':1186,'position':'slope','watershed':'Ribeirao'},
                       'C06':{'Lon':-46.252861,'Lat':-22.877917,'network':'Head','side':'West','Altitude':1140,'position':'slope','watershed':'Ribeirao'},
                       'C05':{'Lon':-46.251667,'Lat':-22.881167,'network':'Head','side':'West','Altitude':1075,'position':'valley','watershed':'Ribeirao'},
                       'C04':{'Lon':-46.249083,'Lat':-22.880972,'network':'Head','side':'West','Altitude':1061,'position':'valley','watershed':'Ribeirao'},
                       'C16':{'Lon':-46.247306,'Lat':-22.863028,'network':'Medio','side':'West','Altitude':1078,'position':'slope','watershed':'Ribeirao'},
                       'C17':{'Lon':-46.243944,'Lat':-22.864778,'network':'Medio','side':'East','Altitude':1005,'position':'valley','watershed':'Ribeirao'},
                       'C18':{'Lon':-46.238611,'Lat':-22.864139,'network':'Medio','side':'East','Altitude':1069,'position':'slope','watershed':'Ribeirao'},
                       'C19':{'Lon':-46.236139,'Lat':-22.864389,'network':'Medio','side':'East','Altitude':1113,'position':'slope','watershed':'Ribeirao'}
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
        position=[]
        staname=[]
        pos={}
        for i in sta:
            pos[i]=self.getatt(i, latlon)
        sorted_pos = sorted(pos.items(), key=operator.itemgetter(1))
        for i in sorted_pos:
            position.append(i[1])
            staname.append(i[0])
        return {'staname':staname,'position':position}

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

    def getatt(self,staname,att):
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
        if type(staname) != list:
            print ('staname should be a list')
            try:
                print('convert into a list')
                if type(staname) == str:
                    staname = [staname]
            except:
                raise
        
        for s in staname:
            try:
                staatt.append(self.__stapos[s][att])
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


class LCB_station(man):
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
                    "staname":os.path.basename(InPath)[0:3]
        }
        self.__poschar()

    def __poschar(self):
        AttSta = att_sta()
        attributes = AttSta._att_sta__attributes
        try:
            for att in attributes:
                self.setpara(att,AttSta.getatt(self.getpara('staname'), att))
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

    def setpara(self,name,value):
        self.para[name]=value
        if name == 'To' or name == 'From':
            time=pd.to_datetime(value)
            self.para[name]=time
        print('The para '+str(name)+ 'has been set to'+str(value))

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
        self.__guys = {}
        self.__stanames=[]
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

    def getsta(self,staname, all=None, sorted=None):
        """
        Description
            Input the name of the station and give you the station object of the network
            staname : List
            if "all==True" then return a dictionnary containing 
            all the stations names and their respective object
            
            if sorted != None. Le paramtere fourni sera utilise pour ordonner en ordre croissant la liste de stations 
            dans le reseau
        Example
            net.getsta('C04')
            net.getsta('',all=True)
            net.getsta('',all=True,sorted='Lon')
        """
        
        if type(staname) != list:
            print ('staname should be a list')
            try:
                print('convert into a list')
                if type(staname) == str:
                    staname = [staname]
            except:
                raise

        if all==True:
            try:
                sta = self.getpara('guys')
            except:
                print('COULDNT GIVE THE NAME OF ALL THE STATIONS')

        else:
            try:
                sta = [ ]
                for s in staname:
                    sta.append( self.__guys[s])
            except KeyError:
                raise KeyError('This stations is not in the network')

        if sorted != None and all == True:
            names = self.getpara('stanames')
            sortednames=att_sta().sortsta(names,sorted)
            sortednames=sortednames['staname']
            sta = [ ]
            for i in sortednames:
                sta.append(self.__guys[i])
        return sta 

    def report(self):
        print "Their is %d stations in the network" % len(self.__guys)
        for i,v in enumerate(self.__guys):
            v
            self.__stanames[i]
    def remove(self, item):
        item._com_sta__deregister(self)# _Com_sta - why  ? ask Marcelo
        self.guys.remove(item)
        self.min = None
        self.max = None
        self.Data = None
        for guy in self.__guys:
            self.__update__(guy)

    def AddFilesSta(self,Files):
        print('Adding ')
        for i in Files:
            print(i)
            sta=LCB_station(i)
            self.add(sta)

    def add(self, item):
        print('========================================================================================')
        print('Adding To the network -> '+item.getpara('InPath'))
        staname=item.getpara('staname')
        self.__stanames.append(staname)
        self.__guys[staname]=item

        
        self.setpara('stanames',self.__stanames)
        self.setpara('guys',self.__guys)
        self.__update(item)
        item._LCB_station__register(self)# _com_sta why ? ask Marcelo

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
            net_data=[self.Data]*(len(self.__guys)-1)# Cette technique de concatenation utilise beaucoup de mermoire
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
        # I dont know what is the following code
#         for i in item.daily().columns:
#                 try:
#                     self.max[i]= pd.DataFrame({ 'net' : self.max[i], 'NewSta' : item.daily()[i] }).max(1)
#                     self.min[i]= pd.DataFrame({ 'net' : self.min[i], 'NewSta' : item.daily()[i] }).min(1)
#                     self.mean[i]= (self.mean[i]*(len(self.guys)-1)+item.daily()[i])/len(self.guys)
#                 except KeyError:
#                     print('Adding a new column: '+i)
#                     print(item.daily()[i])
#                     self.max[i]=item.daily()[i]
#                     self.min[i]=item.daily()[i]
#                     self.mean[i]=item.daily()[i]
#                 except TypeError:
#                     print('Initiating data network')
#                     self.max=item.daily()
#                     self.min=item.daily()
#                     self.mean=item.daily()


class Plots():
    """
    Class container
    Contain all the plot which can be applied to a station
    """

    def __init__(self):
        pass

    def TimePlot(self,var, Byvar = None, group = None):
            InPath=self.LCB_station.getpara('InPath')
            out=self.getpara('OutPath')
            
            From=self.LCB_station.getpara('From')
            To=self.LCB_station.getpara('To')
            data=self.LCB_station.getData(var=var,From=From,To=To,by=Byvar, group = group)
            staname= self.LCB_station.getpara('staname')

            data.plot()
            plt.savefig(out+"_"+staname+'_'+var+'.png')
            print('Saved at -> ' +out)
            plt.close()

    

class LCBplot(Plots):
    def __init__(self,LCB_object):
        """
        INPUT
            LCB_object: can be either a station or a network
        This class could made to be used both on the ARPS and LCBnet program
        """

        self.para= { }
        self.paradef= {
            'OutPath':'/home/thomas/',
            'screen_width':1920,
            'screen_height':1080,
            'DPI':96,
            }
        self.__figwidth()
        self.__subtitle(LCB_station)
        self.LCB_station=LCB_station

    def __figwidth(self):
        width=self.getpara('screen_width')
        height=self.getpara('screen_height')
        DPI=self.getpara('DPI')
        wfig=width/DPI #size in inches 
        hfig=height/DPI
        self.setpara('wfig',wfig)
        self.setpara('hfig',hfig)

    def __subtitle(self,LCB_station):
        """
        Write the subtitle of the plot
        """
        sub=LCB_station.getpara('From')
        self.setpara('subtitle', sub)

    def setpara(self,parameter,value):
        self.para[parameter]=value
        print(str(parameter)+' has been set to -> '+ str(value))

    def getpara(self,parameter):
        try:
            return self.para[parameter]
        except KeyError:
            print(parameter + ' has been not set -> Default value used ['+str(self.paradef[parameter])+']')
            try:
                return self.paradef[parameter]
            except KeyError:
                print(parameter+ ' dont exist')

    def delpara(self,varname):
        try:
            del self.para[varname]
            print('Deleted parameter-> ',varname)
        except KeyError:
            print('This parameter dont exist')

    def __setparadef(self,parameter,value):
        self.paradef[parameter]=value
        print(str(parameter)+' has been set by [default] to -> '+ str(value))

    def __getparadef(self,parameter):
        try:
            return self.paradef[parameter]
        except KeyError:
                print(parameter+ ' by [default] dont exist')

    def __levels(self,varname):
        self.paradef['nlevel']=10# number of discrete variabel level
        self.paradef['varmax']=int(self.arps.get(varname).data.max())
        self.paradef['varmin']=int(self.arps.get(varname).data.min())
        varmax=self.getpara('varmax')
        varmin=self.getpara('varmin')
        nlevel=self.getpara('nlevel')
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