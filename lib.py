import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ManageDataLCB(object):
    """Read data LCB observations, clean the possible error and write a new file"""
    Header_info='Modulo XBee Unknown  Adress SL: Unknown\r\n'
    NbLineHeader=2
    def __init__(self,InPath,fname):
        """Take The Path and the filname of the raw files folder """
        self.InPath=InPath
        self.fname=fname
        self.__read()
        self.Header()
        self.clear()
        self.Threshold={'Pa H':{'Min':850,'Max':920},
                        'Ta C':{'Min':5,'Max':50,'MavgA':6,'MavgB':6,'MavgC':11},
                        'Ua %':{'Min':0,'Max':100},
                        'Rc mm':{'Min':0,'Max':10000},
                        'Sm m/s':{'Min':0,'Max':100},
                        'Dm G':{'Min':0,'Max':360},
                        'Bat mV':{'Min':0,'Max':10000},
                        'Vs V':{'Min':8.99,'Max':9.5}
                        }
    def Header(self):
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
        if self.fname[0] =='C' or self.fname[0] =='c':
            self.Header_var='T_St,Bat mV,Dm G,Sm m/s,Ta C,Tint C,Ua %,Pa H,Rc mm\r\n'
        if self.fname[0] == 'H' or self.fname[0] == 'h' :
            self.Header_var='T_St,Bat mV,S_10cm,S_20cm,S_30cm,S_40cm,S_60cm,S_100cm,,\r\n'
    def nbcol(self):
        nbcol=len(self.Header_var.split(','))
        return nbcol
    def __read(self):
        with open(self.InPath+self.fname) as f:
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
        if content_clear[0][0:11] != self.Header_info[0:11]:
            print("-> Rewrite title info")
            content_clear.insert(0,self.Header_info)
            print('New title',content_clear[0])
        if content_clear[1][0:4] != self.Header_var[0:4] or self.nbcol() > len(content_clear[1].split(',')):
            print("-> Rewrite Variables columns")
            content_clear.insert(1,self.Header_var)
        for idx,line in enumerate(content_clear[self.NbLineHeader::]):
            if len(line.split(',')) != self.nbcol() or line[0:1].isdigit() ==False or line[-2:]!='\r\n' or len(line.split(',')[0])!=16:
                print('-> Deleting the line ',idx+self.NbLineHeader," :",line)
                LineToDel.append(self.NbLineHeader+idx)
        for i in sorted(LineToDel, reverse=True):
            del content_clear[i]
        self.content_clear=content_clear
    def write_clean(self,OutPath,fname):
        self.clear()
        f = open(OutPath+fname+'clear',"w")
        for line in self.content_clear:
            f.write(line)
        f.close
    def append_dataframe(self,fileobject):
        """
        User input: A list of file path with the different files to merge
pd.rolling_mean(rr.Data['Ta C'],3)        Description    
            exemple: H05XXX240 will be merged with H05XXX245
        """
        try:
            self.Dataframe=self.Dataframe.append(fileobject.Dataframe).sort_index(axis=0)
            print("Merging dataframe "+fileobject.fname)
        except:
            print('It cant merge dataframe')
    def write_dataframe(self,OutPath,fname):
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
        print('000000000000000000000000000000000000000000000000000000000000')
                
                
class LCB_net(object):
    def __init__(self):
        self.guys = []
        self.min = None
        self.max = None
        self.mean=None
    def Max(self):
        return self.max
    def Min(self):
        return self.min
    def Mean(self):
        return self.mean
    def remove(self, item):
        item.deregister(self)
        self.guys.remove(item)
        self.min = None
        self.max = None
        self.mean = None
        for guy in self.guys:
            self.__update__(guy)
    def add(self, item):
        print('Adding To the network -> '+item.get_InPath())
        self.guys.append(item)
        self.__update__(item)
        item.register(self)
    def __update__(self, item):
        for i in item.daily().columns:
                try:
                    self.max[i]= pd.DataFrame({ 'net' : self.max[i], 'NewSta' : item.daily()[i] }).max(1)
                    self.min[i]= pd.DataFrame({ 'net' : self.min[i], 'NewSta' : item.daily()[i] }).min(1)
                    self.mean[i]= (self.mean[i]*(len(self.guys)-1)+item.daily()[i])/len(self.guys)
                except KeyError:
                    print('Adding a new column: '+i)
                    print(item.daily()[i])
                    self.max[i]=item.daily()[i]
                    self.min[i]=item.daily()[i]
                    self.mean[i]=item.daily()[i]
                except TypeError:
                    print('Initiating data network')
                    self.max=item.daily()
                    self.min=item.daily()
                    self.mean=item.daily()
    def report(self):
        print "Their is %d stations in the network" % len(self.guys)
        for guy in self.guys:
            print guy
        print ""

class LCB_station(object):
    """
    Contain the data of ONE WXT! and the different module to transform the data in some specific way.
    """
    def __init__(self,InPath):
        self.InPath=InPath
        self.rawData=pd.read_csv(InPath,sep=',',index_col=0,parse_dates=True)#on raw file the option "Header=True is needed"
        self.Data=pd.read_csv(InPath,sep=',',index_col=0,parse_dates=True)
        #self.daily()        
        self.From=self.Data.index[0]
        self.To=self.Data.index[-1]
        self.By='H'
        self.q()
        self.my_net=None
    def get_InPath(self):
        return self.InPath
    def deregister(self, net):
        self.my_net = None
    def register(self, net):
        self.my_net = net
        print(self.my_net)
    def report(self):
        print "Myself is:", self.InPath, self
        if self.my_net:
            print "My net: ",self.my_net
        else:
            print "i dont belong to any net"
        print ""
    def Set_From(self,From):
        """
        Permit to set the start of the time serie use in the module FromToBy()
        Take 1 argument on the following form "2014-09-03 22:30:00"
        """
        self.From=From
    def Set_To(self,To):
        """
        Permit to set the end of the time serie use in the module FromToBy()
        Take 1 argument on the following form "2014-09-03 22:30:00"
        """
        self.To=To
    def Set_By(self,By):
        """
        Permit to set the time range of the time serie use in the module FromToBy()
        Take 1 argument on the following form "3H"
        """
        self.By=By    
    def daily(self):
        data=self.Data[self.From:self.To].groupby(lambda t: (t.hour,t.minute)).mean()
        return data
    def daily_h(self):
        data=self.Data[self.From:self.To].groupby(lambda t: t.hour).mean()
        return data
    def DailyDiffnet(self):
        """
        return a dataframe which contain the difference between the station and the network
        """
        try:
            Ndata=self.daily()-self.my_net.Mean()
        except NameError:
            self.report()
        return Ndata
    def daily_3h(self):
        DD=self.Data['2014-09-03 22:30:00':'2014-09-05 00:00:00']
        DD_0h=DD.between_time('22:30','01:30')
        DD_3h=DD.between_time('01:30','04:30')
        DD_6h=DD.between_time('04:30','07:30')
        DD_9h=DD.between_time('07:30','10:30')
        DD_13h=DD.between_time('10:30','13:30')
        DD_15h=DD.between_time('13:30','16:30')
        DD_18h=DD.between_time('16:30','19:30')
        DD_21h=DD.between_time('19:30','22:30')    
        DD_std=[DD_6h['Dm G'].describe()[2],DD_9h['Dm G'].describe()[2],DD_13h['Dm G'].describe()[2],DD_15h['Dm G'].describe()[2],DD_18h['Dm G'].describe()[2]]
        DD_mean=[DD_6h['Sm m/s'].describe()[1],DD_9h['Sm m/s'].describe()[1],DD_13h['Sm m/s'].describe()[1],DD_15h['Sm m/s'].describe()[1],DD_18h['Sm m/s'].describe()[1]]
        DD_T=[DD_6h['Ta C'].describe()[1],DD_9h['Ta C'].describe()[1],DD_13h['Ta C'].describe()[1],DD_15h['Ta C'].describe()[1],DD_18h['Ta C'].describe()[1]]
        self.Data_3H=DD.resample('3H',how='mean')
        self.Data_3H_min=DD.resample('3H',how='min')
        self.Data_3H_max=DD.resample('3H',how='max')
        N =len(self.Data_3H)
        self.theta=map(math.radians,np.array(DD_mean)-np.array(DD_std))
        self.radii=DD_mean
        self.width=map(math.radians,np.array(DD_std)*2)
        self.color=(np.array(DD_T)-10)/(20-10)
    def FromToBy(self):
        data=self.rawData[self.From:self.To]
        data=data.resample(self.By,how='mean')
        return data
    def Es(self):
        """
        Return the vapor pressure at saturation from the Temperature
        
        Bolton formula derivated from the clausius clapeyron equation 
        Find in the American Meteorological Glaussary
        T is given in Degree celsus
        es is given in Kpa (but transform in hPa in this code)
        """
        es=0.6112*np.exp((17.67*self.Data['Ta C'])/(self.Data['Ta C']+243.5))*10 #hPa
        return es
    def Ws(self):
        """
        Return Mixing ratio at saturation calculated from
        the vapor pressure at saturation and the total Pressure
        """
        E=0.622
        ws=E*(self.Es()/(self.Data['Pa H']-self.Es()))
        return ws
    def q(self):
        """
        Compute the specfic humidity from the relative humidity, Pressure and Temperature
        """
        try:
            q=((self.Data['Ua %']/100)*self.Ws())/(1-(self.Data['Ua %']/100)*self.Ws())*1000
            self.Data['Ua g/kg']=q
        except KeyError:
            print('Cant compute the specific humidity')           

class Vector(object):
    def __init__(self,data,Theta,Norm):
        self.data=data
        self.Theta=Theta
        self.Norm=Norm
        self.type='default'
        self.arg=dict()
        self.__Types()#Create an initial library of differents option defining different type of grafics
        self.__Arg()# Final library of option
    def __Types(self):
        self.Types={
        'default': {
            'colorvar':['k','b'],
            'Poswind':0,
            'colorfig':'k',
            'colorwind':'k',
            'vectorlength':40,
            'Poswindscale':[5,3],
            'ScaleLength':5,
            'fontsize':40,
            'twin':False,# to plot on two axis
            'linewidth':4,
            'y_lim':[-4,4]},
        'AnomalieT': {
            'Poswind':-3.5,
            'colorvar':['k','b'],
            'colorfig':'b',
            'colorwind':'k',
            'vectorlength':40,
            'Poswindscale':[5,3],
            'ScaleLength':5,
            'fontsize':40,
            'y_lim':[-6,6],
            'linewidth':4},
        'AnomalieH':{
            'Poswind':-1.5,
            'colorvar':['k','b'],
            'colorfig':'k',
            'colorwind':'k',
            'vectorlength':30,
            'Poswindscale':[5,1.25],
            'ScaleLength':2,
            'fontsize':40,
            'linewidth':4,
            'y_lim':[-2,2]},
        'AbsolueT':{
            'Poswind':14,
            'colorvar':['k','b'],
            'colorfig':'b',
            'colorwind':'k',
            'vectorlength':40,
            'Poswindscale':[5,25],
            'ScaleLength':5,
            'fontsize':40,
            'linewidth':4,
            'y_lim':[5,30]},
        'AbsolueH':{
            'Poswind':7,
            'colorvar':['k','b'],
            'colorfig':'k',
            'colorwind':'k',
            'vectorlength':40,
            'Poswindscale':[5,11],
            'ScaleLength':5,
            'fontsize':40,
            'linewidth':4,
            'y_lim':[6.5,12]},
            }
    def __Extras(self):
        if self.type=='AnomalieH' or self.type=='AnomalieT':
            self.ax.plot(self.Theta.index,[0]*len(self.Theta.index),color=self.arg['colorfig'],linestyle='-', linewidth=self.arg['linewidth']-2)
        if self.arg['twin']==True:
            self.ax2 = self.ax.twinx()
            self.ax2.plot([x[0]+x[1]/60 for x in self.data[1].index],self.data[1],color=self.argTwin['colorvar'][1],linestyle='-', linewidth=self.argTwin['linewidth'])
            self.Properties_twin(self.ax2)
    def __Arg(self):
        self.arg=dict(self.arg.items()+self.Types[self.type].items())
    def SetOption(self,option,var):
        "Change the value of a default option. type-> library. option-> graphical option, var-> new value to collocate"
        self.arg[option]=var
    def SetType(self,Type):
        "Change the defaut option with another library of option"
        self.type=Type
        self.__Arg()
    def SetTypeTwin(self,Type):
        self.argTwin=self.Types[Type]
    def report(self):
        print('current option choosen: '+str(self.type))
        print('The parameters of this option are: '+str(self.arg))
        print('To change the Type please use .SetType(''''option'''')')
        print('To change an option in the current Type please use .SetOption')
    def plot(self):
        print("Plot")
        self.Main()
        self.__Extras()
        self.Properties(self.ax)
    def Main(self):
        print("Main")
        V=np.cos(map(math.radians,self.Theta+180))*self.Norm
        U=np.sin(map(math.radians,self.Theta+180))*self.Norm
        X=self.Theta.index
        Y=[self.arg['Poswind']]*len(X)
        Fig=plt.figure()
        ax=plt.gca()
        q=ax.quiver(X,Y,U,V,scale=self.arg['vectorlength'],color=self.arg['colorwind'])
        p = plt.quiverkey(q,self.arg['Poswindscale'][0],self.arg['Poswindscale'][1],self.arg['ScaleLength'],str(self.arg['ScaleLength'])+" m/s",coordinates='data',color=self.arg['colorwind'],labelcolor=self.arg['colorwind'],fontproperties={'size': 30})
        for idx,p in enumerate(self.data):
            if self.arg['twin']==False or idx<1:
                print("is printing :  "+str(idx) )
                ax.plot([x[0]+x[1]/60 for x in p.index],p,color=self.arg['colorvar'][idx],linestyle='-', linewidth=self.arg['linewidth'])
                print("finish :  "+str(idx) )
        self.ax=ax
    def Properties(self,ax):
        print("Properties")
        ax.grid(True, which='both', color=self.arg['colorfig'], linestyle='--', linewidth=0.5)
        ax.set_ylim(self.arg['y_lim']) # modify the Y axis length to colocate the vector field at zero
        ax.spines['bottom'].set_color(self.arg['colorfig'])
        ax.spines['top'].set_color(self.arg['colorfig'])
        ax.spines['left'].set_color(self.arg['colorfig'])
        ax.spines['bottom'].set_linewidth(self.arg['linewidth'])
        ax.spines['top'].set_linewidth(self.arg['linewidth'])
        ax.spines['left'].set_linewidth(self.arg['linewidth'])
        ax.yaxis.label.set_color(self.arg['colorfig'])
        ax.tick_params(axis='x', colors=self.arg['colorfig'], labelsize=self.arg['fontsize'],width=self.arg['linewidth'])
        ax.tick_params(axis='y', colors=self.arg['colorfig'], labelsize=self.arg['fontsize'],width=self.arg['linewidth'])
        ax.set_xticks(np.arange(0,25,4))
        plt.draw()
    def Properties_twin(self,ax):
        print("Properties")
        ax.grid(True, which='both', color=self.argTwin['colorfig'], linestyle='--', linewidth=0.5)
        ax.set_ylim(self.argTwin['y_lim']) # modify the Y axis length to colocate the vector field at zero
        ax.spines['bottom'].set_color(self.argTwin['colorfig'])
        ax.spines['top'].set_color(self.argTwin['colorfig'])
        ax.spines['left'].set_color(self.argTwin['colorfig'])
        ax.spines['bottom'].set_linewidth(self.argTwin['linewidth'])
        ax.spines['top'].set_linewidth(self.argTwin['linewidth'])
        ax.spines['left'].set_linewidth(self.argTwin['linewidth'])
        ax.yaxis.label.set_color(self.argTwin['colorfig'])
        ax.tick_params(axis='x', colors=self.argTwin['colorfig'], labelsize=self.argTwin['fontsize'],width=self.argTwin['linewidth'])
        ax.tick_params(axis='y', colors=self.argTwin['colorfig'], labelsize=self.argTwin['fontsize'],width=self.argTwin['linewidth'])
        ax.set_xticks(np.arange(0,25,4))
        plt.draw()


def PolarPlot(self):
        """
        Plot a polar plot with rectangle representing the different characteristics of the wind and a climatic variable
        """
        plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.patch.set_facecolor('None')
        ax.patch.set_visible(False)
        ax.set_theta_zero_location("N")
        bars = ax.bar(self.theta, self.radii, width=self.width, bottom=0.0)
        for r, bar in zip(self.color, bars):
            bar.set_facecolor(plt.cm.jet(r))
            bar.set_alpha(0.7)
        for h,r,t in zip([6,9,13,15,18],self.radii,self.theta):
            ax.annotate(h,xy = (t,r), fontsize=30,color='b')
        ax.set_yticks(range(0,6,1)) 