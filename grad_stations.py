#===============================================================================
# DESCRIPTION
#    Group the stations in different group and plot 
#    their difference of a specified variables.
#===============================================================================

#------------------------------------------------------------------------------ 
# Library
import glob
from LCBnet_lib import *
import matplotlib

class plotclass():
    def __init__(self,series):
        self.series=series
    def ClassPeriod(self,serie):
        """
        INPUT
         time serie
        Descrpition
            Useful to make statistic  by a period of time determined by resample
        OutPut
            dataframe where the columns represent the resample period
            e.g.
                "T"
            1    A
            2    F
            1    B
            2    G
            
            out:
                1    2
            1    A    F
            2    B    G
        """
        newdf=pd.DataFrame()
        for col in range(1,24):
            subdata=serie[serie.index == col]
            subserie = pd.DataFrame(np.array(subdata),index=range(len(subdata.index)),columns=[col])
            newdf = newdf.join(subserie,how='outer')
        return newdf
    def tsplot(self,zero=None,grey=None):
        fig, ax = plt.subplots()
        if grey == True:
            colors=list()
            for i in np.arange(1,0,-0.2):
                print(plt.cm.jet(i))
                print(colors)
                colors.append(plt.cm.Greys(i))
        else:
            colors=['b','g','r','c','m','y','k','w']
        linestyles = ['-', '--', ':']
        for serie,c,l in zip(self.series,colors,linestyles):
            name=serie.columns[0]
            print(name)
            df=self.ClassPeriod(serie)
            median=df.quantile(q=0.5,axis=0)
            quartile1=df.quantile(q=0.25,axis=0)
            quartile3=df.quantile(q=0.75,axis=0)
            ax.fill_between(quartile1.index.values,quartile1.values,quartile3.values, alpha=0.3,color=c)
            ax.plot(median.index.values,median.values,linestyle=l,color=c,alpha=0.8,label=name)
        legend = ax.legend(loc='upper left', shadow=True)
        if zero == True:
            plt.axhline(0,color='black',alpha=0.2)

if __name__=='__main__':
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
    
    # Find all the clima and Hydro
    Files=glob.glob(InPath+"*")
    
    #------------------------------------------------------------------------------ 
    # Select network stations
    AttSta=att_sta()
    West=AttSta.stations(['West'])
    East=AttSta.stations(['East'])
    Valley=AttSta.stations(['valley'])
    Slope=AttSta.stations(['slope'])
    Mouth=AttSta.stations(['Medio'])
    Head=AttSta.stations(['Head'])
    Ribeirao=AttSta.stations(['Ribeirao'])
    
    SlopeEast=AttSta.stations(['slope','East'])
    SlopeWest=AttSta.stations(['slope','West'])
    ValleyEast=AttSta.stations(['valley','East'])
    ValleyWest=AttSta.stations(['valley','West'])
    
    HeadValleyEast=AttSta.stations(['Head','valley','East'])
    HeadValleyWest=AttSta.stations(['Head','valley','West'])


    Westnet=LCB_net()
    Eastnet=LCB_net()
    Valleynet=LCB_net()
    Slopenet=LCB_net()
    Mouthnet=LCB_net()
    Headnet=LCB_net()
    Ribeiraonet=LCB_net()
    
    HeadValleyWestnet=LCB_net()
    HeadValleyEastnet=LCB_net()
    
    
    for i in Files:
        print(i)
        station = LCB_station(i)
        print(station.reindex)
        if station.getpara('InPath') =='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C18clear_merge.TXT':
            station.Data.index=station.Data.index- pd.DateOffset(hours=3)
        if station.getpara('staname') in West: Westnet.add(station)
        if station.getpara('staname') in East: Eastnet.add(station)
        if station.getpara('staname') in Valley: Valleynet.add(station)
        if station.getpara('staname') in Slope: Slopenet.add(station)
        if station.getpara('staname') in Mouth: Mouthnet.add(station)  
        if station.getpara('staname') in Head: Headnet.add(station) 
        if station.getpara('staname') in Ribeirao: Ribeiraonet.add(station)
        if station.getpara('staname') in HeadValleyWest: HeadValleyWestnet.add(station) 
        if station.getpara('staname') in HeadValleyEast: HeadValleyEastnet.add(station)

# getvar dosent work for the network!!!!!
# du coup il faut tous que je fasse a la main meme pour le filtre rain



#Rain Selection
AccDailyRain=Ribeiraonet.Data['Rc mm'].resample("3H",how='sum').reindex(index=Ribeiraonet.Data.index,method='ffill')

netrain=Ribeiraonet.Data

Ribeiraonet.Data=Ribeiraonet.Data[AccDailyRain < 0.1]
Valleynet.Data=Valleynet.Data[AccDailyRain < 0.1]
Slopenet.Data=Slopenet.Data[AccDailyRain < 0.1]
Headnet.Data=Headnet.Data[AccDailyRain < 0.1]
#Mouthnet.Data=Mouthnet.Data[AccDailyRain < 0.1]
Westnet.Data=Westnet.Data[AccDailyRain < 0.1]
Eastnet.Data=Eastnet.Data[AccDailyRain < 0.1]

HeadValleyWestnet.Data=HeadValleyWestnet.Data[AccDailyRain < 0.1]
HeadValleyEastnet.Data=HeadValleyEastnet.Data[AccDailyRain < 0.1]
#------------------------------------------------------------------------------ 

DT_SV = Valleynet.getData(var='Theta C', From ='2014-08-31 00:00:00',To ='2015-04-12 00:00:00') - Slopenet.getData(var='Theta C', From ='2014-08-31 00:00:00',To ='2015-04-12 00:00:00')
DT_HM = Headnet.getData(var='Theta C', From ='2015-03-12 00:00:00',To ='2015-04-12 00:00:00') - Mouthnet.getData(var='Theta C', From ='2015-03-12 00:00:00',To ='2015-04-12 00:00:00')
DT_WE = Westnet.getData(var='Theta C', From ='2014-08-31 00:00:00',To ='2015-04-12 00:00:00') - Eastnet.getData(var='Theta C', From ='2014-08-31 00:00:00',To ='2015-04-12 00:00:00')

DU_SV = Valleynet.getData(var='Sm m/s', From ='2014-08-31 00:00:00',To ='2015-04-12 00:00:00')-Slopenet.getData(var='Sm m/s', From ='2014-08-31 00:00:00',To ='2015-04-12 00:00:00')
DU_HM = Headnet.getData(var='Sm m/s', From ='2015-03-12 00:00:00',To ='2015-04-12 00:00:00')-Mouthnet.getData(var='Sm m/s', From ='2015-03-12 00:00:00',To ='2015-04-12 00:00:00')
DU_WE = Westnet.getData(var='Sm m/s', From ='2014-08-31 00:00:00',To ='2015-04-12 00:00:00')-Eastnet.getData(var='Sm m/s', From ='2014-08-31 00:00:00',To ='2015-04-12 00:00:00')

DH_SV = Valleynet.getData(var='Ua g/kg', From ='2014-09-05 00:00:00',To ='2015-04-12 00:00:00')-Slopenet.getData(var='Ua g/kg', From ='2014-09-05 00:00:00',To ='2015-04-12 00:00:00')
DH_HM = Headnet.getData(var='Ua g/kg', From ='2015-03-12 00:00:00',To ='2015-04-12 00:00:00')-Mouthnet.getData(var='Ua g/kg', From ='2015-03-12 00:00:00',To ='2015-04-12 00:00:00')
DH_WE = Westnet.getData(var='Ua g/kg', From ='2014-09-05 00:00:00',To ='2015-04-12 00:00:00')-Eastnet.getData(var='Ua g/kg', From ='2014-09-05 00:00:00',To ='2015-04-12 00:00:00')

DH_SV=DH_SV.resample("H",how='mean')
DH_SV.index=DH_SV.index.hour
DH_SV.columns=['DH_VS']

DH_HM=DH_HM.resample("H",how='mean')
DH_HM.index=DH_HM.index.hour
DH_HM.columns=['DH_HM']

DH_WE=DH_WE.resample("H",how='mean')
DH_WE.index=DH_WE.index.hour
DH_WE.columns=['DH_WE']



DT_HM=DT_HM.resample("H",how='mean')
DT_HM.index=DT_HM.index.hour
DT_HM.columns=['DT_HM']


DT_WE=DT_WE.resample("H",how='mean')
DT_WE.index=DT_WE.index.hour
DT_WE.columns=['DT_WE']

DT_SV=DT_SV.resample("H",how='mean')
DT_SV.index=DT_SV.index.hour
DT_SV.columns=['DT_VS']

matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

plotclass([DT_SV,DT_WE,DT_HM]).tsplot(zero=True)
#plotclass([DH_SV,DH_WE,DH_HM]).tsplot(zero=True)

plt.show()

plt.close()




