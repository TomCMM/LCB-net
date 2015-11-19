#===============================================================================
# DESCRIPTION
#    Maniuplate the irradiance observed 
#    by the pyranometers in the Ribeirao Das Posses
#
# AUTHOR
#    Thomas Martin 11 August 2015
#===============================================================================
import pandas as pd
import glob
import datetime
#------------------------------------------------------------------------------ 
class LCB_Irr():
    """
    INPUT
        InPath: list 
        file's Path from data logger
    EXAMPLE
        'ID','Year','Day','Hour','Pira_397','Pira_369','Tlogger','LoggerV'
        105;2015;71;2;0;0;18.72;12.34
    """
    def __init__(self, InPaths):
        
        if isinstance(InPaths,list) != True:
            raise('Should be a list')

        self.InPaths = InPaths

        self._read(InPaths)

    def _read(self, InPaths):

        for InPath in InPaths:
            print(50*'-')
            print(InPath)
            try:
                newdata = pd.read_csv(InPath,sep = None)
                newdata.columns=['ID','Year','Day','Hour','Pira_397','Pira_369','Tlogger','LoggerV']
                data = pd.concat([data, newdata], ignore_index=True)
            except UnboundLocalError:
                data = pd.read_csv(InPath,sep = None)
                data.columns=['ID','Year','Day','Hour','Pira_397','Pira_369','Tlogger','LoggerV']

        data = self.__newindex(data)
        self.data = data

    def __newindex(self,data):
        """
        Convert the data logger date into a datetime time serie index
        """
        # creating index
        newindex = [ ]
        for i in data.index:
            hour = data['Hour'][i]
            hour = str(hour).zfill(4)[0:2]
            if hour == '24':
                hour ='00'
            hour = int(hour)
            minute = data['Hour'][i]
            minute = int(str(minute).zfill(4)[2:4])
            year = int(data['Year'][i])
            day = int(data['Day'][i])

            date=datetime.datetime(year,1,1,hour ,minute) + datetime.timedelta(day-1)
            newindex.append( date )
        
        data['newindex']=newindex
        data=data.set_index('newindex')
        
        return data



if __name__=='__main__':
    InPath = '/home/thomas/PhD/obs-lcb/LCBData/obs/Irradiance/data/'
    Files = glob.glob(InPath+"*")
    print Files
    Irr = LCB_Irr(Files)
    print(Irr.data)

    # data from Rsun
    InPath='/home/thomas/PhD/rsun/res/Sim_C05March2015/Irrdiance_20-02-2015.csv'
    IRsun=pd.read_csv(InPath,sep=',',index_col=0)
    IRsun.index=pd.to_datetime(IRsun.index)
    IRsun.columns=['C05']


    Irr_mean=Irr.data['Pira_397'].resample("1H",how='mean')
    Irr_mean=Irr_mean[Irr_mean < 1500]
    DT_SV_mean=DT_SV.resample("1H",how='mean')
    ClearSkyIrr=IRsun['C05'].groupby(lambda t: (t.hour)).mean() # clear sky irradiance
    
    
    DI=pd.Series(index=Irr_mean.index)
    for i in Irr_mean.index:
        DI[i]=(Irr_mean[i]/ClearSkyIrr[i])*100
    
    
    df=pd.concat([DT_SV_mean,DI],axis=1,join_axes=[Irr.index])
    df.columns=['DT','Irr']
    
    sns.set(style="darkgrid")
    color = sns.color_palette()[2]
    g = sns.jointplot("DT", "Irr", data=df, kind="reg", color=color, size=7)
    plt.show()


