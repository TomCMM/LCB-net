#==============================================================================
#   Description
#       module which fill the missing data of the station observations through multiregression analysis
#==============================================================================


# Library
from __future__ import division
from LCBnet_lib import *

import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools


class FillGap():
    """
    INPUT
        A network object
    """
    def __init__(self,network):
        self.network = network
        self.newdataframes = { } # contain the new dataframes
#        self.__findpredictor(network)

#     def __findpredictor(self,network):
#         stations = self.network.getsta([], all=True, sorted='Lon')
#         
#         available = pd.DataFrame()
#         for s in stations:
#             print(s.getData(reindex = True).dropna(how='all').index)
#             s.getData(reindex = True).dropna(how='all').index
#             news = pd.DataFrame()

    def __getpredictors(self,staname):
        """
        DESCRIPTION
            Return the stations object closest to the stations selected
        INPUT
            Station name
        EXAMPLE
            __getsta('C04')
        """
        stations = self.network.getsta([], all=True, sorted='Lon')
        station = self.network.getsta(staname)
        
        idx = stations.index(station[0])
        
        left = stations[:idx-1]
        right = stations[idx+1:]

        select = [ ]
        # "Leap frog" around the point
        for l,r in zip(range(len(left)-1),range(len(right)-1)):
            print l,r
            select.append([left[l],right[r]])
            select.append([left[l],right[r+1]])
            select.append([left[l+1],right[r]])
        
        # group pair at the right and left
        if len(left) > len(right):
            for l in range(len(left)-1):
                select.append([left[l],left[l+1]])
            for r in range(len(right)-1):
                select.append([right[r],right[r+1]])
        else:
            for l in range(len(left)-1):
                select.append([left[l],left[l+1]])
            for r in range(len(right)-1):
                select.append([right[r],right[r+1]])
        
        # rest of all possible solution to fill the possible gap
        left.reverse()
        rest = [list(x) for x in itertools.combinations(left+right, 2)]
        for i in rest:
            select .append(i)

        print select
        return select

    def fillstation(self, stations, all=None, plot = None, summary = None):

        if all == True:
            stations = self.network.getsta([], all=True).values()

        for station in stations:
            print('bite')
            newdataframe = station.getData(reindex = True) # Dataframe which stock the new data of the stations
            print('chatte')
            newdataframe['U m/s'] = station.getvar('U m/s')
            newdataframe['V m/s'] = station.getvar('V m/s')
            staname = station.getpara('staname')
            selections = self.__getpredictors(staname)
            variables = newdataframe.columns
            
            for selection in selections:
                for var in variables:
                    print(var)
                    Y = station.getvar(var) # variable to be filled
                    X1 = selection[0].getvar(var) # stations variable used to fill
                    X2 = selection[1].getvar(var)# stations variable used to fill

                    try:
                        # get parameters
                        data=pd.concat([Y, X1, X2],keys=['Y','X1','X2'],axis=1, join='inner').dropna()
                        params = self.MLR(data[['X1','X2']], data['Y'], summary = summary)
        
                        # get new fitted data
                        select = pd.concat([X1, X2],keys=['X1','X2'],axis=1, join='inner').dropna()
                        newdata = params[0] + params[1]*select['X1'] + params[2]*select['X2']
        
                        # Place fitted data in original dataframe
                        idxmissing = newdataframe[var][newdataframe[var].isnull() == True].index # slect where their is missing data
                        newdataframe[var][idxmissing] = newdata[idxmissing] # Fill the missing data with the estimated serie
                    except KeyError:
                        print('Data not present in all station')

            # Recalculate "Sm m/s" and "Dm G" with U and V
            # speed,dir = cart2pol(-1,0)
            # dir = -dir*(180/np.pi)+270
            # print(dir)
            speed,dir = cart2pol(newdataframe['U m/s'],newdataframe['V m/s'])
            dir = -dir*(180/np.pi)+270
            newdataframe['Dm G'] = dir
            newdataframe['Sm m/s'] = speed

            if plot == True:
                df = pd.concat([Y,X1,X2,newdata,newdataframe[var] ], keys=['Y','X1','X2','estimated data','Estimated replaced'],axis=1, join='outer')
                self.plotcomparison(df)

            self.newdataframes[staname] = newdataframe

    def GetDataframes(self):
        return self.newdataframes

    def WriteDataFrames(self, Outpath):
        newdataframes = self.GetDataframes()
        for staname in newdataframes.keys():
            fname = staname + '.TXT'
            newdataframes[staname].to_csv(OutPath+fname, float_format="%.2f")
            print('--------------------')
            print('Writing dataframe')
            print('--------------------')

    def MLR(self,X,Y, summary = None):
        """
        INPUT
            X: dataframe of predictor
            Y: predictant
        OUTPUT
            estimator object
        EXAMPLE
            X = df_adv[['TV', 'Radio']]
            y = df_adv['Sales']
            
            ## fit a OLS model with intercept on TV and Radio
            X = sm.add_constant(X)
            est = sm.OLS(y, X).fit()
            
            est.summary()
        """
        X = sm.add_constant(X)
        est = sm.OLS(Y, X).fit()
        if summary == True:
            print(est.summary())
            print(est.params)
        return est.params

    def plotcomparison(self,df):
        df.plot(subplots = True)
        df.plot(kind='scatter', x='Y', y='estimated data')
        plt.show()


if __name__=='__main__':
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
    OutPath= '/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    Files=glob.glob(InPath+"*")
    
    net=LCB_net()
    AttSta = att_sta()
    AttSta.setInPaths(InPath)
    AttSta.showatt()
    
    stanames = AttSta.stations(['Medio'])
    staPaths = AttSta.getatt(stanames , 'InPath')
    net.AddFilesSta(staPaths)

    gap = FillGap(net)
    gap.fillstation([], all = True, summary = True)
    gap.WriteDataFrames(OutPath)
















