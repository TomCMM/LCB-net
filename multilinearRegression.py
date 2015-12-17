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

    def __getpredictors(self,staname):
        """
        DESCRIPTION
            Return a couple object closest to the stations selected
        INPUT
            Station name
        EXAMPLE
            __getpredictors('C04')
        """

        stations = self.network.getsta([], all=True, sorted='Lon')['stations']
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

        return select

    def fillstation(self, stanames, all=None, plot = None, summary = None, From=None, To=None):
        """
        DESCRIPTION
            Check every variable of every stations and try to fill 
            them with the variables of the two nearest station for every time.
        INPUT
            From: From where to select the data
            To: when is the ending
        """
        if all == True:
            stations = self.network.getsta([], all=True).values()
        else:
            stations = self.network.getsta(stanames)

        for station in stations:
            newdataframe = station.getData(reindex = True, From=From, To=To) # Dataframe which stock the new data of the stations
            print "x"*80
            print newdataframe.index
            print "x"*80
            newdataframe['U m/s'] = station.getvar('U m/s')
            newdataframe['V m/s'] = station.getvar('V m/s')
            staname = station.getpara('stanames')
            selections = self.__getpredictors(staname)
            variables = newdataframe.columns
            for i,selection in enumerate(selections):
                print "="*20
                print str(i), ' on ', str(len(selections)), ' completed'
                print "="*20
                for var in variables:
                    Y = station.getvar(var) # variable to be filled
                    X1 = selection[0].getvar(var) # stations variable used to fill
                    X2 = selection[1].getvar(var)# stations variable used to fill

                    try:
                        # get parameters
                        data=pd.concat([Y, X1, X2],keys=['Y','X1','X2'],axis=1, join='outer').dropna()
                        params = self.MLR(data[['X1','X2']], data['Y'], summary = summary)
        
                        # get new fitted data
                        select = pd.concat([X1, X2],keys=['X1','X2'],axis=1, join='inner').dropna()
                        newdata = params[0] + params[1]*select['X1'] + params[2]*select['X2']
        
                        # Place fitted data in original dataframe
                        idxmissing = newdataframe[var][newdataframe[var].isnull() == True].index # slect where their is missing data
                        newdataframe[var][idxmissing] = newdata[idxmissing] # Fill the missing data with the estimated serie
                    except KeyError:
                        print('Data not present in all station')

            speed,dir = cart2pol(newdataframe['U m/s'],newdataframe['V m/s'])
            dir = -dir*(180/np.pi)+180
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
    
    stanames = AttSta.stations(['Head'])
    staPaths = AttSta.getatt(stanames , 'InPath')
    net.AddFilesSta(staPaths)

    From='2014-10-01 00:00:00' 
    To='2015-11-01 00:00:00'
    
    gap = FillGap(net)
    gap.fillstation([], all = True, From=From, To=To)

    gap.WriteDataFrames(OutPath)

#===============================================================================
# Clean Full
#===============================================================================

InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'


Files=glob.glob(InPath+"*")

#     threshold={
#                 'Pa H':{'Min':850,'Max':920},
#                 'Ta C':{'Min':5,'Max':40,'gradient_2min':4},
#                 'Ua %':{'Min':0.0001,'Max':100,'gradient_2min':15},
#                 'Rc mm':{'Min':0,'Max':8},
#                 'Sm m/s':{'Min':0,'Max':30},
#                 'Dm G':{'Min':0,'Max':360},
#                 'Bat mV':{'Min':0.0001,'Max':10000},
#                 'Vs V':{'Min':9,'Max':9.40}}

for f in Files:
    print f
    if f == "/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C15.TXT":
        df = pd.read_csv(f, sep=',', index_col=0,parse_dates=True)
        print df
        df['Ta C'][(df['Ta C']<5) | (df['Ta C']>35) ] = np.nan
        df['Ta C'] = df['Ta C'].fillna(method='pad')
        df['Ua %'][(df['Ua %']<=0) | (df['Ua %']>=100) ] = np.nan
        df['Ua %'] = df['Ua %'].fillna(method='pad')
        df.to_csv(f)
    if f == "/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C10.TXT":
        print "allo"
        df = pd.read_csv(f, sep=',', index_col=0, parse_dates=True)
        df['Ta C'][(df['Ta C']<0) | (df['Ta C']>36) ] = np.nan
        df['Ua %'][(df['Ua %']<0) | (df['Ua %']>100) ] = np.nan
        df['Ua %'] = df['Ua %'].fillna(method='pad')
        df.to_csv(f)
















