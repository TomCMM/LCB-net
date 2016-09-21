#==============================================================================
#   Description
#       module which fill the missing data of the station observations through multiregression analysis
#    Update
#        August 2016:
#            
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
    DESCRIPTION
        Bootstrap data from a station network
    INPUT
        a network object
    """
    def __init__(self,network):
        self.network = network
        self.newdataframes = { } # contain the new dataframes

    def fillstation(self, stanames, all=None, plot = None, summary = None, From=None, To=None, by=None, 
                    how='mean', variables = None, distance=None, sort_cor = True, constant =True, cor_lim=None):
        """
        DESCRIPTION
            Check every variable of every stations and try to fill 
            them with the variables of the two nearest station for every time.
        INPUT
            From: From where to select the data
            To: when is the ending
            by: resample the data with the "by" time resolution
            sort_cor, if True sort the selected predictors stations by correlation coefficient
            plot: Plot a comparison of the old data and the new filled data
            variables: The variables of the dataframe to be filled
            summary: print the summary of the multilinear regression
            distance: It use the longitude to determine the nearest stations
                    A dataframe from Attsta dist_matrix containing the distance between each stations can be used
            cor_lim: Default, none. if Int given it will be used as a threshold to select the stations based on their correlation coefficient
        OLD CODE:
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
                    except ValueError:
                        print('The variable '+var+ "Does not exist to do the multilinear regression ")
 
                        # get new fitted data
                        select = pd.concat([X1, X2],keys=['X1','X2'],axis=1, join='inner').dropna()
                        newdata = params[0] + params[1]*select['X1'] + params[2]*select['X2']
         
                        # Place fitted data in original dataframe
                        idxmissing = newdataframe[var][newdataframe[var].isnull() == True].index # slect where their is missing data
                        newdataframe[var][idxmissing] = newdata[idxmissing] # Fill the missing data with the estimated serie
                    except KeyError:
                        print('Data not present in all station')
                    except ValueError:
                        print('The variable '+var+ "Does not exist to do the multilinear regression ")
        """
        
        if all == True:
            stations = self.network.getsta([], all=True).values()
        else:
            stations = self.network.getsta(stanames)

        for station in stations:
            staname = station.getpara('stanames')
            
            if variables ==None:
                newdataframe = station.getData(reindex = True, From=From, To=To, by=by, how=how) # Dataframe which stock the new data of the stations
                newdataframe['U m/s'] = station.getData('U m/s', reindex = True, From=From, To=To, by=by, how=how)
                newdataframe['V m/s'] = station.getData('V m/s',reindex = True, From=From, To=To, by=by, how=how)
                newdataframe['Ua g/kg'] = station.getData('Ua g/kg',reindex = True, From=From, To=To, by=by, how=how)
                newdataframe['Theta C'] = station.getData('Theta C',reindex = True, From=From, To=To, by=by, how=how)
            else:
                newdataframe = station.getData(var=variables, reindex = True, From=From, To=To, by=by, how=how) # Dataframe which stock the new data of the stations

            # select and sort nearest stations
            selections, selectionsnames = self.__getpredictors_distance(staname, distance)

            if not variables:
                variables = newdataframe.columns
            else:
                variables = variables

            for var in variables:
                print "I"*30
                print "variable -> " + var

                try:
                    selections, params = self.__sort_predictors_by_corr(station,selections, var, From,To, by,how, constant=constant,
                                                                             selectionsnames=selectionsnames, sort_cor=sort_cor, cor_lim=cor_lim)
    
                        
                    selections_iter = iter(selections)
                    params_iter = iter(params)
    #                 print newdataframe
                    idxmissing = newdataframe[var][newdataframe[var].isnull() == True].index # slect where their is missing data
                    
                    while len(idxmissing) > 0:
                        print ("Their is  [" + str(len(idxmissing)) + "] events missing")
                        
                        try: # Try if their is still other stations to fill with
                            selection = selections_iter.next()
                            param = params_iter.next()
                        except StopIteration:
                            print "NO MORE SELECTED STATIONS"
                            break
                        
                        try:
                            Y = station.getData(var, From=From,To=To,by=by, how=how) # variable to be filled
                            X1 = selection[0].getData(var, From=From,To=To,by=by, how=how) # stations variable used to fill
                            X2 = selection[1].getData(var, From=From,To=To,by=by, how=how)# stations variable used to fill
                            
                            select = pd.concat([X1, X2],keys=['X1','X2'],axis=1, join='inner').dropna()
                            
    
                            if constant:
                                newdata = param[0] + param[1]*select['X1'] + param[2]*select['X2'] # reconstruct the data
                            else:
                                newdata = param[0]*select['X1'] + param[1]*select['X2'] # reconstruct the data
    
                            newdataframe.loc[idxmissing,var] = newdata.loc[idxmissing, var]
                            idxmissing = newdataframe[var][newdataframe[var].isnull() == True].index # slect where their is missing data
    
    
                        except KeyError:
                            print "&"*60
                            print('Selected stations did not fill any events')
                except ValueError:
                    print('The variable '+var+ "Does not exist to do the multilinear regression ")
                    
                    if plot == True:
                        df = pd.concat([Y,X1,X2,newdata,newdataframe[var]], keys=['Y','X1','X2','estimated data','Estimated replaced'],axis=1, join='outer')
                        self.plotcomparison(df)
    
            print ("Their is  [" + str(len(idxmissing)) + "] FINALLY events missing")
            # Recalculate the wind direction and speed from the U an V components
            if newdataframe.columns.isin(['U m/s', 'V m/s']):
                speed,dir = cart2pol(newdataframe['U m/s'],newdataframe['V m/s'])
                newdataframe['Dm G'] = dir
                newdataframe['Sm m/s'] = speed

            self.newdataframes[staname] = newdataframe

    def WriteDataFrames(self, Outpath):
        """
        DESCRIPTION
            Write the bootstraped in a file
        INPUT:
            Outpath, path of the output directory
        """
        
        newdataframes = self.newdataframes
        for staname in newdataframes.keys():
            fname = staname + '.TXT'
            newdataframes[staname].to_csv(Outpath+fname, float_format="%.2f")
            print('--------------------')
            print('Writing dataframe')
            print('--------------------')

    def __getpredictors_distance(self,staname, distance):
        """
        Get preditors base on their distance
        The predictors are selected as following
            [1,2], [1,3], [1,4], [2,3], [2,4], [2,5], [2,6]
        
        """

        stanames = distance[staname]
        del stanames[staname] # remove the station to be fill from the dataframe
        stanames= stanames.sort_values()
        stations = self.network.getsta(stanames.index.values)
        station = self.network.getsta(staname)
        
        # Only 3 closest stations
#         sel1 = [ (i,e) for i,e in zip(stations[0:2], stations[1:3])] # selction predictors with spacing 1
#         sel2 = [ (i,e) for i,e in zip(stations[0:2], stations[2:4])] # selction predictors with spacing 2

        # Use all stations
        sel1 = [ (i,e) for i,e in zip(stations[0:-1], stations[1:])] # selction predictors with spacing 1
        sel2 = [ (i,e) for i,e in zip(stations[0:-2], stations[2:])] # selction predictors with spacing 2    
        
        
#         sel3 = [ (i,e) for i,e in zip(stations[0:-3], stations[3:])] # selction predictors with spacing 3
#         sel4 = [ (i,e) for i,e in zip(stations[0:-4], stations[4:])] # selction predictors with spacing 4

        # Only 3 closest stations
#         sel1names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:2], stations[1:3])] # selction predictors with spacing 1
#         sel2names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:2], stations[2:4])] # selction predictors with spacing 1

        # using all stations
        sel1names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:-1], stations[1:])] # selction predictors with spacing 1
        sel2names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:-2], stations[2:])] # selction predictors with spacing 1



#         sel3names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:-3], stations[3:])] # selction predictors with spacing 1
#         sel4names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:-4], stations[4:])] # selction predictors with spacing 1

        selection = [x for x in itertools.chain.from_iterable(itertools.izip_longest(sel1,sel2)) if x]
        selectionnames = [x for x in itertools.chain.from_iterable(itertools.izip_longest(sel1names, sel2names)) if x]
        
        return selection, selectionnames
          
    def __sort_predictors_by_corr(self, station, selections, var, From,To,by,how, constant=True, selectionsnames=None, sort_cor=True, cor_lim=None):
        """
        Return a sorted  selections by the correlation rsquared scores
        
        """
        
        scores_corel = pd.DataFrame(index = np.arange(0,len(selections)), columns=['corel', 'selections', 'params', 'selectionname']) #correlation of each selections and variables

        for i, (selection, selectionname) in enumerate(zip(selections, selectionsnames)):
            Y = station.getData(var, From=From,To=To, by=by, how=how) # variable to be filled
            X1 = selection[0].getData(var, From=From,To=To, by=by, how=how) # stations variable used to fill
            X2 = selection[1].getData(var, From=From,To=To, by=by, how=how)# stations variable used to fill
 
            data=pd.concat([Y, X1, X2],keys=['Y','X1','X2'],axis=1, join='outer').dropna()
            
            est = self.__MLR(data[['X1','X2']], data['Y'], constant=constant)
            rsquared = est.rsquared

            scores_corel.loc[i, 'corel'] = rsquared
            scores_corel.loc[i, 'selections'] = selection
            scores_corel.loc[i, 'selectionname'] =selectionname
            
            if constant:
                scores_corel.loc[i, 'params'] = [est.params[0], est.params[1], est.params[2]]
            else:
                scores_corel.loc[i, 'params'] = [est.params[0], est.params[1]]
    
        if sort_cor:
            scores_corel = scores_corel.sort_values('corel',ascending=False )
        
        if cor_lim:
            scores_corel = scores_corel[scores_corel['corel'] > cor_lim]
        
        scores_corel.index = np.arange(0,len(scores_corel.index))
        selections = scores_corel['selections'].values
        params = scores_corel['params'].values
        
        print "u"*30
        print "Correlation coefficient of the multilinear regression"
        print "u"*30
        print scores_corel[['corel', 'selectionname']]
        print "u"*30
        return selections, params

    def __MLR(self,X,Y, summary = None, corel=False, constant=True):
        """
        INPUT
            X: dataframe of predictor
            Y: predictant
            summary; True, print the summary of the linear regression
            corel, if True return the correlation and not the parameters
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
        
        if constant:
            X = sm.add_constant(X)
        else:
            print "No Constant use in the linear regression!"

        est = sm.OLS(Y, X).fit()

        if summary == True:
            print(est.summary())
            print(est.params)

        return est

    def plotcomparison(self,df):
        df.plot(subplots = True)
        df.plot(kind='scatter', x='Y', y='estimated data')
        plt.show()


if __name__=='__main__':

#===============================================================================
# Bootstrap data
#===============================================================================
    InPath ='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
    OutPath = '/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    Files = glob.glob(InPath+"*")
    
    net = LCB_net()
    AttSta = att_sta()
    AttSta.setInPaths(InPath)
    stanames = AttSta.stations(['Ribeirao'])
    distance = AttSta.dist_matrix(stanames)

    staPaths = AttSta.getatt(stanames , 'InPath')
    net.AddFilesSta(staPaths)

    From='2014-10-15 00:00:00' 
    To='2016-08-01 00:00:00'
   
    gap = FillGap(net)
    gap.fillstation([], all = True, From=From, To=To, by='H', how='mean',
                    summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
  
    gap.WriteDataFrames(OutPath)
#     



#==============================================================================
# Specific bootstraping for Rainfall - TEST
#==============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
#     OutPath= '/home/thomas/PhD/obs-lcb/LCBData/obs/Full_nearest/'
#     OutPath2= '/home/thomas/PhD/obs-lcb/LCBData/obs/Full_nearest2/'
#     OutPath3= '/home/thomas/PhD/obs-lcb/LCBData/obs/Full_nearest3/'
# #     OutPath3= '/home/thomas/PhD/obs-lcb/LCBData/obs/Full_Allperiod_medio/'
# #     Files=glob.glob(InPath+"*")
#      
# #     stanames = stanames# +['C12', 'C04']
# #     stanames = ['C10','C11', 'C12']
#      
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(InPath)
#     stanames = AttSta.stations(['Ribeirao'])
# #     stanames = ['C04','C05','C06','C07','C13','C14','C15']
# 
# #     [stanames.remove(x) for x in ['C05','C13'] if x in stanames ]
# #     stanames = stanames +['C12','C10','C05' 'C04']
#      
#      
#     distance = AttSta.dist_matrix(stanames)
#     print distance
#     
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#        
# #     # Middle
#     From='2014-10-15 00:00:00' 
#     To='2016-04-01 00:00:00'
#   
#     #Head
# #     From='2015-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#            
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#  
#     gap.WriteDataFrames(OutPath)
# #     
# # #------------------------------------------------------------------------------ 
# # #     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#       
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#        
# #      
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#                
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#      
#     gap.WriteDataFrames(OutPath2)
# # 
# # #------------------------------------------------------------------------------ 
# #     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2)
#    
#    
# #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2)
#    
# #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#    
# #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
# # #  
# #------------------------------------------------------------------------------ 
# #     # Perform another bootstrapping using the previous bootstraped data as input
# #     net=LCB_net()
# #     AttSta = att_sta()
# #     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
# #     distance = AttSta.dist_matrix(stanames)
# #     staPaths = AttSta.getatt(stanames , 'InPath')
# #     net.AddFilesSta(staPaths)
# #    
# #     # Middle
# # #     From='2015-03-15 00:00:00' 
# # #     To='2016-01-01 00:00:00'
# #     
# #       
# # #     From='2014-11-01 00:00:00' 
# # #     To='2016-01-01 00:00:00'
# #             
# #     gap = FillGap(net)
# #     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
# #                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False, cor_lim=0.90)
# #   
# #     gap.WriteDataFrames(OutPath2) 



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
        if f == "/home/thomas/PhD/obs-lcb/LCBData/obs/Full_new/C15.TXT":
            df = pd.read_csv(f, sep=',', index_col=0,parse_dates=True)
            print df
            df['Ta C'][(df['Ta C']<5) | (df['Ta C']>35) ] = np.nan
            df['Ta C'] = df['Ta C'].fillna(method='pad')
            df['Ua %'][(df['Ua %']<=0) | (df['Ua %']>=100) ] = np.nan
            df['Ua %'] = df['Ua %'].fillna(method='pad')
            df['Ua g/kg'][(df['Ua g/kg']<0) | (df['Ua g/kg']>25) ] = np.nan
            df['Ua g/kg'] = df['Ua g/kg'].fillna(method='pad')
            df.to_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full_new_2/C15.TXT")
        if f == "/home/thomas/PhD/obs-lcb/LCBData/obs/Full_new/C10.TXT":
            print "allo"
            df = pd.read_csv(f, sep=',', index_col=0, parse_dates=True)
            df['Ta C'][(df['Ta C']<0) | (df['Ta C']>36) ] = np.nan
            df['Ua %'][(df['Ua %']<0) | (df['Ua %']>100) ] = np.nan
            df['Ua g/kg'][(df['Ua g/kg']<0) | (df['Ua g/kg']>25) ] = np.nan
            df['Ua g/kg'] = df['Ua g/kg'].fillna(method='pad')
            df['Ua %'] = df['Ua %'].fillna(method='pad')
            df.to_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full_new_2/C10.TXT")


    #===============================================================================
    # Interpolate Pressure C08
    #===============================================================================
    
#     from sklearn import datasets, linear_model
#     import statsmodels.api as sm
#     
#     
#     df9 = pd.read_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT", sep=',', index_col=0,parse_dates=True)
#     df7 = pd.read_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C07.TXT", sep=',', index_col=0,parse_dates=True)
#     df8 = pd.read_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C08.TXT", sep=',', index_col=0,parse_dates=True)
#     
#     dfpa9 = df9['Pa H']
#     dfpa8 = df8['Pa H']
#     dfpa7 = df7['Pa H']
#     
#     dfPa = pd.concat([dfpa9,dfpa7,dfpa8], axis=1, join ='inner')
#     dfPa.columns = ['C09', 'C07', 'C08']
#     
#     # dfPa = dfPa[-50:]
#     # print dfPa
#     
#     elev9 = 1356
#     elev7 = 1186
#     elev8 = np.array([1225])
#          
#     def inter(row):
#         y = np.array([[row['C09']], [row['C07']]])
#         x= np.array([[elev9], [elev7]])
#         x = sm.add_constant(x)
#         regr = sm.OLS(y,x)
#         results = regr.fit()
#         return results.params[1]*elev8 + results.params[0]
#              
#     dfPa['C08_new'] = dfPa.apply(inter, axis=1)['C08']
#     df8 = pd.concat([df8, dfPa['C08_new']], axis=1)
#     df8['Pa H']=df8['C08_new']
#     df8 = df8.drop(['C08_new'],1)
#     df8.to_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C08.TXT")
#          
#     df8[['Pa H', 'C08_new']].plot()
#     plt.show()




