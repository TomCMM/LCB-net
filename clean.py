from __future__ import division
import os
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch
import copy
from LCBnet_lib import *
from scipy import interpolate
import seaborn as sns
from pandas.core.frame import DataFrame






class ManageDataLCB(object):
    """
    DESCRIPTION
        1) Read data LCB observations, make the file readable by pandas and write a new file
        2) Clean the file and Merge them into a new dataframe
    """

    def __init__(self,InPath,fname,staname=None):
        """Take The Path and the filname of the raw files folder 
        
        NOTE:
            Threshold:
                gradient_2min:
                    by looking at the histogramm of the 2min gradient of temperature
                    the valor never exceed 3.5C in 2 minutes which correspond to a rain 
                    event or cold front 
        """
        self.InPath=InPath
        self.fname=fname
        self.staname = staname
        self.Header_info = 'Modulo XBee Unknown  Adress SL: Unknown\r\n'
        self.NbLineHeader = 2
        self.threshold={
                        'Pa H':{'Min':850,'Max':920},
                        'Ta C':{'Min':5,'Max':40,'gradient_2min':4},#
                        'Ua %':{'Min':0.0001,'Max':100,'gradient_2min':15},
                        'Rc mm':{'Min':0,'Max':8},
                        'Sm m/s':{'Min':0,'Max':30},
                        'Dm G':{'Min':0,'Max':360},
                        'Bat mV':{'Min':0.0001,'Max':10000},
                        'Vs V':{'Min':9,'Max':9.40}#'Vs V':{'Min':8.5,'Max':9.5}

                         
                         
                         
                        }
        self.__read(InPath,fname)
        self.Header()
        self.clear()

    def Header(self,type=None):
        """
        DESCRIPTION
            Return the header in function of the type of the file
        NOTE
            The file header changed sevral times, especially at the begining when the 
            file pattern was not yet defined by Nilson
        """
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
        """
        DESCRIPTION
            Return the number of column in the header of the file opened
        """
        nbcol=len(self.Header_var.split(','))
        return nbcol

    def __read(self, InPath, fname):
        """
        DESCRIPTION
            Read the raw file
            check if the data can be read in a dataframe
            if not it has to be clean, see the function clear
        """
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
        """
        DESCRIPTION
            This methods delete all the line which does not fit some criteria
            At the end it permits to create a data frame readeable by pandas
        """
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
        """
        DESCRIPTION
            write the dataframe wich has been cleared
        """
        self.clear()
        fname = fname.upper()
        f = open(OutPath+fname+'clear',"w")
        for line in self.content_clear:
            f.write(line)
        f.close

    def append_dataframe(self,fileobject):
        """
        DESCRIPTION
            Append the dataframe together
        USERINPUT
             A list of file path with the different files to merge
            exemple: H05XXX240 will be merged with H05XXX245
        """
        try:
            self.Dataframe=self.Dataframe.append(fileobject.Dataframe).sort_index(axis=0)
            print("Merging dataframe "+fileobject.fname)
        except:
            print('It cant merge dataframe')

    def write_dataframe(self,OutPath, fname):
        """
        DESCRIPTION
            write the dataframe
        """
        self.Dataframe.to_csv(OutPath+fname)
        print('--------------------')
        print('Writing dataframe')
        print('--------------------')

    def clean_dataframe(self, threshold = True, specific = True, reindex = True, gradient = True):
        """
        DESCRIPTION
            Apply filter to the dataframe
        """
        dataframe = self.Dataframe
        

        self.old_dataframe = self.Dataframe.copy()
        print('0'*80)
        dataframe = dataframe.drop_duplicates()


        if specific:
            dataframe = self._specific_clean(dataframe)

        if threshold:
            print'Apply threeshold filter'
            dataframe = self._threshold(dataframe)


        dataframe = dataframe.convert_objects(convert_numeric=True) # convert the data frame into numeric if not put nan
        # remove null colomns (e.g. Unnamed)
        dataframe = dataframe.dropna(axis = 1, how ='all')


        
        if reindex:
            dataframe = self._reindex(dataframe)


        if gradient:
            dataframe = self._grad_threshold(dataframe,'Ta C')
            dataframe = self._grad_threshold(dataframe,'Ua %')

        
        self.Dataframe = dataframe
        print('0'*80)

    def _reindex(self,dataframe):
        """
        Reindex with a 2min created index
        
        NOTE
            The nan are not represented in the graphic 
            compared to the missing value wich are linearly interpolated on a plot
            So after reindexing the time serie plot appears differently and this is normal
            
            If during a period their is some sparse data, for example 4 or 5 data by day.
            without nan the serie will look like their is data but it is not the case.
        """
        Initime = dataframe.index[0]
        Endtime = dataframe.index[-1]
        Initime=pd.to_datetime(Initime)# be sure that the dateset is a Timestamp
        Endtime=pd.to_datetime(Endtime)

        # remove all nan to be sure that the reindex does not keep bad values
        dataframe = dataframe.dropna(axis = 0, how ='all') 

        dataframe = dataframe.drop_duplicates() # I do not know which this one does not work properly
        dataframe = dataframe.groupby(dataframe.index).first()# take the first index encontered. A mthod to drop duuplicate
        idx=pd.date_range(Initime,Endtime,freq='2min')
        dataframe = dataframe.reindex(index=idx)
        return dataframe

    def comparison_clean(self,vars = None, just_clean = False, subplot = False, outpath='/home/thomas/',staname = None, From=None, To=None):
        """
        DESCRIPTION
            Plot time serie before and after the data cleaned
        INPUT
            just_clean: True, plot only the cleaned data
        """
        old_dataframe = self.old_dataframe 
        dataframe = self.Dataframe
        
        if From:
            old_dataframe = old_dataframe[From:To]
            dataframe = dataframe[From:To]
        
        
        if vars == None:
            vars = dataframe.columns

        if subplot:
            f, axarr = plt.subplots(len(vars), sharex=True,figsize=(1920/96, 1080/96), dpi=96)
            if len(vars) ==1:
                axarr = [axarr]
            for ax,var in zip(axarr, vars):
                if not just_clean:
                    ax.plot(old_dataframe.index, old_dataframe[var], color='red')
                    ax.plot(dataframe.index, dataframe[var], color='blue')
                elif just_clean:
                    ax.plot(dataframe.index, dataframe[var], color='blue')
    
            plt.savefig(outpath+"_"+var[0:2]+"_"+staname, dpi=96)
        else:
            for var in vars:
                try:
                    plt.figure(figsize=(1920/96, 1080/96), dpi=96)
                    if not just_clean:
                        plt.plot(old_dataframe.index, old_dataframe[var], color='red')
                        plt.plot(dataframe.index, dataframe[var], color='blue')
                    elif just_clean:
                        plt.plot(dataframe.index, dataframe[var], color='blue')
                    plt.savefig(outpath+"_"+var[0:2]+"_"+staname, dpi=96)
                except ValueError:
                    print "Could not plot surely a problem with the variable: ", str(var), " in the uncleaned dataframe"

    def _threshold(self,dataframe):
        """
        DESCRIPTION
            Remove all the data aboce the threesholds
        RETURN 
            New clean dataframe
        NOTE
            pour les stations C05, C08 et C09
            il y a des problemes de PA et Ua qui enleve des bonne valeurs de TA et Sm 
            donc pour ces 3 stations, seulement la valeur de la variables est drop est non pas 
            toute la ligne
            
            Pour le cross specific il faut que la variable qui permet
             de filtrer ne soit pas deja filtrer elle meme
        """
        print "-> Threshold"
        threshold = self.threshold
        newdataframe = dataframe.copy()
        staname = self.staname

        for var in newdataframe.columns:
            print var
            if (staname == "C05" and var == "Rc mm")\
            or (staname == "C05" and var == "Ua %"):
                newdataframe = self._specific_threshold(newdataframe, var, var2 = 'Pa H')
                newdataframe = self._specific_threshold(newdataframe, var, var2 = 'Ua %')
                newdataframe = self._specific_threshold(newdataframe, var, var2 = 'Rc mm')
    
            if (staname == "C04" and var == "Rc mm"):
                newdataframe = self._specific_threshold(newdataframe, var, var2 = 'Ua %')

            if (staname == "C15" and var == "Ta C"):
                threshold = {"Vs V":{'Min':9.1,'Max':9.40}}
                newdataframe = self._specific_threshold(newdataframe, var, var2 = 'Vs V', threshold=threshold)

            if (staname == "C05" and var == "Pa H")\
                   or (staname == "C05" and var == "Ua %")\
                   or (staname == "C05" and var == "Rc mm")\
                   or (staname == "C05" and var == "Vs V")\
                   or (staname == "C08" and var == "Ua %")\
                   or (staname == "C08" and var == "Pa H")\
                   or (staname == "C09" and var == "Ua %")\
                   or (staname == "C04" and var == "Ua %")\
                   or (staname == "C04" and var == "Pa H")\
                   or (staname == "C18" and var == "Pa H")\
                   or (staname == "C18" and var == "Ua %")\
                   or (staname == "C18" and var == "Rc mm")\
                   or (staname == "C09" and var == "Pa H"):
                print "specific threshold"
                
                newdataframe = self._specific_threshold(newdataframe, var)
            else:
                try:
                    index = newdataframe[(newdataframe[var]<threshold[var]['Min']) | (newdataframe[var]>threshold[var]['Max']) ].index
                    newdataframe = newdataframe.drop(index)
                except KeyError:
                    print('no threshold for '+var)




        return newdataframe 

    def _specific_threshold(self,df,var, var2 = None, threshold=None):
        """
        DESCRIPTION
            do a specific threshold which will only remove a variable and
            not the entire column
        INPUT
            var2 = True, use var1 to clean var2
            Threshold= None, use the by default threeshold
        """
        print "-> Specific threshold"
        
        if not threshold:
            threshold = self.threshold
        
        if not var2:
            print "allo"
            df[var][(df[var]<threshold[var]['Min']) | (df[var]>threshold[var]['Max']) ] = np.nan
        else:
            print "cross variable filtering"
            df[var][(df[var2]<threshold[var2]['Min']) | (df[var2]>threshold[var2]['Max']) ] = np.nan
        return df

    def _grad_threshold(self,df, var):
        """
        DESCRIPTION 
            Remove value exceding a threshold gradient
        """
        print "-> Gradient"
        threshold = self.threshold
        gradient = df[var].diff().abs()
        df[var][gradient > threshold[var]['gradient_2min']] = np.nan
        return df

    def _movingaverage(self,df):
        """
        DESCRIPTION
            Remove all the data above a threeshold after moving average
        RETURN 
            New clean dataframe
        NOTE 
            When the difference between running mean x_window and the data is superior at MavgX then remove the data
        IMPORTANT
            Not implemented yet
        """
        tr = self.threshold
        
        s_window = 5
        m_window = 30
        l_window = 180
        for var in df.columns:
            try:
                df = df[(np.abs(pd.rolling_mean(df[var],s_window) - df[var])<tr[var]['MavgS'])]# Small window
                df = df[(np.abs(pd.rolling_mean(df[var],m_window) - df[var])<tr[var]['MavgM'])]# Medium window
                df = df[(np.abs(pd.rolling_mean(df[var],l_window) - df[var])<tr[var]['MavgL'])]# Large window

                print('The running mean filter on',[var],' as removed  |---> [',
                      len(df[(np.abs(pd.rolling_mean(df[var],s_window)-df[var])>tr[var]['MavgS'])]),
                      ' and ',
                      len(df[(np.abs(pd.rolling_mean(df[var],m_window)-df[var])>tr[var]['MavgM'])]),
                      'and',
                      len(df[(np.abs(pd.rolling_mean(df[var],l_window)-df[var])>tr[var]['MavgL'])])
                      ,'] data')
            except KeyError:
                print('no Mavg threshold for this variable:->  ',var)

    def _var_blocked(self, df, var, var2= None):
        """
        DESCRIPTION
            Check if a variable is blocked in time
            - Resample by hours and check if the variable is exactly the same on the next hour
            which is impossible to be exactly the same with the same degree of accuracy
        
        INPUT
            var: string, variable to check the stationnarity
            var2: other variable to remove based on var (for example sm/ms when Dm G is stationnary)
        """
        print "removing blocked wind"
        df['diff_var'] = df.loc[:,var].resample('1H',how='sum').diff().asfreq('2Min', method='pad')

        df[var][df['diff_var'] == 0 ] = np.nan
        df[var2][df['diff_var'] == 0 ] = np.nan

        return df 
        False

    def _specific_clean(self,df):
        """
        DESCRIPTION
            Do some specific cleaning on: 
            - specific stations
            - specific period
        NOTE
            - Correct rain accumulation into rain intensity
            - Correct time delay
        """
        if self.staname == "C06":
            print df[:"2014-09-04 13:36:00"]['Rc mm']
            df[:"2014-09-04 13:36:00"]['Rc mm'] = df[:"2014-09-04 13:36:00"]['Rc mm'].diff(periods=1).abs()
            print df[:"2014-09-04 13:36:00"]['Rc mm'].diff(periods=1).abs()
            print df[:"2014-09-04 13:36:00"]['Rc mm']
            
        if self.staname == "C05":
            df[:"2014-10-02 17:10:00"]['Rc mm'] = df[:"2014-10-02 17:10:00"]['Rc mm'].diff(periods=1).abs()

        if self.staname == "C04":
            df[:"2014-10-02 17:44:00"]['Rc mm'] = df[:"2014-10-02 17:44:00"]['Rc mm'].diff(periods=1).abs()

        if self.staname == "C08":
            df[:"2014-09-04 12:32:00"]['Rc mm'] = df[:"2014-09-04 12:32:00"]['Rc mm'].diff(periods=1).abs()
            df["2014-09-17 14:02:00":"2014-10-09 16:10:00"]['Rc mm']  = np.nan

        if self.staname == "C09":
            df[:"2014-09-04 10:36:00"]['Rc mm'] = df[:"2014-09-04 10:36:00"]['Rc mm'].diff(periods=1).abs()
            df["2014-09-17 13:32:00":"2014-10-09 13:56:00"]['Rc mm'] = np.nan
            df["2014-09-02 20:52:00":"2014-09-05 10:32:00"]['Rc mm'] = np.nan
        if self.staname == "C18":
            df[:"2015-05-09 13:10:00"].index = df[:"2015-05-09 13:10:00"].index + pd.DateOffset(hours = 3)

        if self.staname == "C08":
            df = self._var_blocked(df, 'Dm G', 'Sm m/s')
            df = self._var_blocked(df, 'Sm m/s', 'Dm G')
            
        if self.staname == "C04":
            df = self._var_blocked(df, 'Dm G', 'Sm m/s')
            df = self._var_blocked(df, 'Sm m/s', 'Dm G')

        if self.staname == "C06":
            df= df.drop(df["2014-09-19 00:00:00":"2014-09-21 00:00:00"].index)

        if self.staname == "C05":
            df['Ua %'][:"2014-09-17 14:52:00"]= np.nan
        return df

#===============================================================================
# Bulk Clean Data
#===============================================================================
#====== User Input

if __name__=='__main__':
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data/Extrema20151116/Extrema20151116'
#     OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data/Extrema20151116/Clean/'
#     #====== Find all the clima and Hydro
#     Files=glob.glob(InPath+"/*/*")
#     print(Files)
#     if not os.path.exists(OutPath):
#         os.makedirs(OutPath)
#     for i in Files:
#         data=ManageDataLCB(os.path.dirname(i)+"/",os.path.basename(i))
#         print("Writing file "+OutPath+os.path.basename(i))
#         data.write_clean(OutPath,os.path.basename(i))

#===============================================================================
#  Merge and Filter - Bulk cleanED data
#===============================================================================
#===============================================================================
# Merge the clean files
#===============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/data'
#     OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
# #     OutPath='/home/thomas/'
# #     
# #     From='2015-05-01 00:00:00' 
# #     To='2015-08-01 00:00:00'
#     
#     if not os.path.exists(OutPath):
#         os.makedirs(OutPath)
#       
#     AttSta=att_sta()
#     stations=AttSta.stations(['Ribeirao'])
# #     stations = ['C18']
#     for sta in stations:
#         # Permit to find all the find with the extension .TXTclear
#         print(sta)
#         matches = []
#         datamerge=None
#         for root, dirnames, filenames in os.walk(InPath):# find all the cleared files
#             for filename in fnmatch.filter(filenames, sta+'*.TXTclear'):
#                 matches.append(os.path.join(root, filename))    
#         for i in matches:
#             print(i)
#             data=ManageDataLCB(os.path.dirname(i)+"/",os.path.basename(i),sta)
#             try:
#                 datamerge.append_dataframe(data)
#             except:
#                 datamerge=data
#         datamerge.clean_dataframe()
#         datamerge.comparison_clean(vars=['Ta C', 'Vs V', 'Ua %', 'Pa H', 'Rc mm', 'Sm m/s', 'Bat mV'], staname=sta)
#         datamerge.write_dataframe(OutPath,sta+'clear_merge.TXT')




    
    

