#===============================================================================
# DESCRIPTION
#    make a boxplot of the difference between each station and the network mean
#
# It already exist a function to put the stations of a network in columns
#    NOTE
#        A large part of this program should be implemented in the network basics
#===============================================================================
import glob
from LCBnet_lib import *

#===============================================================================
# For the math/tex font be the same than matplotlib
#===============================================================================




class variability():
    """
    DESCRIPTION
        contain method to manipulate and plot the spatial and temporal variability
    """
    def __init__(self, net):
        self.net= net
        self.AttVar = AttVar()
    
    def plot(self,var = ['Ta C'], From=None, To=None, kind = "spatial",save = False, diff = None, acc=None, min_max=None, boxplot= None, outpath = '/home/thomas/', errbar=None):
        """
        DESCRIPTION
             plot the network variability
             
        INPUT
            kind: spatial_min_max: plot a bar plot of the min and max of each stations
            Diff: TRUE, return the difference with the network mean
                  False, return the absolute value of the variable
            Acc: TRUE, group the variable by sum
            min_max:True: return the min and max of the stations
        """
        net = self.net
        attsta = att_sta()

        lcbplot = LCBplot() # get the plot object
        argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
        arglabel = lcbplot.getarg('label')
        argticks = lcbplot.getarg('ticks')
        argfig = lcbplot.getarg('figure')
        arglegend = lcbplot.getarg('legend')
#         plt.rc('text', usetex=True)
        
        data_net = net.getData(var= var, From=From,To=To)
        var_diff = pd.DataFrame(data_net, index=data_net.index)
        var_diff.columns = ['network']

        stations = net.getsta('', all=True, sorted='Alt')
        print stations
        Alt = attsta.getatt(stations['stanames'],'Alt')

        for sta,staname in zip(stations['stations'],stations['stanames']) :
                var_diff[staname] = sta.getData(var=var, From=From,To=To)

        if kind =="spatial_min_max":
            var_diff = var_diff.groupby(lambda t: (t.hour)).mean()
            mean = var_diff.mean().transpose()
            min = var_diff.min().transpose()
            max = var_diff.max().transpose()
            
            min_max = pd.concat([min,max,mean],axis=1 )
            min_max.columns = ['min','max','mean']
            min_max = min_max.transpose()
            del min_max['network']

        if kind == "spatial":
            if boxplot:
                print "boxplot"
                if diff:
                    print "diff"
                    var_diff = var_diff.subtract(var_diff['network'], axis=0)
                elif acc:
                    print "acc"
                    var_diff = var_diff.groupby(lambda t: (t.hour)).sum()
                elif min_max:
                    var_diff = var_diff.groupby(lambda t: (t.hour)).mean()
                    print var_diff
                else:
                    print "mean"
                    del var_diff['network']
                    var_diff = var_diff.groupby(lambda t: (t.hour)).mean()
            else:
                print "not boxplot"
                if acc:
                    print "acc"
                    var_diff = var_diff.sum(axis=0)
                else:
                    print "mean"
                    var_diff = var_diff.mean(axis=0)
                var_diff = var_diff.transpose()
                    
        fig = plt.figure(**argfig)
        
        if kind == "temporal":
            del var_diff['network']
            if acc:
                var_diff = var_diff.groupby(lambda t: (t.hour)).sum()
            else:
                var_diff = var_diff.groupby(lambda t: (t.hour)).mean()
            var_diff['mean'] = var_diff.mean(axis=1)
            if diff:
                print var_diff
                var_diff = var_diff.subtract(var_diff['mean'], axis=0)
                print var_diff
            var_diff = var_diff.transpose()

        if boxplot:
            var_diff.boxplot()
#             plt.ylim(-5,5)
        elif errbar and kind =="spatial_min_max":
            print min_max.loc['mean']
            lower_error = min_max.loc['mean'] - min_max.loc['min']
            upper_error = min_max.loc['max'] - min_max.loc['mean']
            print lower_error
            asymmetric_error = [lower_error, upper_error]
            (_, caps, _) = plt.errorbar(range(len(min_max.columns)), min_max.loc['mean'], yerr=asymmetric_error,linestyle='',
                          marker='o', color='0.15', capsize=20, elinewidth=6, markersize=12)
            for cap in caps:
                cap.set_markeredgewidth(6)
            # ['C17', 'C10', 'C04', 'C18', 'C05', 'C11', 'C16', 'C19', 'C12', 'C06', 'C07', 'C13', 'C08', 'C14', 'C15', 'C09']
#             stations_names_article = ['S14', 'S7', 'S6', 'S15', 'S5','S8','S13', 'S16', 'S9', 'S4', 'S3', 'S10', 'S2', 'S11', 'S12', 'S1']
            plt.xticks( range(len(Alt)), Alt,rotation='vertical')
            if var =='Ev hpa':
                longname = ['Vapor pressure (hpa)']
            else:
                longname = self.AttVar.getatt(var, 'longname_latex')
            plt.ylabel(longname[0], **arglabel)
            plt.grid()
            plt.margins(0.05)
        else:
            var_diff.plot(kind='bar')
        
#         plt.xticks( fontsize = 30)
#         plt.yticks( fontsize = 30)
        plt.tick_params(axis='both', which='major', **argticks)

        if not save:
            plt.show()
        else:
            if kind == "temporal":
                plt.xlabel('Hours (h)', **arglabel)
                plt.savefig(outpath+var[0:2]+"_vartemporal.svg")
            else:
                plt.xlabel('stations', fontsize=30)
                plt.savefig(outpath+var[0:2]+"_varspatil.svg")



if __name__=='__main__':
    Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    OutPath='/home/thomas/'
    Files=glob.glob(Path+"*")
    print Files
    Files.remove('/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C11.TXT')
#     Files.remove('/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C13.TXT')
    
    net=LCB_net()
    net.AddFilesSta(Files)
    
    From = "2014-11-01 00:00:00"
    To = "2016-01-01 00:00:00 "
    
    variability = variability(net)
    
    #===========================================================================
    # Spatial Variability
    #===========================================================================
#     variability.plot(kind ='spatial',var= "Rc mm",From=From, To=To, diff=True, acc=True, boxplot=True)
#      
#     variability.plot(kind ='spatial',var= 'Ta C',From=From, To=To)
      
#     variability.plot(kind ='spatial',var= 'Ua g/kg',From=From, To=To, diff=True, boxplot=True)
#      
#     variability.plot(kind = 'spatial',var= 'Pa H',From=From, To=To, diff=True, boxplot=True)
#      
#     variability.plot(kind ='spatial',var= 'Sm m/s',From=From, To=To, diff=True, boxplot=True)
    
    
    #===========================================================================
    # Temporal Variability
    #===========================================================================
#     variability.plot(kind ='temporal',var= "Rc mm",From=From, To=To, diff=True, acc=True, boxplot=True, save=True)
      
#     variability.plot(kind ='temporal',var= 'Ta C',From=From, To=To, diff=True, boxplot=True, save=True)
#       
#     variability.plot(kind ='temporal',var= 'Ua g/kg',From=From, To=To, diff=True, boxplot=True, save=True)
#       
#     variability.plot(kind = 'temporal',var= 'Pa H',From=From, To=To, diff=True, boxplot=True, save=True)
#      
#     variability.plot(kind ='temporal',var= 'Sm m/s',From=From, To=To, diff=True, boxplot=True, save=True)

#===============================================================================
# Spatial variability Tmin Tmax
#===============================================================================
    variability.plot(kind ='spatial_min_max',var= 'Ta C',From=From, To=To, errbar=True, save=True)
    variability.plot(kind ='spatial_min_max',var= 'Ev hpa',From=From, To=To, errbar=True, save=True)
    
    
    
    
    
    
