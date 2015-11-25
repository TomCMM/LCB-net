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


class variability():
    """
    DESCRIPTION
        contain method to manipulate and plot the spatial and temporal variability
    """
    def __init__(self, net):
        self.net= net
    
    def plot(self,var = ['Ta C'], From=None, To=None, kind = "spatial",save = False, diff = True, acc=False, boxplot= True, outpath = '/home/thomas/'):
        """
        DESCRIPTION
             plot the network variability
             
        INPUT
            Diff: TRUE, return the difference with the network mean
                  False, return the absolute value of the variable
            Acc: TRUE, group the variable by sum
        """
        net = self.net
        
        
        data_net = net.getData(var= var, From=From,To=To)
        var_diff = pd.DataFrame(data_net, index=data_net.index)
        var_diff.columns = ['network']

        stations = net.getsta('', all=True, sorted='Altitude')
        for sta,staname in zip(stations['stations'],stations['stanames']) :
                var_diff[staname] = sta.getData(var=var, From=From,To=To)

        if kind == "spatial":
            if boxplot:
                print "boxplot"
                if diff:
                    print "diff"
                    var_diff = var_diff.subtract(var_diff['network'], axis=0)
                if acc:
                    print "acc"
                    var_diff = var_diff.groupby(lambda t: (t.hour)).sum()
                else:
                    print "mean"
                    del var_diff['network']
#                     var_diff = var_diff.groupby(lambda t: (t.hour)).mean()
            else:
                print "not boxplot"
                if acc:
                    print "acc"
                    var_diff = var_diff.sum(axis=0)
                else:
                    print "mean"
                    var_diff = var_diff.mean(axis=0)
                var_diff = var_diff.transpose()

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
#             
#         print var_diff
#         print var_diff['mean']
        if boxplot:
            var_diff.boxplot()
            plt.ylim(-5,5)
        else:
            print "allo"
            var_diff.plot(kind='bar')
        
        if not save:
            plt.show()
        else:
            if kind == "temporal":
                plt.savefig(outpath+var[0:2]+"_vartemporal.png")
            else:
                plt.savefig(outpath+var[0:2]+"_varspatil.png")
            plt.close()


if __name__=='__main__':
    Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    OutPath='/home/thomas/MergeDataThreeshold/'
    Files=glob.glob(Path+"*")
    net=LCB_net()
    net.AddFilesSta(Files)
    
    From = "2014-10-01 00:00:00"
    To = "2015-10-01 00:00:00 "
    
    variability = variability(net)
    variability.plot(kind ='temporal',var= "Rc mm",From=From, To=To, diff=True, acc=True, boxplot=True, save=True)
    
    variability.plot(kind ='temporal',var= 'Ta C',From=From, To=To, diff=True, boxplot=True, save=True)
    
    variability.plot(kind ='temporal',var= 'Ua g/kg',From=From, To=To, diff=True, boxplot=True, save=True)
    
    variability.plot(kind ='temporal',var= 'Pa H',From=From, To=To, diff=True, boxplot=True, save=True)
    
    variability.plot(kind ='temporal',var= 'Sm m/s',From=From, To=To, diff=True, boxplot=True, save=True)

