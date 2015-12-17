#===============================================================================
# DESCRIPTION
#    Group the stations in different group and plot 
#    their difference of a specified variables.
#===============================================================================

import glob
from LCBnet_lib import *
import matplotlib

class Gradient():
    """
    DESCRIPTION
        Calculate the gradient between two group of stations
    INPUT
        list of a list of the couple name 
        e.g.: [[West, East],[Valley, Slope], [Mouth, Head]]
    RETURN
        A plot or the value of the gradient
        
    EXAMPLE
        dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
        AttSta = att_sta()
        AttSta.setInPaths(dirInPath)

        grad = Gradient([['West', 'East'],['valley','slope'], ['Medio', 'Head']])
        grad.grad(var=['Theta C'], by = "H", From ='2014-10-15 00:00:00', To = '2015-07-01 00:00:00' )
        grad.tsplot(zero=True)
        plt.show()
    
        plt.close()
    """
    def __init__(self, dirInPath):
        self.AttSta = att_sta()
        self.AttSta.setInPaths(dirInPath)

    def couples_net(self, couples_name):
        """
        Description
            Return couple of network
        The third argument is applied for the selection of the both networks
        """
        couples_net = {}
        new_couples_name = []
        for couple_name in couples_name:

            if len(couple_name) == 3:
                print('Argument passed ')
                group1 = self.AttSta.stations([couple_name[0], couple_name[2]])
                group2 = self.AttSta.stations([couple_name[1], couple_name[2]])
                couplename = couple_name[0]+"_"+couple_name[1]+"_"+couple_name[2]

            else:
                print "No argument passed"
                group1 = self.AttSta.stations([couple_name[0]])
                group2 = self.AttSta.stations([couple_name[1]])
                couplename = couple_name[0]+"_"+couple_name[1]

            Files1 = self.AttSta.getatt(group1, 'InPath')
            Files2 = self.AttSta.getatt(group2, 'InPath')

            net1 = LCB_net()
            net2 = LCB_net()
            net1.AddFilesSta(Files1)
            net2.AddFilesSta(Files2)

            couples_net[couplename] = [net1,net2]
            new_couples_name.append(couplename)

        self.couples_name = new_couples_name
        self.couples_net = couples_net

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
        NOTE
            I am doing way better nowadays but it is working :)
        """
        newdf=pd.DataFrame()
        column = serie.columns
        serie.index = serie.index.hour
        serie.columns = column
        for col in range(1,24):
            subdata=serie[serie.index == col]
            subserie = pd.DataFrame(np.array(subdata),index=range(len(subdata.index)),columns=[col])
            newdf = newdf.join(subserie,how='outer')
        return newdf

    def grad(self, rainfilter = False, var = 'Ta C', by= None, From = None, To = None, group=None, how=None, return_=None):
        """
        DESCRITPION
            give the difference between a station and another
        INPUT
            stanames1: stations names of the first network
            stanames2: stations names of the second network
        """
        
        couples_net = self.couples_net
        couples_name = self.couples_name
        couples_grad = {}

        if not isinstance(From, list):
            From = [From]
        if not isinstance(To, list):
            To = [To]

        new_couples_name = []
        for from_ , to_ in zip(From, To):
            for couple_name in couples_name:
                new_couple_name =couple_name+str(from_)
                print couples_net
                net1 = couples_net[couple_name][0]
                net2 = couples_net[couple_name][1]
    
                if not From:
                    From = net1.getpara('From')
                if not To:
                    To = net1.getpara('To')
                new_couples_name.append(new_couple_name)
                couples_grad[new_couple_name] = net1.getData(var=var, From= from_, To=to_, by= by, how=how, group=group, rainfilter=rainfilter) - net2.getData(var=var, From= from_, To=to_, by= by, how=how, group=group, rainfilter=rainfilter)
    
        self.new_couples_name = new_couples_name
        if return_:
            return couples_grad
        else:
            self.couples_grad = couples_grad

    def tsplot(self, zero=None, grey=None, outpath=None, quartile=True):
        """
        DESCRIPTION
            make a time serie plot of the gradient of temperature given by the couples
        INPUT
            Need to run the methods grad before to launch this one
        """
        try:
            couples_grad = self.couples_grad
        except AttributeError:
            print("Need to run the method grad before to run this one")

        fig, ax = plt.subplots()
        if grey == True:
            colors=list()
            for i in np.arange(1,0,-0.2):
                colors.append(plt.cm.Greys(i))
        else:
            colors = ['b', 'g', 'r','b', 'g', 'r']
        linestyles = ['-', '-', '-','--', '--', '--']
        for couple, c, l in zip(self.new_couples_name, colors, linestyles):
            print couples_grad
            serie = couples_grad[couple]
            name = couple
            df = self.ClassPeriod(serie)
            median = df.quantile(q=0.5,axis=0)
            if quartile:
                quartile1 = df.quantile(q=0.25,axis=0)
                quartile3 = df.quantile(q=0.75,axis=0)
                ax.fill_between(quartile1.index.values, quartile1.values, quartile3.values, alpha=0.3,color=c)

            ax.plot(median.index.values, median.values, linestyle=l, color=c, alpha=0.8, label=name)
        legend = ax.legend(loc='upper left', shadow=True)
        if zero:
            plt.axhline(0,color='black',alpha=0.2)
        if outpath:
            print "PLOTTED"
            plt.savefig(outpath+ str(serie.columns[0][0:2])+ "_gradient.png")



if __name__=='__main__':
    dir_inpath = '/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    outpath = '/home/thomas/Z_article/'

#===============================================================================
# Quartiles
#===============================================================================
    grad = Gradient(dir_inpath)
#     grad.couples_net([['West', 'East'],['valley','slope'],['Medio', 'Head', 'valley']])
#     grad.grad(var=['Ta C'], by = "H", From ='2014-10-15 00:00:00', To = '2015-07-01 00:00:00')
#     grad.tsplot(zero=True, outpath=outpath)
# # 
#     grad.grad(var=['Ua g/kg'], by = "H", From ='2014-10-15 00:00:00', To = '2015-07-01 00:00:00' )
#     grad.tsplot(zero=True, outpath=outpath)
#   
#     grad.grad(var=['Sm m/s'], by = "H", From ='2014-10-15 00:00:00', To = '2015-07-01 00:00:00' )
#     grad.tsplot(zero=True, outpath=outpath)
# #  
#     grad.grad(var=['Theta C'], by = "H", From ='2014-10-15 00:00:00', To = '2015-07-01 00:00:00' )
#     grad.tsplot(zero=True, outpath=outpath)

#===============================================================================
# Difference Summer Winter
#===============================================================================
    grad.couples_net([['West', 'East','slope'],['valley','slope'],['Medio', 'Head', 'valley']])

    grad.grad(var=['Ta C'], by = "H", From =['2014-11-01 00:00:00','2015-05-01 00:00:00'], To = ['2015-05-01 00:00:00','2015-10-01 00:00:00'])
    grad.tsplot(zero=True, outpath=outpath, quartile=False)
   
    grad.grad(var=['Ua g/kg'], by = "H", From =['2014-11-01 00:00:00','2015-05-01 00:00:00'], To = ['2015-05-01 00:00:00','2015-10-01 00:00:00'])
    grad.tsplot(zero=True, outpath=outpath, quartile=False)
  
    grad.grad(var=['Sm m/s'], by = "H", From =['2014-11-01 00:00:00','2015-05-01 00:00:00'], To = ['2015-05-01 00:00:00','2015-10-01 00:00:00'])
    grad.tsplot(zero=True, outpath=outpath, quartile=False)
     
    grad.grad(var=['Theta C'], by = "H", From =['2014-11-01 00:00:00','2015-05-01 00:00:00'], To = ['2015-05-01 00:00:00','2015-10-01 00:00:00'])
    grad.tsplot(zero=True, outpath=outpath, quartile=False)










