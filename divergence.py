
#===============================================================================
# # Thomas 23 July
#    DESCRIPTION
#         Calculate the divergence between a group of
#    
#    MODULE
#        GEOPY
#            EXAMPLE
#                >>> from geopy.distance import vincenty
#                >>> newport_ri = (41.49008, -71.312796)
#                >>> cleveland_oh = (41.499498, -81.695391)
#                >>> print(vincenty(newport_ri, cleveland_oh).miles)

#===============================================================================

from LCBnet_lib import *


from LCBnet_lib import *

class Divergence():
    """
        DESCRIPTION 
            Calculate the divergence between two network
        INPUT
            arg1: network1 
            arg2: network2 
    """
    def __init__(self,net_West, net_East):
        self.net_West = net_West
        self.net_East = net_East

        self.stanames1 = self.net_West.getpara('stanames')
        self.stanames2 = self.net_East.getpara('stanames')
        self.stanamesall = self.stanames1 + self.stanames2
        self.att_sta = att_sta()
        self.sortednames = self.att_sta.sortsta(self.stanamesall, 'Lat')['stanames']

    def dist(self, total=None):
        """
        DESCRIPTION
            Return a dictionnary with the stations name and their respective distance
        INPUT
            total, True, return the total distance between the stations
        """
        AttSta = att_sta()
        dist = AttSta.dist_sta(self.sortednames, formcouples= True)
        L = self.Lengths(dist)
        L = np.array(L)*1000 # convert in meters 

        Length = dict(zip(self.sortednames,L)) # creer un dictionnaire avec le nom de la station associer a ca longeur L
        if total:
            return L
        else:
            return Length

    def MeanU(self, df,Length):
        """
        DESCRIPTION
            Calculate the mean velocity of a network
        """
        lenghtnet =  [ ]
        for c in df.columns:
            df[c] = df[c]*Length[c]
            lenghtnet.append(Length[c])

        Lnet = np.array([lenghtnet]).sum()
        df['sum'] = df.sum(axis =1,skipna = False)
        df['meanU']= df['sum']/Lnet
        return df

    def Lengths(self, dist):
        """
        DESCRIPTION 
            return the half distance
            imagine 3 points [1,2,3]
            with distance X_12 and X_23 between them
            This module will return 
            L1 = X_12
            L2 = X_12/2 + X_23/2
            l3 = X_23
        """''
        L = [ ]
        # Lenght initial and final
        dist.insert(0,dist[0])
        dist.insert(len(dist),dist[-1])
        for i,e in zip(dist[:-1],dist[1:]):
            length = (i+e)/2
            L.append(length)
        return L


    def div(self, From=None, To=None, From2=None,To2=None):
        """
        DESCRIPTION
            Calculate the divergence between the 2 networks onlong the line
        
        """
#         
#         U_west = self.net_West.getvarallsta(var=['U m/s'], all=True,From=From, To=To)
#         U_east = self.net_East.getvarallsta(var=['U m/s'], all=True,From=From, To=To)
# 
#         V_west = self.net_West.getvarallsta(var=['V m/s'], all=True,From=From, To=To)
#         V_east = self.net_East.getvarallsta(var=['V m/s'], all=True,From=From, To=To)

        norm_west = self.net_West.getvarallsta(var=['Sm m/s'],From=From, To=To)
        theta_west = self.net_West.getvarallsta(var=['Dm G'],From=From, To=To)
        
        norm_east = self.net_East.getvarallsta(var=['Sm m/s'],From=From, To=To)
        theta_east = self.net_East.getvarallsta(var=['Dm G'],From=From, To=To)
        
        new_U_west = pd.DataFrame(index=norm_west.index, columns=norm_west.columns) 
        new_U_east = pd.DataFrame(index=norm_east.index, columns=norm_east.columns) 
    
        for staname in norm_west.columns:
            n_west = norm_west[staname]
            t_west = theta_west[staname]
            U_west, V_west = PolarToCartesian(n_west, t_west, rot=45)
            new_U_west[staname] = U_west

        for staname in norm_east.columns:
            n_east = norm_east[staname]
            t_east = theta_east[staname]
            U_east, V_east = PolarToCartesian(n_east, t_east, rot=45)
            new_U_east[staname] = U_east


        df_west = self.MeanU(new_U_west, self.dist())
        df_east = self.MeanU(new_U_east, self.dist())

        conv = (df_east['meanU'] - df_west['meanU'])/(sum(self.dist(total=True))/2)
        return conv

if __name__ == "__main__":
    dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    outpath="/home/thomas/"
    AttSta = att_sta()
    AttSta.setInPaths(dirInPath)
    
    net_West = LCB_net()
    net_East = LCB_net()
    
    
    files_west = AttSta.getatt(AttSta.stations(['Head','West','valley']),'InPath')
    files_west = files_west + AttSta.getatt(AttSta.stations(['Head','West','slope']),'InPath')
    
    files_east = AttSta.getatt(AttSta.stations(['Head','East','valley']),'InPath')
    files_east = files_east + AttSta.getatt(AttSta.stations(['Head','East','slope']),'InPath')

    
    net_West.AddFilesSta(files_west)
    net_East.AddFilesSta(files_east)

    Div = Divergence(net_West, net_East)
    
    conv_spring = Div.div( From='2014-10-01 00:00:00', To='2015-01-01 00:00:00', From2='2015-10-01 00:00:00', To2= '2016-01-01 00:00:00')
    print "spring"
    print conv_spring.index

    conv_summer = Div.div(From='2014-10-01 00:00:00', To='2015-04-01 00:00:00', From2='2015-10-01 00:00:00', To2= '2016-01-01 00:00:00')
    print "summer"
    print conv_summer.index
    
    conv_winter = Div.div(From='2015-04-01 00:00:00', To='2015-10-01 00:00:00')
    print "winter"
    print conv_winter.index

    lcbplot = LCBplot() # get the plot object
    argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
    arglabel = lcbplot.getarg('label')
    argticks = lcbplot.getarg('ticks')
    argfig = lcbplot.getarg('figure')
    arglegend = lcbplot.getarg('legend')

    fig = plt.figure(**argfig)
    plt.plot(conv_spring.groupby(lambda t: (t.hour)).mean(), c='k', linewidth=6)
    plt.plot(conv_summer.groupby(lambda t: (t.hour)).mean(), c='r', linewidth=6)
    plt.plot(conv_winter.groupby(lambda t: (t.hour)).mean(), c='b', linewidth=6)

    
    plt.plot([0]*len(conv_spring.groupby(lambda t: (t.hour)).mean()), c='k', linewidth=3)
    plt.yticks( **arglabel)
    plt.xticks(range(0,24,6), **arglabel)
    plt.xlim([0,24])
    plt.tick_params(axis='both', which='major', **argticks)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('hours (h)', **arglabel)
    plt.ylabel('Divergence (s-1)', **arglabel)
    plt.grid(True)
    plt.savefig(outpath+"Divergence.svg", transparent=True)



# def MeanU(df,Length):
#     """
#     DESCRIPTION
#         Calculate the mean velocity of a network
#     """
#     lenghtnet =  [ ]
#     for c in df.columns:
#         df[c] = df[c]*Length[c]
#         lenghtnet.append(Length[c])
# 
#     Lnet = np.array([lenghtnet]).sum()
#     df['sum'] = df.sum(axis =1,skipna = False)
#     df['meanU']= df['sum']/Lnet
#     return df
# 
# def Lengths(dist):
#     """
#     DESCRIPTION 
#         return the half distance
#         imagine 3 points [1,2,3]
#         with distance X_12 and X_23 between them
#         This module will return 
#         L1 = X_12
#         L2 = X_12/2 + X_23/2
#         l3 = X_23
#     """''
#     L = [ ]
#     # Lenght initial and final
#     dist.insert(0,dist[0])
#     dist.insert(len(dist),dist[-1])
#     for i,e in zip(dist[:-1],dist[1:]):
#         length = (i+e)/2
#         L.append(length)
#     return L
# 
# 
# 
# if __name__ == "__main__":
#     dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     AttSta = att_sta()
#     AttSta.setInPaths(dirInPath)
#     
#     net_West = LCB_net()
#     net_East = LCB_net()
#     
#     net_West.AddFilesSta(AttSta.getatt(AttSta.stations(['Head','West','valley']),'InPath'))
#     net_East.AddFilesSta(AttSta.getatt(AttSta.stations(['Head','East','valley']),'InPath'))
#     
#     
#     
# #     group = LCB_group()
# #     group.add([net_West,net_East])
# #     
# #     group.getpara('stanames')
# #     group.report()
#     
#     stanames1 = net_West.getpara('stanames')
#     stanames2 = net_East.getpara('stanames')
#     
#     stanamesall = stanames1 + stanames2
#     print stanamesall
#     sortednames = att_sta().sortsta(stanamesall, 'Lat')['stanames']
#     
#     print sortednames
#     
#     dist = AttSta.dist_sta(sortednames, formcouples= True)
# 
# 
#     L = Lengths(dist)
#     L = np.array(L)*1000 # convert in meters 
# 
#     Length = dict(zip(sortednames,L)) # creer un dictionnaire avec le nom de la station associer a ca longeur L
# 
#     df_west = net_West.getvarallsta(var=['U m/s'], all=True)
#     df_east = net_East.getvarallsta(var=['U m/s'], all=True)
# 
#     df_west = MeanU(df_west, Length)
#     df_east = MeanU(df_east, Length)
# 
#     conv = ( df_west['meanU'] - df_east['meanU'])/(sum(L)/2)
# 
# 
#     plt.close()
#     plt.plot(conv.groupby(lambda t: (t.hour)).mean())
#     plt.plot([0]*len(conv.groupby(lambda t: (t.hour)).mean()))
#     plt.show()






