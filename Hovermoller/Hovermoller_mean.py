#===============================================================================
# Thomas July 2015
# DESCRIPTION
#     Run Classical Hovermoller Graph
#===============================================================================

# Library
import glob
from LCBnet_lib import *


class Hovermoller():
    def __init__(self,net):
        self.net=net
        self.para=self.net.paradef
        self.paradef= {
               "Bywind":'1H',
               "Byvar":'1H',
               'pos':'Lon',
               'att':['Head'],
               'var' : 'Ta C'
               }
        self.att_sta = att_sta()

    def setpara(self,parameter,value):
        self.para[parameter]=value
        print(str(parameter)+' has been set to -> '+ str(value))

    def getpara(self,parameter):
        try:
            return self.para[parameter]
        except KeyError:
            print(parameter + ' has been not set -> Default value used ['+str(self.paradef[parameter])+']')
            try:
                return self.paradef[parameter]
            except KeyError:
                print(parameter+ ' dont exist')

    def delpara(self,varname):
        try:
            del self.para[varname]
            print('Deleted parameter-> ',varname)
        except KeyError:
            print('This parameter dont exist')

    def __setparadef(self,parameter,value):
        self.paradef[parameter]=value
        print(str(parameter)+' has been set by [default] to -> '+ str(value))

    def __getparadef(self,parameter):
        try:
            return self.paradef[parameter]
        except KeyError:
                print(parameter+ ' by [default] dont exist')

    def Select(self, att = None, pos = None):
        """
        DESCRIPTION
            return a sorted list of stations of the network given a list of parameters 
        EXAMPLE
            att=['Lon','Lat','network','side','Altitude','metadata']
        """

        if att == None:
            att = self.getpara('att')

        if pos == None:
            pos = self.getpara('pos')

        stanames = att_sta().stations(att)
        position=att_sta().sortsta(stanames, pos)['metadata']
        staname=att_sta().sortsta(stanames, pos)['stanames']

        network=[]
        print(staname)
        for sta in staname:
            print(sta)
            s = self.net.getsta(sta)
            network.append(s[0])

        self.setpara('nbsta',len(staname))
        self.setpara('metadata',position)
        self.setpara('stanames',staname)

        return network

    def Calc(self,network, by = None, var=None,From=None,group = None, To=None, Bywind=None, Byvar=None, rainfilter = None, overlap=False, mod=False):
        """
        DESCRIPTION
            Manipulate the data of a list of stations to be in the format to be plotted
        INPUT
            overlap: True, perform the opposite of U on the West side 
        """
        if From == None:
            From=self.getpara('From')
        if To == None:
            To=self.getpara('To')
        if Bywind == None:
            Bywind=self.getpara('Bywind')
        if Byvar == None:
            By=self.getpara('Byvar')
        if var == None:
            var = self.getpara('var')

        self.network = network
        data = {}
        U = {}
        V = {}

        for sta in network:
            
            stadata = sta.getData(var=var, From=From, To=To, by=Byvar, group = group, rainfilter = rainfilter)
            staU = sta.getData(var='U m/s', From=From, To=To, by=Bywind, group = group, rainfilter = rainfilter)
            staV = sta.getData(var='V m/s', From=From, To=To, by=Bywind, group = group, rainfilter = rainfilter)
            stadir = sta.getData(var='Dm G', From=From, To=To, by=Bywind, group = group, rainfilter = rainfilter)
    
            if overlap:
                print self.att_sta.getatt(sta.getpara('stanames'),'side')
                if self.att_sta.getatt(sta.getpara('stanames'),'side')[0] == "West":
                    print "West"
                    staU = -staU
#             print stadata.index.format(formatter=lambda x: x.strftime('%Y%m%d'))
#             

            stadata['hours'] = stadata.index.hour
            staU['hours'] = staU.index.hour
            staV['hours'] = staV.index.hour
            stadir['hours'] = stadir.index.hour
#             stadata['dayofyear'] = stadata.index.dayofyear
#             staU['dayofyear'] = staU.index.dayofyear
#             staV['dayofyear'] = staV.index.dayofyear

            stadata['date'] = stadata.index.format(formatter=lambda x: x.strftime('%Y%m%d'))
            staU['date'] = staU.index.format(formatter=lambda x: x.strftime('%Y%m%d'))
            staV['date'] = staV.index.format(formatter=lambda x: x.strftime('%Y%m%d'))
            stadir['date'] = stadir.index.format(formatter=lambda x: x.strftime('%Y%m%d'))

            if mod:
                    stadir['round']=((stadir['Dm G']/5).round(0)*5)
                    
                    staU_mod = []
                    staV_mod = []
                    for hour in range(0,23):
                        maxvalue_dir = stadir[stadir['hours'] ==hour]['round'].dropna().mode()
                        print maxvalue_dir

                        d = pd.concat([staU['U m/s'], staV['V m/s'], stadir['round'], stadir['hours']], axis=1, join="outer")

                        print d['hours'].shape
                        print d['round'].shape
                        select = d[(d['hours']==hour) & (d['round']==maxvalue_dir[0])]
                        staU_mod.append(select['U m/s'].mean())
                        staV_mod.append(select['V m/s'].mean())

#                     staU=staU.groupby(lambda t: (t.hour)).agg(lambda x: stats.mode(x)[0])
#                     staV=staV.groupby(lambda t: (t.hour)).agg(lambda x: stats.mode(x)[0])

                    
                    
                    staU = pd.DataFrame({'U m/s':staU_mod}, index=range(len(staU_mod)))
                    staV =pd.DataFrame({'V m/s':staV_mod}, index=range(len(staV_mod)))
                    stadata = stadata[var].groupby(lambda t: (t.hour)).mean()
            else:
                    stadata = stadata.pivot('hours','date','Ta C')
                    staU = staU.pivot('hours','date','U m/s')
                    staV = staV.pivot('hours','date','V m/s')

            data[sta.getpara('stanames')] = stadata
            U[sta.getpara('stanames')] = staU
            V[sta.getpara('stanames')] = staV
            
        
        # Create panel
#         print data
#         print "0"*120
#         print U['C07']
        data = pd.Panel(data)
        U = pd.Panel(U)
        V = pd.Panel(V)
        

        data =  data.transpose(2,1,0)
        U = U.transpose(2,1,0)
        V = V.transpose(2,1,0)
        
        
        self.data_hov = data
        self.U_hov = U 
        self.V_hov = V
        
        print "Calculated with sucess"

    def plot(self, OutPath = None, averaged = False, pivot = False):
        """
        DESCRTIPTION
            make an hovermoller plot
        INPUT
            OutPath:
            averaged> True, make a mean hovermoller plot for the entire period
            pivot: True, pivot the hovermoller plot with the time axis on the x
        """
        
        if averaged:
            dates = ['Averaged']
        else:
            dates = self.data_hov.items

        for date in dates:
            
            if averaged:
                var = self.data_hov.median(axis='items')
                U = self.U_hov.median(axis='items')
                V = self.V_hov.median(axis='items')
                print U
            else:
                var = self.data_hov[date]
                U = self.U_hov[date]
                V = self.V_hov[date]

            # sort dataframe by position 
            var = var.reindex_axis(self.getpara('stanames'), axis=1)
            U = U.reindex_axis(self.getpara('stanames'), axis=1)
            V = V.reindex_axis(self.getpara('stanames'), axis=1)

            position, time = np.meshgrid(self.getpara('metadata'), var.index)
 
            Levels=np.linspace(var.min().min(),var.max().max(),100)
            cmap = plt.cm.get_cmap("RdBu_r")

            if pivot:
                plt.contourf(time,position,var,levels=Levels,cmap=cmap)
                plt.colorbar()
                a=plt.quiver(time, position,V.values, U.values,scale=40)
            else:
                print "ALLLOOOOO"
                print U
                plt.contourf(position,time,var.values,levels=Levels,cmap=cmap)
                plt.colorbar()
                a=plt.quiver(position,time,U.values,V.values,scale=40)


            qk = plt.quiverkey(a, 0.9, 1.05, 1, r'$1 \frac{m}{s}$',
                                        labelpos='E',
                                        fontproperties={'weight': 'bold'})
         
            l,r,b,t = plt.axis()
            dx, dy = r-l, t-b
            plt.axis([l-0.2*dx, r+0.2*dx, b-0.2*dy, t+0.2*dy])
            plt.title("hovermoller_"+str(date))
  
            if not OutPath:
                OutPath = '/home/thomas/'
                 
            plt.savefig(OutPath+"hovermoller_"+str(date)+'.png')
            print('Plot with sucess')
            plt.close()



if __name__=='__main__':

#===============================================================================
# Average Hovermoller
#===============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     OutPath='/home/thomas/'
#     Files=glob.glob(InPath+"*")
#     net=LCB_net()
#     net.AddFilesSta(Files)
#     print('Success')
#     hov=Hovermoller(net)
#     network=hov.Select(att=['Head'],pos='Lon')
#     network=hov.Select(att=['Ribeirao'],pos='Altitude')
#     print(network)
 
#     data_verao, U_verao, V_verao = hov.Calc(network, var=['Ta C'],From='2014-11-01 00:00:00',To='2015-03-01 00:00:00',Bywind='1H',Byvar='1H', group = "MH")
#     hov.Calc(network, var=['Ta C'], From='2014-11-01 00:00:00', To='2015-11-01 00:00:00', Bywind='1H', Byvar='1H', overlap=True)
  
#     data = data_verao - data
#     U = U_verao - U
#     V = V_verao - V
#  
#  
#     hov.plot(OutPath= OutPath, averaged=True)
#     hov.plot(OutPath= OutPath, averaged=True, pivot=True)

# #===============================================================================
# # Hovermoller Everyday
# #===============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     OutPath='/home/thomas/'
#     Files=glob.glob(InPath+"*")
#     net=LCB_net()
#     net.AddFilesSta(Files)
#     print('Success')
#     hov=Hovermoller(net)
#     network=hov.Select(att=['Head'],pos='Lon')
#     print(network)
# 
#     hov.Calc(network, var=['Ta C'], From='2014-08-01 00:00:00', To='2015-10-01 00:00:00', Bywind='1H', Byvar='1H')
# 
#     hov.plot(OutPath= OutPath)

#===============================================================================
# Mode Hovermoller
#===============================================================================
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    OutPath='/home/thomas/'
    Files=glob.glob(InPath+"*")
    net=LCB_net()
    net.AddFilesSta(Files)
    print('Success')
    hov=Hovermoller(net)
#     network=hov.Select(att=['Head'],pos='Lon')
    network=hov.Select(att=['Medio'],pos='Lon')
    print(network)
 
#     data_verao, U_verao, V_verao = hov.Calc(network, var=['Ta C'],From='2014-11-01 00:00:00',To='2015-03-01 00:00:00',Bywind='1H',Byvar='1H', group = "MH")
    hov.Calc(network, var=['Ta C'], From='2014-11-01 00:00:00', To='2015-11-01 00:00:00', Bywind='1H', Byvar='1H', mod=True)
  
#     data = data_verao - data
    hov.plot(OutPath= OutPath, averaged=True)




