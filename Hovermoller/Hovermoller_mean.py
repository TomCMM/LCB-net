#===============================================================================
# Thomas July 2015
# DESCRIPTION
#     Run Classical Hovermoller Graph
#===============================================================================

# Library
import glob
from LCBnet_lib import *


#===============================================================================
# tets
#===============================================================================
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


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

    def Calc(self,network, by = None, var=None,From=None,group = None, To=None, From2=None,To2=None, Bywind=None, Byvar=None, rainfilter = None, overlap=False, mod=False, return_=None):
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
            
            stadata = sta.getData(var=var, From=From, To=To,From2=From2,To2=To2, by=Byvar, group = group, rainfilter = rainfilter)
            staU = sta.getData(var='U m/s', From=From, To=To,From2=From2,To2=To2, by=Bywind, group = group, rainfilter = rainfilter)
            staV = sta.getData(var='V m/s', From=From, To=To, From2=From2,To2=To2,by=Bywind, group = group, rainfilter = rainfilter)
            stadir = sta.getData(var='Dm G', From=From, To=To, From2=From2,To2=To2,by=Bywind, group = group, rainfilter = rainfilter)
    
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
                    stadata = stadata.pivot('hours','date',var[0])
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
        
        if return_:
            return data,U,V
        
        print "Calculated with sucess"

    def plot(self, OutPath = None, wind=True, averaged = False, pivot = False, data=None, U=None, V=None,
              title=True, zonal=None, vmin=None,vmax=None, step_contour=None,step_colorbar=None,wind_scale=None):
        """
        DESCRTIPTION
            make an hovermoller plot
        INPUT
            OutPath:
            averaged> True, make a mean hovermoller plot for the entire period
            pivot: True, pivot the hovermoller plot with the time axis on the x
            zonal: plot only the U component
            xlim: define the limite of the xaxis
        """

        lcbplot = LCBplot() # get the plot object
        argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot 
        arglabel = lcbplot.getarg('label')
        argticks = lcbplot.getarg('ticks')
        argfig = lcbplot.getarg('figure')
        arglegend = lcbplot.getarg('legend')
#         plt.rc('text', usetex=True)
#         print "o"*80
#         print argfig
#         
#         matplotlib.rc('font', family='sans-serif') 
#         matplotlib.rc('font', serif='Helvetica Neue') 
#         matplotlib.rc('text', usetex='false') 
#         matplotlib.rcParams.update({'font.size': 22})


        if averaged:
            dates = ['Averaged']
        else:
            dates = self.data_hov.items
 
        for date in dates:
             
            if averaged:
                if isinstance(data, pd.DataFrame):
                    var=data
                    U=U
                    V=V
                else:
                    var = self.data_hov.median(axis='items')
                    U = self.U_hov.median(axis='items')
                    V = self.V_hov.median(axis='items')
                    print U
            else:
                var = self.data_hov[date]
                U = self.U_hov[date]
                V = self.V_hov[date]

            if zonal:
                V = pd.DataFrame(0, index = V.index, columns = V.columns)

            # sort dataframe by position 
            var = var.reindex_axis(self.getpara('stanames'), axis=1)
            U = U.reindex_axis(self.getpara('stanames'), axis=1)
            V = V.reindex_axis(self.getpara('stanames'), axis=1)
 
            position, time = np.meshgrid(self.getpara('metadata'), var.index)

            print vmin
            if vmin:
                levels_contour = np.linspace(vmin, vmax, step_contour,endpoint=True)
                levels_colorbar = np.linspace(vmin, vmax, step_colorbar,endpoint=True)
            else:
                print "NO"*1010
                levels_contour=np.linspace(var.min().min(),var.max().max(),100)
                levels_colorbar=np.linspace(var.min().min(),var.max().max(),100)

            
            cmap = plt.cm.get_cmap("RdBu_r")
 
            fig = plt.figure(**argfig)
 
            if pivot:
                
                c = plt.contourf(time,position,var,levels=levels_contour,cmap=cmap)
                cbar = plt.colorbar(ticks=levels_colorbar)
                
                if wind:
                    a=plt.quiver(time, position,V.values, U.values,scale=20)
            else:

                # contour the plot first to remove any AA artifacts
                
                cnt= plt.contourf(position,time,var.values,levels_contour,cmap=cmap)
                #cnt.set_clim(vmin=15, vmax=25)
                for c in cnt.collections:
                    c.set_edgecolor("face")
                
                cbar = plt.colorbar(ticks=levels_colorbar)
#                 cbar.set_ticks(np.arange(12,26,2))
#                 cbar.set_ticklabels(np.arange(12,26,2))
                if wind:
                    if not wind_scale:
                        a=plt.quiver(position,time,U.values,V.values,scale=20)
                    else:
                        a=plt.quiver(position,time,U.values,V.values,scale=wind_scale)
 
 
            cbar.ax.tick_params( **argticks)
            if wind:
                qk = plt.quiverkey(a, 0.9, 1.05, 1, r'$1 \frac{m}{s}$',
                                        labelpos='E',
                                        fontproperties={'weight': 'bold'})
                if not zonal:
                    l,r,b,t = plt.axis()
                    dx, dy = r-l, t-b
                    plt.axis([l-0.1*dx, r+0.1*dx, b-0.1*dy, t+0.1*dy])
                else:
                    l,r,b,t = plt.axis()
                    dx, dy = r-l, t-b
                    plt.axis([l-0.1*dx, r+0.1*dx, b-0*dy, t+0*dy])
            
            
            if pivot:
                plt.ylabel(r"Longitude (Degree)", **arglabel)
                plt.xlabel( r"Hours", **arglabel)
            else:
                plt.ylabel(r"Hours", **arglabel)
                plt.xlabel( r"Longitude (Degree)", **arglabel)
            plt.grid(True, color="0.5")
            plt.tick_params(axis='both', which='major', **argticks)
#             plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))



            if title:
                plt.title("hovermoller_"+str(date))
   
            if not OutPath:
                
                OutPath = '/home/thomas/'
                  
#             plt.savefig(OutPath+"hovermoller_"+str(date)+'.png')
            plt.savefig(OutPath+"Hovermoller.svg", format='svg', dpi=1200)
            print('Plot with sucess')
            plt.close()



if __name__=='__main__':

#===============================================================================
# DIFF VERAO WINTER Average Hovermoller
#===============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     OutPath='/home/thomas/'
#     Files=glob.glob(InPath+"*")
#     net=LCB_net()
#     net.AddFilesSta(Files)
#     print('Success')
#     hov=Hovermoller(net)
#     network=hov.Select(att=['Head'],pos='Lon')
# #     network=hov.Select(att=['Ribeirao'],pos='Altitude')
#     print(network)
#   
#     data_verao, U_verao, V_verao = hov.Calc(network, var=['Ta C'],From='2014-11-01 00:00:00',To='2015-03-01 00:00:00', 
#                                             From2='2015-10-01 00:00:00',  To2='2016-01-01 00:00:00', Bywind='1H',Byvar='1H', group = "MH", return_=True)
#     data, U, V = hov.Calc(network, var=['Ta C'],From='2015-04-01 00:00:00',To='2015-10-01 00:00:00',Bywind='1H',Byvar='1H', group = "MH", return_=True)
#   
#   
#     data = data_verao.median(axis='items') - data.median(axis='items')
#     U = U_verao.median(axis='items') - U.median(axis='items')
#     V = V_verao.median(axis='items') - V.median(axis='items')
#   
#   
#     hov.plot(OutPath= OutPath, averaged=True, data=data, U=U, V=V, title=None, vmin=1, vmax=6, step_colorbar=11,step_contour=21, wind_scale=15)


# # ===============================================================================
# # # Average Hovermoller
# # #===============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     OutPath='/home/thomas/'
#     Files=glob.glob(InPath+"*")
#     net=LCB_net()
#     net.AddFilesSta(Files)
#     print('Success')
#     hov=Hovermoller(net)
#     network=hov.Select(att=['Head'],pos='Lon')
# #     network=hov.Select(att=['Ribeirao'],pos='Altitude')
#     print(network)
#   
#     data, U, V = hov.Calc(network, var=['Ta C'],From='2014-11-01 00:00:00',To='2016-01-01 00:00:00',Bywind='1H',Byvar='1H', group = "MH", return_=True)
#   
#   
#     hov.plot(OutPath= OutPath, averaged=True, data=data, U=U, V=V, title=None,vmin=14,vmax=26,step_colorbar=7,step_contour=25)

# # ===============================================================================
# # # Average Hovermoller Zonal
# # #===============================================================================
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    OutPath='/home/thomas/'
    Files=glob.glob(InPath+"*")
    net=LCB_net()
    net.AddFilesSta(Files)
    print('Success')
    hov=Hovermoller(net)
    network=hov.Select(att=['Head'],pos='Lon')
#     network=hov.Select(att=['Ribeirao'],pos='Altitude')
    print(network)
    
    data, U, V = hov.Calc(network, var=['Ua g/kg'],From='2014-11-01 00:00:00',To='2016-01-01 00:00:00',Bywind='1H',Byvar='1H', group = "MH", return_=True)
    
    
    hov.plot(OutPath= OutPath, averaged=True, zonal=True, data=data, U=U, V=V, title=None,vmin=11,vmax=13.5,step_colorbar=6,step_contour=21,wind_scale=15)


# # #===============================================================================
# # # Average Hovermoller MEDIO
# # #===============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     OutPath='/home/thomas/'
#     Files=glob.glob(InPath+"*")
#     net=LCB_net()
#     net.AddFilesSta(Files)
#     print('Success')
#     hov=Hovermoller(net)
#     network=hov.Select(att=['Medio'],pos='Lon')
# #     network=hov.Select(att=['Ribeirao'],pos='Altitude')
#     print(network)
#    
#     data, U, V = hov.Calc(network, var=['Ta C'],From='2014-11-01 00:00:00',To='2016-01-01 00:00:00',Bywind='1H',Byvar='1H', group = "MH", return_=True)
# 
#     hov.plot(OutPath= OutPath, averaged=True, data=data, U=U, V=V, title=None, vmin=14, vmax=26, step_colorbar=7, step_contour=25, wind_scale=10)


# #===============================================================================
# # Hovermoller Everyday
# #===============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
#     OutPath='/home/thomas/'
#     Files=glob.glob(InPath+"*")
#     net=LCB_net()
#     net.AddFilesSta(Files)
#     print('Success')
#     hov=Hovermoller(net)
#     network=hov.Select(att=['Medio'],pos='Lon')
#     print(network)
#  
#     hov.Calc(network, var=['Ta C'], From='2014-08-01 00:00:00', To='2015-10-01 00:00:00', Bywind='1H', Byvar='1H')
#  
#     hov.plot(OutPath= OutPath)

#===============================================================================
# Mode Hovermoller
#===============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     OutPath='/home/thomas/'
#     Files=glob.glob(InPath+"*")
#     net=LCB_net()
#     net.AddFilesSta(Files)
#     print('Success')
#     hov=Hovermoller(net)
#     network=hov.Select(att=['Medio'],pos='Lon')
# #     print(network)
#    
# #     data_verao, U_verao, V_verao = hov.Calc(network, var=['Ta C'],From='2014-11-01 00:00:00',To='2015-03-01 00:00:00',Bywind='1H',Byvar='1H', group = "MH")
#     hov.Calc(network, var=['Ua g/kg'], From='2014-10-01 00:00:00', To='2015-10-01 00:00:00', Bywind='1H', Byvar='1H', mod=True)
#     
# #     data = data_verao - data
#     hov.plot(OutPath= OutPath, averaged=True)




