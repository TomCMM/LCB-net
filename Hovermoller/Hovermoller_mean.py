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
            att=['Lon','Lat','network','side','Altitude','position']
        """

        if att == None:
            att = self.getpara('att')

        if pos == None:
            pos = self.getpara('pos')

        stanames = att_sta().stations(att)
        position=att_sta().sortsta(stanames, pos)['position']
        staname=att_sta().sortsta(stanames, pos)['staname']

        network=[]
        print(staname)
        for sta in staname:
            print(sta)
            s = self.net.getsta(sta)
            network.append(s[0])

        self.setpara('nbsta',len(staname))
        self.setpara('position',position)
        self.setpara('staname',staname)

        return network

    def Calc(self,network,var=None,From=None,group = None, To=None, Bywind=None, Byvar=None, rainfilter = None):
        """
        DESCRIPTION
            Manipulate the data of a list of stations to be in the format to be plotted
        """
        data=np.array([])
        U=np.array([])
        V=np.array([])

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


        for rr in network:
#             try:
#             rr.setpara('From',From)
#             rr.setpara('To',To)

            staU,staV = PolarToCartesian(rr.getvar('Sm m/s'),rr.getvar('Dm G'))
            rr.Data['U m/s']=staU
            rr.Data['V m/s']=staV

            stadata=rr.getData(var=var,From=From,To=To,by=Byvar, group = group, rainfilter = rainfilter)
            print(rr.Data.columns)
            staU=rr.getData(var='U m/s',From=From,To=To,by=Bywind, group = group, rainfilter = rainfilter)
            staV=rr.getData(var='V m/s',From=From,To=To,by=Bywind, group = group, rainfilter = rainfilter)
            
            data=np.append(data,stadata.tolist())
            U=np.append(U,staU.tolist())
            V=np.append(V,staV.tolist())



        return self.__reshape(data,U,V)

    def __reshape(self,data,U,V):
        """
        DESCRIPTION
            sub-method to reshape the data of the station in the correct format
        """
        nbsta=self.getpara('nbsta')
        lendata=len(data)/nbsta
        lenwind=len(U)/nbsta
        self.setpara('lendata',lendata)
        try:
            data=data.reshape(nbsta,lendata)
            U=U.reshape(nbsta,lenwind)
            V=V.reshape(nbsta,lenwind)
        except ValueError:
            print(lendata)
            print(len(data))
            print(nbsta)

        data=data.transpose()
        U=U.transpose()
        V=V.transpose()
        return data,U,V

    def __postime(self):
        """
        DESCRIPTION
            Calculate the X position and the Y time of the hovermoller necessary for the contour plot
        """
        lendata=self.getpara('lendata')
        time=range(0,lendata,1)#

        position=self.getpara('position')
        Position, Time = np.meshgrid(position, time)
        return Position,Time

    def plot(self,var,U,V, OutPath = None):
#         matplotlib.rc('xtick', labelsize=14)
#         matplotlib.rc('ytick', labelsize=14)

        Position,Time = self.__postime()
        Levels=np.linspace(var.min(),var.max(),100)
        cmap = plt.cm.get_cmap("RdBu_r")

        print(var)
        plt.contourf(Position,Time,var,levels=Levels,cmap=cmap)
        plt.colorbar()
 
        a=plt.quiver(Position[:,:],Time[::,:],U[:,:],V[:,:],scale=15)
        qk = plt.quiverkey(a, 0.9, 1.05, 1, r'$1 \frac{m}{s}$',
                                    labelpos='E',
                                    fontproperties={'weight': 'bold'})
 
        l,r,b,t = plt.axis()
        dx, dy = r-l, t-b
        plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

        if OutPath != None:
            plt.savefig('/home/thomas/hovermoler.png')
        else:
            plt.savefig('/home/thomas/hovermoler.png')
        print('Plot with sucess')
        plt.close()



if __name__=='__main__':
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    OutPath='/home/thomas/hovermoler.png'
    Files=glob.glob(InPath+"*")
    net=LCB_net()
    net.AddFilesSta(Files)
    print('Success')
    hov=Hovermoller(net)
    network=hov.Select(att=['Head'],pos='Lon')
    print(network)

    data_verao, U_verao, V_verao = hov.Calc(network, var='Ua g/kg',From='2014-11-01 00:00:00',To='2015-03-01 00:00:00',Bywind='1H',Byvar='1H', group = True)
    data, U , V = hov.Calc(network, var='Ua g/kg', From='2015-03-01 00:00:00', To='2015-08-01 00:00:00', Bywind='1H', Byvar='1H', group = True)
 
    data = data_verao - data
    U = U_verao - U
    V = V_verao - V

    hov.plot(data, U, V, OutPath= OutPath)


