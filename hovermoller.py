
#===============================================================================
# Hovmoller Station + maxarg en minute
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/Hovermoller/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")

Files=[
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C08clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C07clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C06clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C05clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C11clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C12clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C13clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C14clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C15clear_merge.TXT',
 ]

network=[]
for i in Files:
    print(i)
    rr=LCB_station(i)
    if rr.getpara('InPath') =='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C18clear_merge.TXT':
        rr.Data.index=rr.Data.index- pd.DateOffset(hours=3)
    network.append(rr)


#Files=reversed(Files)
position=[]
staname=[]
stationsnames=att_sta().stations(['Head'])
stations=att_sta().sortsta(stationsnames,'Lon')
position=stations['position']
staname=stations['staname']

#position=position[::-1]
nbsta=len(staname)


time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30

Z=np.array([])
Wind_speed=np.array([])
Wind_dir=np.array([])
Norm=np.array([])
Theta=np.array([])
for rr in network:
    #rr.Data['Date']=rr.Data.index.date
    #rr.Data=rr.Data[rr.Data.Date.isin(net_max.Data.index.date)]
    #print(rr.Data.index)
    #del rr.Data['Date']
    #print(len(rr.daily()['Ta C'].tolist()))
#     rr.Set_From('2014-09-01 00:00:00')
#     rr.Set_To('2015-04-01 00:00:00')
    #daily=rr.Data[rr.From:rr.To]
    #daily=rr.Data[rr.From:rr.To].groupby(pd.TimeGrouper('10Min')).mean()
    #print(daily.shape)
    #daily=rr.daily_h()
    daily=rr.Data['2014-09-01 00:00:00':'2015-05-01 00:00:00'].groupby(pd.TimeGrouper('2min')).mean()
    daily=daily.groupby(lambda t: (t.hour,t.minute)).mean()
    Z=np.append(Z,daily['Ta C'].tolist())
    daily_wind=rr.Data['2014-09-01 00:00:00':'2015-05-01 00:00:00']
    daily_wind['Dm G']=((daily_wind['Dm G']/10).round(0)*10)
    grouped=daily_wind.groupby(lambda t: (t.hour,t.minute)).agg(lambda x: stats.mode(x)[0])
    hours=[]
    minutes=[]
    dminute=[]
    for i in grouped.index:
        hours.append(i[0])
        minutes.append(i[1])
    grouped['hours']=hours
    grouped['minutes']=minutes
    grouped['dizaine_minutes']=(((np.array(minutes)-5)/10).round(0))*10
    grouped=grouped.groupby(['hours','dizaine_minutes']).agg(lambda x: stats.mode(x)[0])
    grouped['hours']=grouped.index.get_level_values(0)
    grouped['dizaine_minutes']=grouped.index.get_level_values(1)
    wind_select=daily_wind.groupby([lambda t: (t.hour,t.minute),daily_wind['Dm G']]).mean()# group by hour and by the orientation
    hours=[]
    minutes=[]
    dminute=[]
    for i in wind_select.index:
        hours.append(i[0][0])
        minutes.append(i[0][1])
    wind_select['hours']=hours
    wind_select['minutes']=minutes
    wind_select['dizaine_minutes']=(((np.array(minutes)-5)/10).round(0))*10
    Wind=[]
    for h in range(0,24):
        for m in range(0,60,10):
            print(h,m)
            data=wind_select[(wind_select['hours'] == h) & (wind_select['dizaine_minutes'] == m)]
            dirmax=grouped[(grouped['hours'] == h) & (grouped['dizaine_minutes'] == m)]['Dm G']
            Wind.append(data[data['Dm G'] == int(dirmax)]['Sm m/s'].mean())
    Wind=np.array(Wind)
    Norm=np.append(Norm,Wind)
    Theta=np.append(Theta,grouped['Dm G'].tolist())



Z=Z.reshape(nbsta,720)
V=np.cos(map(math.radians,Theta+180))*Norm # V
U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!


#------------------------------------------------------------------------------ 
#V_wind=np.cos(map(math.radians,Wind_dir+180))*Wind_speed
#U_wind=np.sin(map(math.radians,Wind_dir+180))*Wind_speed
#ratio_circ=((np.abs(U_wind)-np.abs(V_wind))/(np.abs(U_wind)+np.abs(V_wind)))*100# Wind ration between cross-valley mountain and plain valley circulatiion
#Z=ratio_circ.reshape (nbsta,720) 

#------------------------------------------------------------------------------ 
#V_wind=np.cos(map(math.radians,Wind_dir+180))*Wind_speed
#Z=V_wind.reshape(nbsta,720)

U=U.reshape(nbsta,144)
V=V.reshape(nbsta,144)

Z=Z.transpose()
U=U.transpose()
V=V.transpose()


start=0
end=19
U.shape
V.shape
Z.shape
Position.shape
Time.shape
Levels=np.linspace(Z.min(),Z.max(),100)
#Levels=np.linspace(-0.1,0.1,30)
cmap = plt.cm.get_cmap("RdBu_r")
plt.contourf(Position[:,:],Time[:,:],Z[:,:],levels=Levels,cmap=cmap)    
plt.colorbar()
a=plt.quiver(Position[::6,:],Time[::6,:],U[:,:],V[:,:],scale=35)
#plt.gca().invert_xaxis()    


l,r,b,t = plt.axis()
dx, dy = r-l, t-b
plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

plt.savefig('hovermoler.png')
plt.close()


#===============================================================================
# Hovmoller Station + maxarg en hourly
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/Hovermoller/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")

Files=[
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C08clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C07clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C06clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C05clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C11clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C12clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C13clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C14clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C15clear_merge.TXT',
 ]

network=[]
for i in Files:
    print(i)
    rr=LCB_station(i)
    if rr.getpara('InPath') =='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C18clear_merge.TXT':
        rr.Data.index=rr.Data.index- pd.DateOffset(hours=3)
    network.append(rr)


#Files=reversed(Files)
position=[]
staname=[]
stationsnames=att_sta().stations(['Head'])
stations=att_sta().sortsta(stationsnames,'Lon')
position=stations['position']
staname=stations['staname']

#position=position[::-1]
nbsta=len(staname)


time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30

Z=np.array([])
Wind_speed=np.array([])
Wind_dir=np.array([])
Norm=np.array([])
Theta=np.array([])
for rr in network:
    daily=rr.Data['2014-09-10 00:00:00':'2015-04-10 00:00:00'].groupby(pd.TimeGrouper('2min')).mean()
    daily=daily.groupby(lambda t: (t.hour,t.minute)).mean()
    Z=np.append(Z,daily['Ta C'].tolist())
    daily_wind=rr.Data['2014-09-10 00:00:00':'2015-04-10 00:00:00'].groupby(pd.TimeGrouper('10min')).mean()
    daily_wind['Dm G']=((daily_wind['Dm G']/5).round(0)*5)
    grouped=daily_wind.groupby(lambda t: (t.hour)).agg(lambda x: stats.mode(x)[0])
    wind_select=daily_wind.groupby([lambda t: (t.hour),daily_wind['Dm G']]).mean()# group by hour and by the orientation
    hours=[]
    for i in wind_select.index:
        hours.append(i[0])
    wind_select['hours']=hours
    Wind=[]
    for h in range(0,24):
        data=wind_select[(wind_select['hours'] == h)]
        dirmax=grouped[grouped.index == h]['Dm G']
        Wind.append(data[data['Dm G'] == int(dirmax)]['Sm m/s'].mean())
    Wind=np.array(Wind)
    Norm=np.append(Norm,Wind)
    Theta=np.append(Theta,grouped['Dm G'].tolist())



Z=Z.reshape(nbsta,720)
V=np.cos(map(math.radians,Theta+180))*Norm # V
U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!


#------------------------------------------------------------------------------ 
#V_wind=np.cos(map(math.radians,Wind_dir+180))*Wind_speed
#U_wind=np.sin(map(math.radians,Wind_dir+180))*Wind_speed
#ratio_circ=((np.abs(U_wind)-np.abs(V_wind))/(np.abs(U_wind)+np.abs(V_wind)))*100# Wind ration between cross-valley mountain and plain valley circulatiion
#Z=ratio_circ.reshape (nbsta,720) 

#------------------------------------------------------------------------------ 
#V_wind=np.cos(map(math.radians,Wind_dir+180))*Wind_speed
#Z=V_wind.reshape(nbsta,720)

U=U.reshape(nbsta,24)
V=V.reshape(nbsta,24)

Z=Z.transpose()
U=U.transpose()
V=V.transpose()


start=0
end=19
U.shape
V.shape
Z.shape
Position.shape
Time.shape

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

Levels=np.linspace(Z.min(),Z.max(),100)
#Levels=np.linspace(-0.1,0.1,30)
cmap = plt.cm.get_cmap("RdBu_r")
plt.contourf(Position[:,:],Time[:,:],Z[:,:],levels=Levels,cmap=cmap)    
plt.colorbar()
a=plt.quiver(Position[::30,:],Time[::30,:],U[:,:],V[:,:],scale=35)
#plt.gca().invert_xaxis()    


l,r,b,t = plt.axis()
dx, dy = r-l, t-b
plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

plt.savefig('hovermoler.eps')
plt.close()





#===============================================================================
# Hovmoller Station + Mean
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/Hovermoller/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")

Files=[
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C08clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C07clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C06clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C05clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C11clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C12clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C13clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C14clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C15clear_merge.TXT',
 ]

network=[]
for i in Files:
    print(i)
    rr=LCB_station(i)
    if rr.getpara('InPath') =='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C18clear_merge.TXT':
        rr.Data.index=rr.Data.index- pd.DateOffset(hours=3)
    network.append(rr)


#Files=reversed(Files)
position=[]
staname=[]
stationsnames=att_sta().stations(['Head'])
stations=att_sta().sortsta(stationsnames,'Lon')
position=stations['position']
staname=stations['staname']

#position=position[::-1]
nbsta=len(staname)


time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30

Z=np.array([])
Wind_speed=np.array([])
Wind_dir=np.array([])
Norm=np.array([])
Theta=np.array([])
for rr in network:
    #rr.Data['Date']=rr.Data.index.date
    #rr.Data=rr.Data[rr.Data.Date.isin(net_max.Data.index.date)]
    #print(rr.Data.index)
    #del rr.Data['Date']
    #print(len(rr.daily()['Ta C'].tolist()))
#     rr.Set_From('2014-09-01 00:00:00')
#     rr.Set_To('2015-04-01 00:00:00')
    #daily=rr.Data[rr.From:rr.To]
    #daily=rr.Data[rr.From:rr.To].groupby(pd.TimeGrouper('10Min')).mean()
    #print(daily.shape)
    #daily=rr.daily_h()
    daily=rr.Data['2014-09-01 00:00:00':'2015-05-01 00:00:00'].groupby(pd.TimeGrouper('2min')).mean()
    daily=daily.groupby(lambda t: (t.hour,t.minute)).mean()
    daily_wind=rr.Data['2015-03-01 00:00:00':'2015-05-01 00:00:00'].groupby(pd.TimeGrouper('30min')).mean()
    daily_wind=daily_wind.groupby(lambda t: (t.hour,t.minute)).mean()
    #daily=rr.Data.groupby(pd.TimeGrouper('1H')).mean()
    #daily=daily.groupby(lambda t: (t.hour,t.minute)).mean()
    #print(len(daily.index))
    #a=np.array([daily['Theta C'][:-1:]])
    #b=np.array([daily['Theta C'][1::]])
    #c=(b-a)/10
    Z=np.append(Z,daily['Ua g/kg'].tolist())
    #Wind_speed=np.append(Wind_speed,daily['Sm m/s'].tolist())
    #Wind_dir=np.append(Wind_dir,daily['Dm G'].tolist())
    #Z=np.append(Z,c.tolist())
    Norm=np.append(Norm,daily_wind['Sm m/s'].tolist())
    Theta=np.append(Theta,daily_wind['Dm G'].tolist())





Z=Z.reshape(nbsta,720)
V=np.cos(map(math.radians,Theta+180))*Norm # V
U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!


#------------------------------------------------------------------------------ 
#V_wind=np.cos(map(math.radians,Wind_dir+180))*Wind_speed
#U_wind=np.sin(map(math.radians,Wind_dir+180))*Wind_speed
#ratio_circ=((np.abs(U_wind)-np.abs(V_wind))/(np.abs(U_wind)+np.abs(V_wind)))*100# Wind ration between cross-valley mountain and plain valley circulatiion
#Z=ratio_circ.reshape (nbsta,720) 

#------------------------------------------------------------------------------ 
#V_wind=np.cos(map(math.radians,Wind_dir+180))*Wind_speed
#Z=V_wind.reshape(nbsta,720)

U=U.reshape(nbsta,48)
V=V.reshape(nbsta,48)

Z=Z.transpose()
U=U.transpose()
V=V.transpose()


start=0
end=19
U.shape
V.shape
Z.shape
Position.shape
Time.shape
Levels=np.linspace(Z.min(),Z.max(),100)
#Levels=np.linspace(-0.1,0.1,30)
cmap = plt.cm.get_cmap("RdBu_r")
plt.contourf(Position[:,:],Time[:,:],Z[:,:],levels=Levels,cmap=cmap)    
plt.colorbar()
a=plt.quiver(Position[::15,:],Time[::15,:],U[:,:],V[:,:],scale=35)
#plt.gca().invert_xaxis()    


l,r,b,t = plt.axis()
dx, dy = r-l, t-b
plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

plt.savefig('hovermoler.png')
plt.close()



#===============================================================================
# Hover Moller Anomalie versant
#===============================================================================


InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")
net=LCB_net()
Files.sort()
Files_west=[
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C08clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C07clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C06clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT',
 ]

Files_east=[
#'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C11clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C12clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C13clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C14clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C15clear_merge.TXT',
 ]
Files_west=reversed(Files_west)
Files_east=reversed(Files_east)


#Files.remove('/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C06clear_merge.TXT')
#Files.remove('/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT')
#Files.remove('/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C14clear_merge.TXT')
#------------------------------------------------------------------------------ 
# Network
net_west=LCB_net()
network_west=[]
for i in Files_west:
    print(i)
    rr=LCB_station(i)
    net_west.add(rr)
    network_west.append(rr)

network_east=[]
net_east=LCB_net()
for i in Files_east:
    print(i)
    rr=LCB_station(i)
    net_east.add(rr)
    network_east.append(rr)

#------------------------------------------------------------------------------ 
daily=[]
for i in network_west:
    i.Set_From('2014-10-05 00:00:00')
    i.Set_To('2014-11-25 00:00:00')
    daily.append(i.daily())

daily_net=pd.concat(daily)
net_west.daily=daily_net.groupby(daily_net.index).mean()

daily=[]
for i in network_east:
    i.Set_From('2014-10-05 00:00:00')
    i.Set_To('2014-11-25 00:00:00')
    daily.append(i.daily())

daily_net=pd.concat(daily)
net_east.daily=daily_net.groupby(daily_net.index).mean()

#------------------------------------------------------------------------------ 
Theta_west=np.array([])
Norm_west=np.array([])
Z_west=np.array([])
for rr in network_west:
    print(rr.InPath)
    rr.Set_From('2014-10-05 00:00:00')
    rr.Set_To('2014-11-25 00:00:00')
    Z_west=np.append(Z_west,(rr.daily()['Ta C']-net_west.daily['Ta C']).tolist())
    Norm_west=np.append(Norm_west,rr.daily_h()['Sm m/s'].tolist())
    Theta_west=np.append(Theta_west,rr.daily_h()['Dm G'].tolist())




Z_east=np.array([])
Norm_east=np.array([])
Theta_east=np.array([])
for rr in network_east:
    print(rr.InPath)
    rr.Set_From('2014-10-05 00:00:00')
    rr.Set_To('2014-11-25 00:00:00')
    Z_east=np.append(Z_east,(rr.daily()['Ta C']-net_east.daily['Ta C']).tolist())
    Norm_east=np.append(Norm_east,rr.daily_h()['Sm m/s'].tolist())
    Theta_east=np.append(Theta_east,rr.daily_h()['Dm G'].tolist())

#------------------------------------------------------------------------------
#Altitude=[1342,1272,1206,1127,1077,1031,1061,1075,1140,1186,1225,1356]
Name=[9,8,7,6,4,11,12,13,14,15]
position=[46.258833,46.256667,46.254528,46.252861,46.249083,46.245861,46.243694,46.241278,46.238472,46.237139]
Altitude=[1356,1225,1186,1140,1061,1077,1127,1206,1279,1342]

Altitude_east=[1077,1127,1206,1279,1342]#10,11,12,13,14,15

Altitude_west=[1061,1140,1186,1225,1356]#4,6,7,8,9


ZZ=np.array([])
for i,v in enumerate(rr.daily().index):
    print(i)
    T_west=[]
    for rr in network_west:#4,5,7,8,9
        rr.Set_From('2014-10-05 00:00:00')
        rr.Set_To('2014-11-25 00:00:00')
        T_west.append(rr.daily()['Ta C'][i])
    NT_west=np.interp(Altitude_east,Altitude_west,T_west)# temperature de laface west projet a l altitude des stations de la face east
    T_east=[]
    for rr in network_east[::-1]:#15,13,12,11,10
        rr.Set_From('2014-10-05 00:00:00')
        rr.Set_To('2014-11-25 00:00:00')
        T_east.append(rr.daily()['Ta C'][i])
    NT_east=np.interp(Altitude_west,Altitude_east,T_east)# temperature de laface west projet a l altitude des stations de la face east
    ANOT_east=T_east-NT_west
    ANOT_west=T_west-NT_east
    #ANOT_east=NT_east
    #ANOT_west=NT_west
    ZZ=np.append(ZZ,[ANOT_west[::-1],ANOT_east])

    
#position=range(0,2400,200)# Position of the WXT distance in meter


time=range(0,1440,2)#
Position, Time = np.meshgrid(position, time)
Time=Time/60

#Z=np.concatenate((Z_east,Z_west))
Theta=np.concatenate((Theta_east,Theta_west))
Norm=np.concatenate((Norm_east,Norm_west))

Z=ZZ.reshape(720,10)
V=np.cos(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!
U=np.sin(map(math.radians,Theta+180))*Norm

U=U.reshape(10,24)
V=V.reshape(10,24)
#Z=Z.transpose()
U=U.transpose()
V=V.transpose()

U.shape
V.shape
Z.shape
Position.shape
Time.shape


Levels=np.linspace(Z.min(),Z.max(),100)
#Levels=np.linspace(-1,1,50)
plt.contourf(Position,Time,Z,levels=Levels)    
plt.gca().invert_xaxis()    
plt.colorbar()
plt.quiver(Position[::30,:],Time[::30,:],U[:,:],V[:,:],scale=30)

plt.savefig('hovermoler_Thetaweakwind.png')
plt.close()

(net_west.daily['Theta C']-net_east.daily['Theta C']).plot()
plt.savefig('anomalieversant.png')
plt.close()

#===============================================================================
# Hovmoller Station + Mean + sta9winddirsout 12am to 12 pm and sta9wind North 12pm to 12 am 
#===============================================================================
Path='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT'
station = LCB_station(Path)

windsta9=station.getvar('Sm m/s')
dirsta9=station.getvar('Dm G')


InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/Hovermoller/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")

Files=[
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C08clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C07clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C06clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C05clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C11clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C12clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C13clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C14clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C15clear_merge.TXT',
 ]

network=[]
for i in Files:
    print(i)
    rr=LCB_station(i)
    if rr.getpara('InPath') =='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C18clear_merge.TXT':
        rr.Data.index=rr.Data.index- pd.DateOffset(hours=3)
    network.append(rr)


#Files=reversed(Files)
position=[]
staname=[]
stationsnames=att_sta().stations(['Head'])
stations=att_sta().sortsta(stationsnames,'Lon')
position=stations['position']
staname=stations['staname']

#position=position[::-1]
nbsta=len(staname)


time=range(0,540,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30

Z=np.array([])
Wind_speed=np.array([])
Wind_dir=np.array([])
Norm=np.array([])
Theta=np.array([])
for rr in network:
    daily=rr.Data['2014-09-01 00:00:00':'2015-04-10 00:00:00'].groupby(pd.TimeGrouper('2min')).mean()
    daily=daily[AccDailyRain < 0.1].between_time('18:00','11:58')# dry
    dir=dirsta9[daily.index]
    daily=daily[(dir < 60) & (dir > 20)]
#     daily_afternoon=daily.between_time('12:00','23:58')
#     daily_morning=daily.between_time('00:00','11:58')
#     daily_afternoon=daily_afternoon[(dir.between_time('12:00','23:58') <180 ) & (dir.between_time('12:00','23:58') <135 ) ]
#     daily_morning=daily_morning[(dir.between_time('00:00','11:58') <180 ) & (dir.between_time('00:00','23:58') >135 ) ]
#     daily=daily_morning.append(daily_afternoon)
    daily=daily.groupby(lambda t: (t.hour,t.minute)).mean()
    daily_wind=rr.Data['2014-09-01 00:00:00':'2015-04-10 00:00:00'].groupby(pd.TimeGrouper('30min')).mean()
    daily_wind=daily_wind[AccDailyRain < 0.1].between_time('18:00','11:58')# dry
    dir=dirsta9[daily_wind.index]
    daily_wind=daily_wind[(dir < 60) & (dir > 20)]
#     daily_afternoon=daily_wind.between_time('12:00','23:58')
#     daily_morning=daily_wind.between_time('00:00','11:58')
#     daily_afternoon=daily_afternoon[(dir.between_time('12:00','23:58') <180 ) & (dir.between_time('12:00','23:58') >135 ) ]
#     daily_morning=daily_morning[(dir.between_time('00:00','11:58') <180 ) & (dir.between_time('00:00','23:58') >135 ) ]
#     daily_wind=daily_morning.append(daily_afternoon)
    daily_wind=daily_wind.groupby(lambda t: (t.hour,t.minute)).mean()
    data=np.array(daily['Ta C'].tolist())
    Z=np.append(Z,daily['Ta C'].tolist())
    Norm=np.append(Norm,daily_wind['Sm m/s'].tolist())
    Theta=np.append(Theta,daily_wind['Dm G'].tolist())
    


Z=Z.reshape(nbsta,540)
V=np.cos(map(math.radians,Theta+180))*Norm # V
U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!

U=U.reshape(nbsta,36)
V=V.reshape(nbsta,36)

Z=Z.transpose()
U=U.transpose()
V=V.transpose()


start=0
end=19
U.shape
V.shape
Z.shape
Position.shape
Time.shape
#Levels=np.linspace(Z.min(),Z.max(),100)
Levels=np.linspace(12,26,100)
cmap = plt.cm.get_cmap("RdBu_r")
plt.contourf(Position[:,:],Time[:,:],Z[:,:],levels=Levels,cmap=cmap)    
plt.colorbar()
a=plt.quiver(Position[::15,:],Time[::15,:],U[:,:],V[:,:],scale=35)

l,r,b,t = plt.axis()
dx, dy = r-l, t-b
plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

plt.savefig('hovermoler.png')
plt.close()


#===============================================================================
# Hovmoller Station + Mean for article
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/Hovermoller/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")

Files=[
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C08clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C07clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C06clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C05clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C11clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C12clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C13clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C14clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C15clear_merge.TXT',
 ]
# 
# Files=[
# '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C16clear_merge.TXT',
# '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C17clear_merge.TXT',
# '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C18clear_merge.TXT',
# '/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C19clear_merge.TXT',
#  ]

network=[]

net=LCB_net()
for i in Files:
    print(i)
    rr=LCB_station(i)
    if rr.getpara('InPath') =='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C18clear_merge.TXT':
        rr.Data.index=rr.Data.index- pd.DateOffset(hours=3)
    network.append(rr)
    net.add(rr)

#AccDailyRain=net.Data['Rc mm'].resample("1H",how='sum').reindex(index=net.Data.index,method='ffill')






#Files=reversed(Files)
position=[]
staname=[]
stationsnames=att_sta().stations(['Head'])
stations=att_sta().sortsta(stationsnames,'Lon')
position=stations['position']
staname=stations['staname']

#position=position[::-1]
nbsta=len(staname)


time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30

def PolarToCartesian(norm,theta):
    """
    Transform polar to Cartesian where 0 = North, East =90 ....
    """
    U=norm*np.cos(map(math.radians,-theta+270))
    V=norm*np.sin(map(math.radians,-theta+270))
    return U,V


Z=np.array([])
Wind_speed=np.array([])
Wind_dir=np.array([])
U=np.array([])
V=np.array([])
for rr in network:
    #AccDailyRain=AccDailyRain['2014-09-10 00:00:00':'2015-04-10 00:00:00'].groupby(pd.TimeGrouper('2min')).mean()
    daily=rr.Data['2014-09-05 00:00:00':'2015-05-10 00:00:00'].groupby(pd.TimeGrouper('2min')).mean()
#     daily=daily.append(rr.Data['2015-02-12 00:00:00':'2015-03-09 00:00:00'].groupby(pd.TimeGrouper('2min')).mean())
    #daily=daily[AccDailyRain <0.1]
    daily=daily.groupby(lambda t: (t.hour,t.minute)).mean()
    daily_wind=rr.Data['2014-09-05 00:00:00':'2015-05-10 00:00:00']
    Uwind,Vwind = PolarToCartesian(daily_wind['Sm m/s'],daily_wind['Dm G'])
    daily_wind['U']=Uwind
    daily_wind['V']=Vwind
    daily_wind=daily_wind.groupby(pd.TimeGrouper('30min')).mean()
#   daily_wind=daily_wind.append(rr.Data['2015-02-12 00:00:00':'2015-03-09 00:00:00'].groupby(pd.TimeGrouper('30min')).mean())
    #daily_wind=daily_wind[AccDailyRain < 0.1]
    daily_wind=daily_wind.groupby(lambda t: (t.hour,t.minute)).mean()
    Z=np.append(Z,daily['Ta C'].tolist())
    U=np.append(U,daily_wind['U'].tolist())
    V=np.append(V,daily_wind['V'].tolist())




Z=Z.reshape(nbsta,720)
# V=np.cos(map(math.radians,Theta+180))*Norm # V
# U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!


U=U.reshape(nbsta,48)
V=V.reshape(nbsta,48)

Z=Z.transpose()
U=U.transpose()
V=V.transpose()


matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

start=0
end=19
U.shape
V.shape
Z.shape
Position.shape
Time.shape
Levels=np.linspace(Z.min(),Z.max(),100)
cmap = plt.cm.get_cmap("RdBu_r")
plt.contourf(Position[:,:],Time[:,:],Z[:,:],levels=Levels,cmap=cmap)    
plt.colorbar()
a=plt.quiver(Position[::15,:],Time[::15,:],U[:,:],V[:,:],scale=15)
qk = plt.quiverkey(a, 0.9, 1.05, 1, r'$1 \frac{m}{s}$',
                            labelpos='E',
                            fontproperties={'weight': 'bold'})

l,r,b,t = plt.axis()
dx, dy = r-l, t-b
plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

plt.savefig('hovermoler.png')
plt.close()



#===============================================================================
# Hovmoller Station - interpolation + selection synoptic condition
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/Hovermollerinterp/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")

#Files=reversed(Files)
position=[]
staname=[]
stations=pos_sta()

for i in stations:
    position.append(i[1])
    staname.append(i[0])


#Files=['/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C05clear_merge.TXT']

network=[]
for i in Files:
    print(i)
    rr=LCB_station(i)
    network.append(rr)



#position=position[::-1]
nbsta=len(Files)


time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30


# Selection events
InPath='/home/thomas/PhD/obs-lcb/synoptic/SyntheseSynopticCPTEC/synoptic_condition.csv'
Eventsynoptic=pd.read_csv(InPath,index_col=0,parse_dates=True)

Eventsynoptic=Eventsynoptic['2015-02-01'::]# avoid september

for event in ['ZCAS']:
    IndexEvent=Eventsynoptic.index[Eventsynoptic[event]==True]
    if event == 'Front':# specify which prefrontal or postfrontal
        IndexEvent=IndexEvent+ pd.DateOffset(-1)
        IndexEvent2=IndexEvent+ pd.DateOffset(-2)
        IndexEvent3=IndexEvent+ pd.DateOffset(-3)
        IndexEvent=IndexEvent+IndexEvent2+IndexEvent3
    var=np.array([])
    Wind_speed=np.array([])
    Wind_dir=np.array([])
    Norm=np.array([])
    Theta=np.array([])
    
    for rr in network:
        variable=rr.getvar('Rc mm')
        vel_10min=rr.getvar('Sm m/s').groupby(pd.TimeGrouper('20Min')).mean()
        dir_10min=rr.getvar('Dm G').groupby(pd.TimeGrouper('20Min')).mean()
        
        newvar=pd.Series()# select Index of Event (Should exist a better way)
        newvel=pd.Series()
        newdir=pd.Series()
        for i in IndexEvent.dayofyear:
            newvar=newvar.append(variable[variable.index.dayofyear==i])
            newvel=newvel.append(vel_10min[vel_10min.index.dayofyear==i])
            newdir=newdir.append(dir_10min[dir_10min.index.dayofyear==i])
        newvar=newvar.groupby(lambda t: (t.hour,t.minute)).sum()
        newvel=newvel.groupby(lambda t: (t.hour,t.minute)).mean()
        newdir=newdir.groupby(lambda t: (t.hour,t.minute)).mean()
        var=np.append(var,newvar.tolist())
        print var.shape
        Norm=np.append(Norm,newvel.tolist())
        Theta=np.append(Theta,newdir.tolist())

    FIG=LCBplot(rr)
    plt.figure(figsize=(FIG.getpara('wfig'),FIG.getpara('hfig')))
    plt.suptitle(FIG.getpara('subtitle'),fontsize=20)
    
    var=var.reshape(nbsta,720)
    V=np.cos(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!
    U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!

    U=U.reshape(nbsta,72)
    V=V.reshape(nbsta,72)
    
    var=var.transpose()
    U=U.transpose()
    V=V.transpose()

#  Interpolation

    newvar=np.array([[]])
    for i in np.arange(var.shape[0]):
        data=var[i,:]
        x=np.array(position)
        mask=~np.isnan(data)
        datamask=data[mask]
        positionmask=x[mask]
        try:
            f=interpolate.InterpolatedUnivariateSpline(positionmask,datamask,k=1)
            newvar=np.append(newvar,f(x))
        except:
            print('Cant interpolate - Therfore let NAN data')
            newvar=np.append(newvar,data)
    
    newvar=newvar.reshape(720,nbsta)
    var=newvar


    U.shape
    V.shape
    var.shape
    Position.shape
    Time.shape
    Levels=np.linspace(0,5,100)
    #Levels=np.linspace(-0.1,0.1,30)
    cmap = plt.cm.get_cmap("RdBu_r")
    plt.contourf(Position,Time,var,levels=Levels,cmap=cmap)    
    plt.colorbar()
    a=plt.quiver(Position[::10,::],Time[::10,::],U[:,:],V[:,:],scale=35)
    #plt.gca().invert_xaxis()    


    l,r,b,t = plt.axis()
    dx, dy = r-l, t-b
    plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

    plt.savefig(str(out)+str(event)+'-hovermoler.png')
    plt.close()


#===============================================================================
# Hovmoller Station - looping over days + interpolation + GFS wind + RAIN
#===============================================================================
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

#------------------------------------------------------------------------------ 
# Select network stations
PosSta=pos_sta()
Sorted=PosSta.sortsta(PosSta.stations('Head'),'Lon')
position=Sorted['position']
staname=Sorted['staname']
nbsta=len(staname)

#------------------------------------------------------------------------------ 
# Select corresponding files
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/'

# Find all the clima and Hydro
#Files=glob.glob(InPath+"*")

network=[]
net=LCB_net()
for sta in staname:
    File=InPath+sta+'clear_merge.TXT'
    print(File)
    rr=LCB_station(File)
    network.append(rr)
    net.add(rr)


time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30

Init=pd.date_range('09/01/2014','04/10/2015', freq='D')# Month- Day - Years
End=pd.date_range('09/01/2014 23:59:59','04/10/2015 23:59:59', freq='D')

# Init=pd.date_range('09/09/2014','09/10/2014', freq='D')# Month- Day - Years
# End=pd.date_range('09/09/2014 23:59:59','09/10/2014 23:59:59', freq='D')


for ini,end in zip(Init,End):
    var=np.array([])
    Wind_speed=np.array([])
    Wind_dir=np.array([])
    Norm=np.array([])
    Theta=np.array([])
    for rr in network:
        print(rr.getpara('InPath'))
        rr.setpara('From',ini)
        rr.setpara('To',end)
        variable=rr.getvar('Ta C')
        print(variable)
        vel_10min=rr.getvar('Sm m/s').groupby(pd.TimeGrouper('10Min')).mean()
        dir_10min=rr.getvar('Dm G').groupby(pd.TimeGrouper('10Min')).mean()
        var=np.append(var,variable.tolist())
        Norm=np.append(Norm,vel_10min.tolist())
        Theta=np.append(Theta,dir_10min.tolist())


    FIG=LCBplot(rr)
    fig=plt.figure(figsize=(FIG.getpara('wfig'),FIG.getpara('hfig')))
    f, (ax1, ax2, ax3) =plt.subplots(1, 3, sharey=True,figsize=(FIG.getpara('wfig'),FIG.getpara('hfig')))


#     gs1 = plt.GridSpec(1, 2)
#     ax1 = fig.add_subplot(gs1[0])
#     ax2 = fig.add_subplot(gs1[1])

#     ax1 = plt.subplot2grid((3,3), (0,0), rowspan=3,colspan=1)
#     ax2 = plt.subplot2grid((3,3), (0,2), rowspan=3,colspan=2)

    plt.suptitle(FIG.getpara('subtitle'),fontsize=20)

    var=var.reshape(nbsta,720)
    V=np.cos(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!
    U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!

    U=U.reshape(nbsta,144)
    V=V.reshape(nbsta,144)
    
    var=var.transpose()
    U=U.transpose()
    V=V.transpose()

#  Interpolation

    newvar=np.array([[]])
    for i in np.arange(var.shape[0]):
        data=var[i,:]
        x=np.array(position)
        mask=~np.isnan(data)
        datamask=data[mask]
        positionmask=x[mask]
        try:
            f=interpolate.InterpolatedUnivariateSpline(positionmask,datamask,k=1)
            newvar=np.append(newvar,f(x))
        except:
            print('Cant interpolate - Therfore let NAN data')
            newvar=np.append(newvar,data)
    
    newvar=newvar.reshape(720,nbsta)
    var=newvar


    U.shape
    V.shape
    var.shape
    Position.shape
    Time.shape
    Levels=np.linspace(5,28,100)
    #Levels=np.linspace(-0.1,0.1,30)
    cmap = plt.cm.get_cmap("RdBu_r")
#------------------------------------------------------------------------------ 
    A=ax3.contourf(Position,Time,var,levels=Levels,cmap=cmap)    
    plt.colorbar(A,ax=ax3)
#    a=plt.quiver(Position[::10,::],Time[::10,::],U[:,:],V[:,:],scale=35)
    a=plt.quiver(Position[::5,::],Time[::5,::],U[:,:],V[:,:],scale=35)
    qk1 = plt.quiverkey(a, 1.3, 1, 2, r'$2 \frac{m}{s}$', labelpos='W',
           fontproperties={'weight': 'bold'})
    plt.gca().invert_xaxis()

    l,r,b,t = plt.axis() # dosent work with subplot()
    dx, dy = r-l, t-b
    plt.axis([l-0.2*dx, r+0.2*dx, b-0.05*dy, t+0.05*dy])
    plt.gca().invert_xaxis()    

#------------------------------------------------------------------------------ 
    #B=ax2.quiver([0,0,0,0],[3,9,15,21],U_gfs[ini:end],V_gfs[ini:end],scale=35)
#     ax1.axis('off') # remove axis
    #qk2 = plt.quiverkey(B, 1.1, 1, 2, r'$2 \frac{m}{s}$', labelpos='W',
    #           fontproperties={'weight': 'bold'})
#     B=ax2.contourf(Position[:,0],Time[::180,0],var,levels=Levels,cmap=cmap)    
#    plt.colorbar(B,ax=ax2)
#     a=plt.quiver(Position[::5,::],Time[::5,::],U[:,:],V[:,:],scale=35)
    #plt.gca().invert_xaxis()
#------------------------------------------------------------------------------ 
    rain=rr.getvar('Rc mm').groupby(pd.TimeGrouper('1H')).mean()
    time=np.arange(0,24)
    ax1.barh(np.array(time),np.array(rain))


# rain.plot(kind='barh',yticks=rain.index.hour)
# plt.savefig('test.png')


    plt.savefig(str(out)+str(ini)+'-hovermoler.png',bbox_inches='tight') # reduire l'espace entre les grafiques
    plt.close()



#===============================================================================
# Hovmoller Station - looping over days + interpolation 
#===============================================================================

#------------------------------------------------------------------------------ 
# Select network stations
PosSta=att_sta()
Sorted=PosSta.sortsta(PosSta.stations(['Head']),'Lon')
position=Sorted['position']
staname=Sorted['staname']
nbsta=len(staname)

#------------------------------------------------------------------------------ 
# Select corresponding files
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/'

# Find all the clima and Hydro
#Files=glob.glob(InPath+"*")

network=[]

for sta in staname:
    File=InPath+sta+'clear_merge.TXT'
    print(File)
    rr=LCB_station(File)
    network.append(rr)


time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30

Init=pd.date_range('09/01/2014','07/01/2015', freq='D')# Month- Day - Years
End=pd.date_range('09/01/2014 23:59:59','07/01/2015 23:59:59', freq='D')

for ini,end in zip(Init,End):
    var=np.array([])
    Wind_speed=np.array([])
    Wind_dir=np.array([])
    Norm=np.array([])
    Theta=np.array([])

    for rr in network:
        print(rr.getpara('InPath'))
        rr.setpara('From',ini)
        rr.setpara('To',end)
        variable=rr.getvar('Ta C')
        print(variable)
        vel_10min=rr.getvar('Sm m/s').groupby(pd.TimeGrouper('10Min')).mean()
        dir_10min=rr.getvar('Dm G').groupby(pd.TimeGrouper('10Min')).mean()
        var=np.append(var,variable.tolist())
        Norm=np.append(Norm,vel_10min.tolist())
        Theta=np.append(Theta,dir_10min.tolist())

    FIG=LCBplot(rr)
    plt.figure(figsize=(FIG.getpara('wfig'),FIG.getpara('hfig')))
    plt.suptitle(FIG.getpara('subtitle'),fontsize=20)
    
    var=var.reshape(nbsta,720)
    V=np.cos(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!
    U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!

    U=U.reshape(nbsta,144)
    V=V.reshape(nbsta,144)
    
    var=var.transpose()
    U=U.transpose()
    V=V.transpose()

#  Interpolation

    newvar=np.array([[]])
    for i in np.arange(var.shape[0]):
        data=var[i,:]
        x=np.array(position)
        mask=~np.isnan(data)
        datamask=data[mask]
        positionmask=x[mask]
        try:
            f=interpolate.InterpolatedUnivariateSpline(positionmask,datamask,k=1)
            newvar=np.append(newvar,f(x))
        except:
            print('Cant interpolate - Therfore let NAN data')
            newvar=np.append(newvar,data)
    
    newvar=newvar.reshape(720,nbsta)
    var=newvar


    U.shape
    V.shape
    var.shape
    Position.shape
    Time.shape
    Levels=np.linspace(5,28,100)
    #Levels=np.linspace(-0.1,0.1,30)
    cmap = plt.cm.get_cmap("RdBu_r")
    plt.contourf(Position,Time,var,levels=Levels,cmap=cmap)    
    plt.colorbar()
    a=plt.quiver(Position[::5,::],Time[::5,::],U[:,:],V[:,:],scale=35)
    #plt.gca().invert_xaxis()    


    l,r,b,t = plt.axis()
    dx, dy = r-l, t-b
    plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

    plt.savefig(str(out)+str(ini)+'-hovermoler.png')
    plt.close()


#===============================================================================
# Hovmoller Station - looping over days + interpolation + Rainfall grafic
#===============================================================================
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

#------------------------------------------------------------------------------ 
# Select network stations
PosSta=att_sta()
Sorted=PosSta.sortsta(PosSta.stations(['Medio']),'Lon')
position=Sorted['position']
staname=Sorted['staname']
nbsta=len(staname)

#------------------------------------------------------------------------------ 
# Select corresponding files
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/'

# Find all the clima and Hydro
#Files=glob.glob(InPath+"*")

network=[]
net=LCB_net()

for sta in staname:
    File=InPath+sta+'clear_merge.TXT'
    print(File)
    rr=LCB_station(File)
    network.append(rr)
    net.add(rr)


time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30

Init=pd.date_range('04/01/2015','07/01/2015', freq='D')# Month- Day - Years
End=pd.date_range('04/01/2015 23:59:59','07/01/2015 23:59:59', freq='D')

for ini,end in zip(Init,End):
    var=np.array([])
    Wind_speed=np.array([])
    Wind_dir=np.array([])
    Norm=np.array([])
    Theta=np.array([])

    for rr in network:
        print(rr.getpara('InPath'))
        rr.setpara('From',ini)
        rr.setpara('To',end)
        variable=rr.getvar('Ta C')
        print(variable)
        vel_10min=rr.getvar('Sm m/s').groupby(pd.TimeGrouper('10Min')).mean()
        dir_10min=rr.getvar('Dm G').groupby(pd.TimeGrouper('10Min')).mean()
        var=np.append(var,variable.tolist())
        Norm=np.append(Norm,vel_10min.tolist())
        Theta=np.append(Theta,dir_10min.tolist())

    FIG=LCBplot(rr)
    plt.figure(figsize=(FIG.getpara('wfig'),FIG.getpara('hfig')))
    plt.suptitle(FIG.getpara('subtitle'),fontsize=20)
    
    var=var.reshape(nbsta,720)
    V=np.cos(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!
    U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!

    U=U.reshape(nbsta,144)
    V=V.reshape(nbsta,144)
    
    var=var.transpose()
    U=U.transpose()
    V=V.transpose()

#  Interpolation

    newvar=np.array([[]])
    for i in np.arange(var.shape[0]):
        data=var[i,:]
        x=np.array(position)
        mask=~np.isnan(data)
        datamask=data[mask]
        positionmask=x[mask]
        try:
            f=interpolate.InterpolatedUnivariateSpline(positionmask,datamask,k=1)
            newvar=np.append(newvar,f(x))
        except:
            print('Cant interpolate - Therfore let NAN data')
            newvar=np.append(newvar,data)
    
    newvar=newvar.reshape(720,nbsta)
    var=newvar


    U.shape
    V.shape
    var.shape
    Position.shape
    Time.shape
    Levels=np.linspace(5,28,100)
    #Levels=np.linspace(-0.1,0.1,30)
    cmap = plt.cm.get_cmap("RdBu_r")
    plt.contourf(Position,Time,var,levels=Levels,cmap=cmap)    
    plt.colorbar()
    a=plt.quiver(Position[::5,::],Time[::5,::],U[:,:],V[:,:],scale=35)
    #plt.gca().invert_xaxis()    


    l,r,b,t = plt.axis()
    dx, dy = r-l, t-b
    plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

    plt.savefig(str(out)+str(ini)+'-hovermoler.png')
    plt.close()


#===============================================================================
# Hovmoller Station - looping over days
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
out='/home/thomas/'

# Find all the clima and Hydro
Files=glob.glob(InPath+"*")

Files=[
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C09clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C08clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C07clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C06clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C05clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C04clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C10clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C11clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C12clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C13clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C14clear_merge.TXT',
'/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/C15clear_merge.TXT'
 ]

network=[]
for i in Files:
    print(i)
    rr=LCB_station(i)
    network.append(rr)



# Select network stations
PosSta=att_sta()
Sorted=PosSta.sortsta(PosSta.stations(['Head']),'Lon')
position=Sorted['position']
staname=Sorted['staname']
nbsta=len(staname)



time=range(0,720,1)#
Position, Time = np.meshgrid(position, time)
Time=Time/30

Init=pd.date_range('03/01/2015','07/01/2015', freq='D')
End=pd.date_range('03/01/2015 23:59:59','07/01/2015 23:59:59', freq='D')

for ini,end in zip(Init,End):
    var=np.array([])
    Wind_speed=np.array([])
    Wind_dir=np.array([])
    Norm=np.array([])
    Theta=np.array([])
    #FIG=LCBplot()
    plt.figure(figsize=(FIG.getpara('wfig'),FIG.getpara('hfig')))
    plt.suptitle(FIG.getpara('subtitle'),fontsize=20)
    for rr in network:
        print(rr.getpara('InPath'))
        rr.setpara('From',ini)
        rr.setpara('To',end)
        variable=rr.getvar('Ta C')
        print(variable)
        vel_10min=rr.getvar('Sm m/s').groupby(pd.TimeGrouper('10Min')).mean()
        dir_10min=rr.getvar('Dm G').groupby(pd.TimeGrouper('10Min')).mean()
        var=np.append(var,variable.tolist())
        Norm=np.append(Norm,vel_10min.tolist())
        Theta=np.append(Theta,dir_10min.tolist())


    var=var.reshape(nbsta,720)
    V=np.cos(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!
    U=np.sin(map(math.radians,Theta+180))*Norm# V AND U ARE WRONG BUT THEY ARE DISPLAY COORECTLY IN THE HOVERMOLLERRRRR !!!!!!!!


    U=U.reshape(nbsta,144)
    V=V.reshape(nbsta,144)

    var=var.transpose()
    U=U.transpose()
    V=V.transpose()

    U.shape
    V.shape
    var.shape
    Position.shape
    Time.shape
    Levels=np.linspace(5,35,100)
    #Levels=np.linspace(-0.1,0.1,30)
    cmap = plt.cm.get_cmap("RdBu_r")
    plt.contourf(Position[:,:],Time[:,:],var[:,:],levels=Levels,cmap=cmap)    
    plt.colorbar()
    a=plt.quiver(Position[::5,::],Time[::5,::],U[:,:],V[:,:],scale=35)
    #plt.gca().invert_xaxis()    


    l,r,b,t = plt.axis()
    dx, dy = r-l, t-b
    plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

    plt.savefig(str(out)+str(ini)+'-hovermoler.png')
    plt.close()


    U.shape
    V.shape
    var.shape
    Position.shape
    Time.shape
    Levels=np.linspace(5,35,100)
    #Levels=np.linspace(-0.1,0.1,30)
    cmap = plt.cm.get_cmap("RdBu_r")
    plt.contourf(Position[:,:],Time[:,:],var[:,:],levels=Levels,cmap=cmap)    
    plt.colorbar()
    a=plt.quiver(Position[::5,::],Time[::5,::],U[:,:],V[:,:],scale=35)
    #plt.gca().invert_xaxis()    


    l,r,b,t = plt.axis()
    dx, dy = r-l, t-b
    plt.axis([l-0.2*dx, r+0.2*dx, b-0*dy, t+0*dy])

    plt.savefig(str(out)+str(ini)+'-hovermoler.png')
    plt.close()


