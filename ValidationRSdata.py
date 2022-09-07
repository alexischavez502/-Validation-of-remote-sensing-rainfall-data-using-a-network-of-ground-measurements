# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:59:52 2022


@authors: Buckstöver Mira, Cardenas Carlos, Chavez Alexis, Rodriguez Erika 
"""

import glob
import pandas as pd
import csv
import fiona
from collections import OrderedDict
import geopandas as gpd
import os
import rasterio
import scipy.sparse as sparse
import numpy as np
from osgeo import osr
import matplotlib.pyplot as plt
import math

crs = osr.SpatialReference()       ###
crs.SetFromUserInput("EPSG:4326")  ###
wgs84 = crs.ExportToProj4()        ###

#OPEN DIRECTORY
pathcsv = r'D:\KU LEUVEN\Courses\Environmental Programming\Project\Prueba final\csv_files'  #change directory
pathTRMM=(r'D:\KU LEUVEN\Courses\Environmental Programming\Project\Prueba final\TRMM')
pathCHIRPS=(r'D:\KU LEUVEN\Courses\Environmental Programming\Project\Prueba final\CHIRPS')
pathOut=(r'D:\KU LEUVEN\Courses\Environmental Programming\Project\Prueba final\Output') #path for  outputs (figures and csv files)
all_files = glob.glob(pathcsv + "/*.csv")

#PROCESSING THE OBSERVED DATA
#following code taken from: https://riptutorial.com/pandas/example/6438/create-a-sample-dataframe-with-datetime
#only takes the data of the observed stations between 2000-01-01 and 2015-12-01 into account, as TRMM and CHIRPS are only available in this time 
rng = pd.date_range('2000-01-01', periods=192, freq='MS')  #MS is for the 1day of month, periods change according to data
df_Obs = pd.DataFrame(index=rng)

coord =[]
Obsdata=pd.DataFrame()

for filename in all_files:
    sname=filename.split('\\')#name of station 
    sname=sname[len(sname)-1]
    sname=sname.split('.')[0]
    #sname=sname.replace(' ', '_')
    df = pd.read_csv(filename, dtype='str', sep=';') #take first line of csv file
    #print(df) #just to check!
    dfcol=df.columns[0:4]
    data=[sname, dfcol[1], dfcol[3]] #only taking the latitude and longitude 
    coord.append(data) #list of data
    Obsdata=df
    Obsdata.rename(columns=Obsdata.iloc[0], inplace=True)
    Obsdata=Obsdata[['datetime']+['data']]
    Obsdata=Obsdata.iloc[1:]
    Obsdata.index=pd.to_datetime(Obsdata['datetime'])
    Obsdata=Obsdata.drop(columns='datetime')
    Obsdata=Obsdata.astype(float)
    Obsdata.columns=[sname]  #rename column
    df_Obs[sname]= Obsdata[sname].where(Obsdata.index>='2000-01-01')
        
# following code taken from: https://pythonguides.com/python-write-a-list-to-csv/ 
with open("coordinates.csv", "w", newline='') as f: #save coordinates of stations to csv
    writer=csv.writer(f)
    writer.writerow(['station', '1', '2'])
    writer.writerows(coord)

#from https://youtu.be/lzg3S2ecPww
coord=pd.DataFrame(coord)               #chaning from a list to a DataFrame
coord=coord.rename(columns = {0:'station'}) #changing name of column 
coord=coord.rename(columns = {1:'lat'}) #changing name of column 
coord['lat'] = coord['lat'].astype(float) # changing from string to float
coord=coord.rename(columns = {2:'lon'})
coord['lon'] = coord['lon'].astype(float)
schema={'properties': OrderedDict([('station', 'str'), ('lon', 'str'), ('lat', 'str')]), 'geometry': 'Point'}
# definition of shape file, composed of Station name as a str, geometry is set to point

coordShp=fiona.open('Coord.shp', mode='w', crs=wgs84, schema=schema, driver='ESRI Shapefile')
#open a shape file 


for index, line in coord.iterrows(): #going through DataFrame line by line
    feature={
        'geometry':{'type':'Point', 'coordinates': (line.lon, line.lat)},#coordinates are lon and lat
        'properties': OrderedDict([('station', line['station']),('lon', line['lon']), ('lat', line ['lat'])])
        }
    coordShp.write(feature) # write to shapefile 
    
coordShp.close()

print("Coordinates for stations found")    

### TRMM AND CHIRPS PROCESSING
#Reference from GeoDeltaLabs: https://www.youtube.com/watch?v=6zzneGT4mkg&ab_channel=GeoDeltaLabs
#table dataframe empty
table= pd.DataFrame(index=np.arange(0,1))

#Read the points shape files using goepandas
stations= gpd.read_file(r'Coord.shp') 
#stations['lon']= stations['geometry'].x
#stations['lat']= stations['geometry'].y

MatrixTRMM= pd.DataFrame()  #For all raster
df_TRMM=pd.DataFrame()    #For the extracted values
df_CHIRPS=pd.DataFrame()
MatrixCHIRPS=pd.DataFrame()

#ITERATIVE READING RASTERS AND SAVE DATA AS MATRIX
def method(path, Matrix, df_kind):
    for files in os.listdir(path): #Change directory
        if files[-4: ]=='.tif':    #only read files .tif; [-4:] means the last four characters
            dataMatrix=rasterio.open(path + '\\' + files)
            data_Array=dataMatrix.read(1)
            array_sparse= sparse.coo_matrix(data_Array, shape=[data_Array.shape]) #size of the Array, change data according to use, Sparse matrices can be used in arithmetic operations
            data=files[33:-4]
            Matrix[data] = array_sparse.toarray().tolist()  #inserting the data into a Matrix
        #print ('Processing done for raster: '+files[33:-4])  #for check!
           
    for index, row in stations.iterrows():
         station_name=str(row['station'])
         #print(station_name)
         lon= float(row['lon'])
         lat= float(row['lat'])
         x,y=(lon,lat)
         row, col= dataMatrix.index(x,y)  #extractting data from pixel
         #print('Processing: ' + station_name)
     
     # rainfall data value from each stored raster array and record it into varaible table
         for records_date in Matrix.columns.tolist():
             a= Matrix[records_date]
             rainfall_v= a.loc[int(row)][int(col)] #loc is to locate the correct row and column in an array
             #print(rainfall_v)
             table[records_date]= rainfall_v #prints the table of each station
             transpose_Mat= table.T  #Transpose the matrix to have the dates as an index
             transpose_Mat=transpose_Mat.rename(columns = {0:station_name})
         df_kind[station_name]=transpose_Mat[station_name]             
             #transpose_Mat.rename(columns={0:'Rainfall[mm]'}, inplace=True)
             #transpose_Mat.to_csv(r'D:\KU LEUVEN\Courses\Environmental Programming\Project\Prueba 2\Salida' + '\\' +station_name+ kind + '.csv' )
    df_kind.index=pd.to_datetime(df_kind.index)
    df_kind=df_kind
    return df_kind
    
df_TRMM=method(pathTRMM, MatrixTRMM, df_TRMM)
print('TRMM processing finished')
df_CHIRPS=method(pathCHIRPS, MatrixCHIRPS, df_CHIRPS)
print('CHIRPS processing finished')

#CHOOSE STATION FOR EVALUATION
user_station=input("Choose your station for the evaluation:")

while user_station not in df_Obs: # checking if station is there 
    # taken from: https://www.askpython.com/python/examples/in-and-not-in-operators-in-python
    print("Station not found. Please try again!")
    user_station=input("Choose your station for the evaluation:")
    
#GRAPHICAL EVALUATION FOR THE REMOTE SENSING TIME SERIES AGAINST OBSERVED DATA
#TIME SERIES PLOT
plt.plot(df_TRMM[user_station], label='TRMM Precipitation')
plt.plot(df_CHIRPS[user_station], label='CHIRPS Precipitation')
plt.plot(df_Obs[user_station], label= 'Observed Precipitation',linestyle='--')
plt.title(user_station)
plt.xlabel('Time (month)')
plt.ylabel('Precipitation (mm)')
plt.legend()
plt.savefig(pathOut+'\ ' + user_station + '_' + 'monthly_data'+'.png')
plt.show()
plt.clf()

#COMPARISON OBSERVED VS REMOTE SENSING FOR PRECIPITATION
df_weigth=pd.DataFrame()

#COMPARISON OBSERVED VS REMOTE SENSING FOR PRECIPITATION
for index, row in stations.iterrows():
    station_name=str(row['station'])  #row from the column station
    def correlation(stat, satelite):
        x, Op, Op2, length, PO_av, POi, satelite_BIAS, satelite_RMSE, satelite_NS= (0,0, 0, 0, 0, 0, 0, 0, 0)#all variables set to 0 bevor next calculation
        PS=satelite[stat] 
        PO=df_Obs[stat]
        f=len(PO) 
    
        while x < f-1: #calculation of the average of the observed data (needed for the calculation of NS)
            if PS[x]>= 0.0 and PO[x] >= 0.0:
                PO_av=PO_av+PO[x]
                length=length+1
                x=x+1
            else:
                x=x+1
        PO_av=PO_av/length
        x, length=(0,0) #setting x and length back to 0 bevor the next loop 

        while x < f-1:
            if PS[x] >= 0.0 and PO[x] >= 0.0:
                Diff=PO[x]-PS[x]
                Op=Op+Diff
                Op2=Op2+Diff**2
                Diff2=PO[x]-PO_av
                POi=POi+Diff2**2
                length=length+1
                x=x+1
            else: 
                x=x+1
        satelite_BIAS=round(Op/length ,4) #calculation of BIAS
        satelite_RMSE=round(math.sqrt((Op2/length)) , 4) #calculation of RMSE
        satelite_NS=round(1-(Op2/POi), 4)#calculation of NS
        #print('BIAS=', satelite_BIAS, '\nRMSE=', satelite_RMSE,'\nNS=', satelite_NS) #\n is for new line in text 
        return satelite_RMSE, satelite_BIAS, satelite_NS
    [TRMM_RMSE, TRMM_BIAS, TRMM_NS]=correlation(station_name, df_TRMM)#station and methode used can be changed here[CHIRPS_RMSE, CHIRPS_BIAS, CHIRPS_NS]=correlation(station_name, CHIRPS)
    [CHIRPS_RMSE, CHIRPS_BIAS, CHIRPS_NS]=correlation(station_name, df_CHIRPS)#station and methode used can be changed here[CHIRPS_RMSE, CHIRPS_BIAS, CHIRPS_NS]=correlation(station_name, CHIRPS)
    df_weigth[station_name]=[TRMM_BIAS, TRMM_RMSE, TRMM_NS, CHIRPS_BIAS, CHIRPS_RMSE, CHIRPS_NS]
    #gives out a dataframe with all the correlation coefficients for all stations 

df_weigth.index=['TR_BIAS', 'TR_RMSE', 'TR_NS', 'CHI_BIAS', 'CHI_RMSE', 'CHI_NS' ]
#letting the user chose the weigthing factors for the correlation 
wei_BIAS=float(input('Enter weighting factor for BIAS in %:'))
wei_RMSE =float(input('Enter weighting factor for RMSE in %:'))
wei_NS =float(input('Enter weighting factor for NS in %:'))

while round(wei_BIAS+wei_RMSE+wei_NS,0)!=100: #makes sure that the weighting factors add up to 100%
    print("The sum of the weighting factors must be 100%")
    wei_BIAS=float(input('Enter weighting factor for BIAS in %:'))
    wei_RMSE =float(input('Enter weighting factor for RMSE in %:'))
    wei_NS =float(input('Enter weighting factor for NS in %:'))
    
BIAS_TRMM=df_weigth[user_station].iloc[0]
RMSE_TRMM=df_weigth[user_station].iloc[1]
NS_TRMM=df_weigth[user_station].iloc[2]
BIAS_CHIRPS=df_weigth[user_station].iloc[3]
RMSE_CHIRPS=df_weigth[user_station].iloc[4]
NS_CHIRPS=df_weigth[user_station].iloc[5]

#CALCULATIONS FOR THE WEIGHTED OUTPUT for TRMM and CHIRPS for the chosen station
#wei closer to 0 the better the correlation 
wei_TRMM=round(BIAS_TRMM*(wei_BIAS/100)+RMSE_TRMM*(wei_RMSE/100)+(1-NS_TRMM)*(wei_NS/100), 4) 
wei_CHIRPS=round(BIAS_CHIRPS*(wei_BIAS/100)+RMSE_CHIRPS*(wei_RMSE/100)+(1-NS_CHIRPS)*(wei_NS/100), 4)
print('Weighted evaluation for TRMM:', wei_TRMM, '\nWeighted evaluation for CHRÌRPS:',wei_CHIRPS)
if wei_TRMM<wei_CHIRPS: #printing out which RM is working better for the chosen station
    print('TRMM is better than CHIRPS for ' + user_station)
else:
    print('CHIRPS is better than TRMM for ' + user_station)
# GRAPHICAL PLOTS
def plotscatter(sat, sat_str, BIAS, RMSE, NS):   #satelite, string name satelite, 
    x=sat[user_station] #Chose satelite
    y=df_Obs[user_station]
    msk=-np.isnan(y) #to avoid nan values
    plt.title(user_station)
    fitt=np.polyfit(x[msk],y[msk],1) # to calculate trend of only paired data
    poly1d_fn=np.poly1d(fitt)
    plt.plot(x[msk],y[msk] ,'ro',markersize=5 ) #scatter plot
    plt.plot(x,poly1d_fn(x),'--',  color='k', lw=1,label='y=%.6fx+%.6f'% (fitt[0],fitt[1])) #trend line
    plt.xlabel(sat_str+' '+'Precipitation [mm/month]')
    plt.ylabel('Observed Precipitation [mm/month]')
    plt.suptitle("BIAS= " + str(BIAS) + "   "+"RMSE= " + str(RMSE)+"   "+"NS= " + str(NS), fontsize=9) #Change satelite
    #print(fitt)
    #45 degree line
    x = y = plt.xlim() #limit axis
    plt.plot(x, y, linestyle='--', color='b', lw=2, scalex=False, scaley=False, label='Bisector') #scales the limits of the graph
    plt.legend()
    plt.savefig(pathOut + '\ ' + user_station+'_'+sat_str+'.png')
    plt.show()
    plt.clf()

plotscatter(df_TRMM, "TRMM", BIAS_TRMM, RMSE_TRMM, NS_TRMM)
plotscatter(df_CHIRPS, "CHIRPS", BIAS_CHIRPS, RMSE_CHIRPS, NS_CHIRPS)

#OUTPUT DATA
df_TRMM.to_csv(pathOut + '\\' + 'df_TRMM'+'.csv')
df_CHIRPS.to_csv(pathOut + '\\' +'df_CHIRPS'+'.csv') 
df_Obs.to_csv(pathOut + '\\' +'df_Observed'+'.csv')