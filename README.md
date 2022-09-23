# -Validation-of-remote-sensing-rainfall-data-using-a-network-of-ground-measurements
Validate two sets of remote sensing rainfall data TRMM and CHIRPS with rain gauge stations and choose the most fitting product per rainfall station.
Remote Sensing (RS) is an attractive new type of data source, especially in data-scarce 
regions. Several dedicated satellite missions exist to measure rainfall since the 80s. Often, 
this data is provided free of charge at daily or monthly resolution. Moreover, remote 
sensing data provides spatial-temporal information which can be used in combination 
ground measurements. In this group project you will work with two specific RS rainfall 
products: CHIRPS and TRMM. For more information of the datasets please visit the 
websites: https://trmm.gsfc.nasa.gov/ , https://www.chc.ucsb.edu/data/chirps 

Important: To try the python code, is to uncompress the .rar files, and have to be saved i the same directory. the latter is just a recomendation.
Since some libraries are dependent on others it is necessary to follow the documentation of the libraries. The most important were GDAL and Geopandas. These libraries were chosen because it was required to work with extensive geospatial data.
Geopandas is an open source project that allows python to work in an easier way with geospatial data. This library has the following required dependencies:
1: Numpy
2: Pandas
3: Shapely
4: Fiona
5: Pyproj
This code has three main parts. The first and second parts extract the information from the observed and RS data, and the third part is the statistical analysis
