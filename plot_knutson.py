import sys
#print(sys.path)
import argparse
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime as dt
import cftime
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

def plot_shapes(levels=True,max_level=5):

	ax = plt.axes(projection=ccrs.Robinson())
	
	if levels:
		
		for level in range(max_level+1):
			fname = r'/Users/shrutinath/Attribution/gadm36_levels_shp/gadm36_%i.shp'%level

			shape_feature = ShapelyFeature(Reader(fname).geometries(),
											ccrs.PlateCarree(), edgecolor='black')
			ax.add_feature(shape_feature)
	else:
		fname = r'/Users/shrutinath/Attribution/TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp'

		shape_feature = ShapelyFeature(Reader(fname).geometries(),
										ccrs.PlateCarree(), edgecolor='black')
		ax.add_feature(shape_feature)
	
	
	plt.show()

def in_gridcell(LON, LAT,df, degree=1):
    df = df[
        (df['lon']>=LON-degree*0.5) & 
        (df['lat']>=LAT-degree*0.5) & 
        (df['lon']<LON+degree*0.5) & 
        (df['lat']<LAT+degree*0.5)
    ]
    return df.shape[0]

def get_datasets():
	
	##get datasets for attribution
	df_1901_2010=xr.open_dataset('9cat.190101-201012.nc').roll(LON=72, roll_coords=True)
	df_1951_2010=xr.open_dataset('9cat.195101-201012.nc').roll(LON=72, roll_coords=True)
	df_1981_2010=xr.open_dataset('9cat.198101-201012.nc').roll(LON=72, roll_coords=True)
	
	##get datasets for Max's stuff
	df_papers = pd.read_csv('places_drivers_prediction.csv')
	precip = df_papers[df_papers['6 - Precipitation - lower_pred']>0.3]

	n_studies=np.zeros([len(df_1901_2010.LON.values),len(df_1901_2010.LAT.values)])
	
	
	##plot Max's datasets
	for i_y,i_lon in enumerate(df_1901_2010.LON.values):
		for i_x,i_lat in enumerate(df_1901_2010.LAT.values):
			if i_lon>180:
				i_lon=i_lon-360
			n_studies[i_y,i_x]=in_gridcell(i_lon, i_lat, precip, degree=2.5)
		
	n_studies[n_studies==0]=np.nan
  
	##plot attribution datasets
	
	arr_1901_2010=np.full([72, 144],np.nan)

	arr_1901_2010[(df_1901_2010.CP4.values==4).reshape(72, 144)]=3
	arr_1901_2010[(df_1901_2010.CP3.values==3).reshape(72, 144)]=3
	arr_1901_2010[(df_1901_2010.CP2.values==2).reshape(72, 144)]=2
	arr_1901_2010[(df_1901_2010.CP1.values==1).reshape(72, 144)]=1
	arr_1901_2010[(df_1901_2010.CP0.values==0).reshape(72, 144)]=0
	arr_1901_2010[(df_1901_2010.CM4.values==-4).reshape(72, 144)]=-3
	arr_1901_2010[(df_1901_2010.CM3.values==-3).reshape(72, 144)]=-3
	arr_1901_2010[(df_1901_2010.CM2.values==-2).reshape(72, 144)]=-2
	arr_1901_2010[(df_1901_2010.CM1.values==-1).reshape(72, 144)]=-1


	arr_1951_2010=np.full([72, 144],np.nan)

	arr_1951_2010[(df_1951_2010.CP4.values==4).reshape(72, 144)]=3
	arr_1951_2010[(df_1951_2010.CP3.values==3).reshape(72, 144)]=3
	arr_1951_2010[(df_1951_2010.CP2.values==2).reshape(72, 144)]=2
	arr_1951_2010[(df_1951_2010.CP1.values==1).reshape(72, 144)]=1
	arr_1951_2010[(df_1951_2010.CP0.values==0).reshape(72, 144)]=0
	arr_1951_2010[(df_1951_2010.CM4.values==-4).reshape(72, 144)]=-3
	arr_1951_2010[(df_1951_2010.CM3.values==-3).reshape(72, 144)]=-3
	arr_1951_2010[(df_1951_2010.CM2.values==-2).reshape(72, 144)]=-2
	arr_1951_2010[(df_1951_2010.CM1.values==-1).reshape(72, 144)]=-1

	arr_1981_2010=np.full([72, 144],np.nan)

	arr_1981_2010[(df_1981_2010.CP4.values==4).reshape(72, 144)]=3
	arr_1981_2010[(df_1981_2010.CP3.values==3).reshape(72, 144)]=3
	arr_1981_2010[(df_1981_2010.CP2.values==2).reshape(72, 144)]=2
	arr_1981_2010[(df_1981_2010.CP1.values==1).reshape(72, 144)]=1
	arr_1981_2010[(df_1981_2010.CP0.values==0).reshape(72, 144)]=0
	arr_1981_2010[(df_1981_2010.CM4.values==-4).reshape(72, 144)]=-3
	arr_1981_2010[(df_1981_2010.CM3.values==-3).reshape(72, 144)]=-3
	arr_1981_2010[(df_1981_2010.CM2.values==-2).reshape(72, 144)]=-2
	arr_1981_2010[(df_1981_2010.CM1.values==-1).reshape(72, 144)]=-1
	
	
	return df_papers, df_1901_2010, df_1951_2010, df_1981_2010, arr_1901_2010, arr_1951_2010, arr_1981_2010, n_studies
def plot_nstudies():
 df_1901_2010=xr.open_dataset('9cat.190101-201012.nc')#.roll(LON=72, roll_coords=True)

 cn=np.load('Grid_level_n_studies_attr.npy')
 
 lat_vals=np.arange(df_1901_2010.LAT.values.min()-1.25,df_1901_2010.LAT.values.max()+1.25,0.25)
 lon_vals=np.arange(df_1901_2010.LON.values.min()-1.25, df_1901_2010.LON.values.max()+1.25, 0.25)

 #print(cn.shape,lat_vals.shape,lat_vals)
 #print(cn[0,:,:][cn[0,:,:]!=0].shape)
 cn_dataset =  xr.DataArray(data=(cn[0,:,:]).T,dims=["LON", "LAT"], coords={"LON": lon_vals, "LAT": lat_vals})

 #cn_dataset = cn_dataset.roll(LON=720, roll_coords=True)
 ax = plt.axes(projection=ccrs.PlateCarree())
 ax.coastlines()
 mesh=ax.pcolormesh(cn_dataset.LON.values,cn_dataset.LAT.values,cn_dataset.values.T,cmap=plt.cm.get_cmap('cubehelix_r'), norm = mpl.colors.LogNorm(vmin=0.0001, vmax=0.3))
 plt.colorbar(mesh)
 plt.show()
 
 attr_dataset =  xr.DataArray(data=(cn[1,:,:]).T,dims=["LON", "LAT"], coords={"LON": lon_vals, "LAT": lat_vals})

 #cn_dataset = cn_dataset.roll(LON=720, roll_coords=True)
 ax = plt.axes(projection=ccrs.PlateCarree())
 ax.coastlines()
 mesh=ax.pcolormesh(attr_dataset.LON.values,attr_dataset.LAT.values,attr_dataset.values.T,cmap=plt.cm.get_cmap('cubehelix_r'))#, norm = mpl.colors.LogNorm(vmin=0.0001, vmax=0.3))
 plt.colorbar(mesh)
 plt.show()
 

#plot_nstudies()

