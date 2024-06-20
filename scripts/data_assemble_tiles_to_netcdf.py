import xarray as xr
import numpy as np
import os
import geopandas as gpd
import dask 

# import postProcessFunctions as myf
import myFunctions as myf
from dask.diagnostics import ProgressBar

''' -----
Script to patch and combine geotiffs organised by tiles to pre-defined AOI (in this case, antarctic sectors).
Data is saved as geotiff and netcdf per sector per year
Changes can be made to the area of interest

Author: M. Izeboud, Dec/2023, TU Delft
---------'''

''' -----
Set paths
---------'''

## Local
homedir = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/'
## VRlab
# homedir = '/net/labdata/maaike/'


''' --------------
Get Shapefiles 
------------------ '''
# geojson
gridTiles_geojson_path = os.path.join(homedir,'Data/tiles/gridTiles_iceShelves_EPSG3031.geojson')
gridTiles = gpd.read_file(gridTiles_geojson_path)


# ## redefined: SECTORS for AIS
sector_path = os.path.join(homedir, 'QGis/data_NeRD/AIS_outline_sectors.shp')
sector_poly = gpd.read_file(sector_path)
sector_ID_list = sector_poly['sector_ID'].to_list()
sector_ID_list.sort()
# print(sector_ID_list)
        

# years_list = ['2015']#,'2016','2017','2018','2019','2020','2021']

# variables_to_save = ['dmg095']
# # variables_to_save = ['dmg-25px']
# # variables_to_save = ['nodata']



''' --------------
Get iceshelves
------------------ '''
import glob 
path2iceshelves = os.path.join(homedir,'Data/Greene2022_AIS_coastlines/shapefiles/annual_iceshelf_polygons/revised_measures_greene/')

iceshelf_flist = glob.glob(path2iceshelves + '*.shp')
iceshelf_flist = [os.path.basename( filepath) for filepath in iceshelf_flist]
iceshelf_flist.sort()
iceshelf_flist

# annual ice shelves
iceshelf_df_1997 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_1997.75.shp' ) )
iceshelf_df_2015 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2015.2.shp' ) )
iceshelf_df_2016 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2016.2.shp' ) )
iceshelf_df_2017 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2017.2.shp' ) )
iceshelf_df_2018 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2018.2.shp' ) )
iceshelf_df_2019 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2019.2.shp' ) )
iceshelf_df_2020 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2020.2.shp' ) )
iceshelf_df_2021 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2021.2.shp' ) )

ishelf_dict = { '1997':iceshelf_df_1997,'2015':iceshelf_df_2015,
                '2016':iceshelf_df_2016,'2017':iceshelf_df_2017,
                '2018':iceshelf_df_2018,'2019':iceshelf_df_2019,
                '2020':iceshelf_df_2020,'2021':iceshelf_df_2021,
}


''' --------------
Select region/sector and corresponding tilenumbers to export

Available Sectors:
'ASE', 
'BSE', 
'EIS' 
'RS', 
'WIS'--> split in WIS-a and WIS-b
'WS' 

Variables to save:
'dmg','dmg-25px','nodata'
------------------ '''

# years_list = ['1997']
years_list = ['2015','2016','2017','2018','2019','2020','2021']
years_list = ['2021']


tilepath_dmg = os.path.join(homedir,'Data/S1_SAR/tiles/dmg_tiled/dmg095/') # DMG S1 annual
# tilepath_dmg = os.path.join(homedir,'Data/RAMP/RAMP_tiled/dmg_tiled/dmg095/') # DMG RAMP 1997
# tilepath_dmg = os.path.join(homedir,'Data/RAMP/RAMP_tiled_mamm/dmg095_tiled/') # DMG RAMP 2000

tilepath_rema = os.path.join(homedir,'Data/REMA/tiles/') # rema

tilepath_velo = os.path.join(homedir,'Data/ITS_LIVE/tiles/velocity/') # velocity annual

variables_to_save = ['dmg','rema']
paths2variables = [tilepath_dmg,tilepath_rema]
# variables_to_save = ['vx','vy']
# paths2variables = [tilepath_velo,tilepath_velo]

sector_ID_list.sort()

save_nc = True 

# set directory to save output
path2save = os.path.join(homedir,'Data/NERD/data_predictor/data_sector/') # save dir

# for varName, tilepath_in in zip(variables_to_save,paths2variables):
varName = 'dmg' ; tilepath_in = tilepath_dmg
for year in years_list:
    res='400m' # default
    if varName == 'dmg-25px':
        region_data = region_data.rename({'dmg-25px':'dmg'})
        res='1000m'
    if int(year) == 1997 or int(year) == 2000:
        res='1000m'
        
    if int(year) < 2015: 
        year_subdir=''
    else:
        if 'dmg' in varName:
            year_subdir = f'{year}-SON'
        elif 'v' in varName:
            year_subdir = year
        else: 
            year_subdir=''

    for sector_ID in sector_ID_list:

        ### Process Weddell Sea sector in 2parts or single part || only required for velocity variables
        # if any( [var in ['vx','vy'] for var in variables_to_save]): 
        if varName in ['vx','vy']:
            if 'EIS' in sector_ID or 'WIS' in sector_ID or 'WS' in sector_ID: 
                if not '-' in sector_ID:
                    print('Skip EIS/WIS/WS large-sector:', sector_ID)
                    continue 
        elif 'WIS' in sector_ID:
            if sector_ID not in ['WIS-a','WIS-b']:
                print('Skipping {} - Need to process WIS in A and B part (netcdf projction error otherwise)'.format(sector_ID))
                continue
        else:
            if '-' in sector_ID:
                print('Processing full sectors; skip part-sector ',sector_ID)
                continue   

        ## Make filename to save 
        # nc_base= 'data_sector-' + sector_ID + '_'+varName
        nc_filename = f'data_sector-{sector_ID}_{varName}_{year}.nc'

        ## Check if variable file already exsts
        if os.path.isfile( os.path.join( path2save, nc_filename ) ):
            print('Variable {} already saved for {} year {}'.format(varName,sector_ID, year))
            continue

        ''' --------------
        Define tileNumbers for selected region
        ------------------ ''' 
        ## select tiles
        tileNums_select = myf.get_tilelist_region(sector_poly, sector_ID, gridTiles=gridTiles)

        # Skip some tiles of FR and ROSS iceshelves that have nodata for S1 observations
        if not varName == 'nodata':
            tileNums_skip = [130,131,146,147,148,158,159,160,167,168,169, 170, 171,177,178,179,180,187,188,189,190,196,197,198,206,207,217,218] # for RV
            tileNums_skip = tileNums_skip + [62,63,70,71,72,78,79,80,86,87,88,89,95,96,97,98,105,106,107,108,115,116,117,118,132,133,134,135,136,137,138,149,150] # for FR
            tileNums_select = [tileNum for tileNum in tileNums_select if tileNum not in tileNums_skip]

        print('--- \nSelected {} sector; {} tiles'.format(sector_ID, len(tileNums_select)))


        print(f'.. loading data for {year}; {varName}')

        
        
        
        ''' --------------
        Load data
        ------------------ ''' 

        ## get all files in directory 
        # year_filelist = os.listdir(os.path.join(tilepath_in,year_subdir ))
        year_filelist = glob.glob(os.path.join(tilepath_in,year_subdir,'*.tif' ))
        year_filelist.sort()

        ## select tiles in region
        fnames_region = [fname for fname in year_filelist if int(fname.split('.')[0].split('tile_')[1]) in tileNums_select]
        filelist_region  = [ os.path.join(tilepath_in,year_subdir, fname) for fname in fnames_region ]

        if len(filelist_region) != len(tileNums_select):
            raise RuntimeError('Expected the same number of tiles({})/files({}) for sector {}'.format(len(tileNums_select),len(filelist_region),sector_ID))

        region_data = (xr.open_mfdataset( filelist_region,  
                    combine="nested", decode_times=False,
                    data_vars='minimal', 
                    coords= 'minimal', 
                    compat='no_conflicts', #  only values which are not null in both datasets must be equal. The returned dataset then contains the combination of all non-null values
                    chunks={'y':'auto','x':'auto','band':1}, # add chucnking info for dask
                    parallel=True,
                    ).isel(band=0).drop('band')
                    .transpose('y','x')
                    .rename({'band_data':varName})
        )

        print(f'.. loaded dataset {region_data.dims};')

        ''' ## Fill dmg NaN values as 0  '''
        if 'dmg' in varName:
            region_data = region_data.where(~np.isnan(region_data),other=0 ) # no-dmg = 0
            path2save = os.path.join(path2save,'damage095')

        ''' ## Set rema values < 0 to NaN '''
        if 'rema' in varName:
            region_data = region_data.where(region_data['rema']>0,np.nan) # rema: set data<0 to NaN:
            # save_nc_variable( path2data, 'rema', years,region_ID,  region_ds_rema, region_sector=roi_type)
        
        ''' ## Resample velocity of 120/240m to same grid as dmg maps'''
        if 'vx' in varName or 'vy' in varName:
            # load single year of dmg data as reference grid for current sector
            # region_ds_dmg  = myf.load_tiles_region_multiyear(  tilepath_dmg, tileNums_select , years_to_load=['2021'] , varname='dmg' )
            dmg_file = os.path.join(path2save, 'damage095','data_sector-{}_dmg_2021.nc'.format(sector_ID))
            region_ds_dmg = xr.open_dataset(dmg_file)

            # velocity is in 240m resolution, other data is 400m resolution --> resample grid.
            region_data = myf.reproject_match_grid( region_ds_dmg['dmg'] ,  region_data[varName]).to_dataset() # use dmg as reference grid

        ''' --------------------------------------
        Small fixes to data for netcdf/tiff saving
        ------------------------------------------ '''
        try:
            region_data = region_data.drop('spatial_ref')
        except: pass
        try:
            del region_data[varName].attrs['grid_mapping']
        except: pass

        if not region_data.rio.crs:
            # print('.. setting CRS to 3031')
            region_data.rio.write_crs(3031,inplace=True)
        
        ## Small fixes to file
        region_data.astype(float)[varName].rio.write_nodata(np.nan, encoded=True, inplace=True)
        # reorder dimensions ( need (y,x) without 3rd time dimension to save to netCDF that QGis can read)
        # data_da = region_data[varName].transpose('y','x')


        ''' --------------------------------------
        Save netCDF/GeoTIFF
        ------------------------------------------ '''

        
        if save_nc:
            if not os.path.isfile( os.path.join( path2save, nc_filename ) ):
                print('.. Saving to nectdf {} '.format(nc_filename))
                ## do the saving
                # data_da.to_netcdf(os.path.join(path2save,nc_filename),mode='w',format='NETCDF4')
                delayed_obj = region_data.to_netcdf(os.path.join(path2save,nc_filename),mode='w',format='NETCDF4',compute=False)
                with ProgressBar():
                    results = delayed_obj.compute()

        print(f'.. Done')


'''
##############################################
SAVING DMG NETCDFS FOR DATA PUBLICATION 
##############################################
'''

# ''' --------------
# Select region/sector and corresponding tilenumbers to export

# Available Sectors:
# 'ASE', 
# 'BSE', 
# 'EIS' 
# 'RS', 
# 'WIS'--> split in WIS-a and WIS-b
# 'WS' 

# Variables to save:
# 'dmg','dmg-25px','nodata'
# ------------------ '''

# # years_list = ['1997']
# years_list = ['2015','2016','2017','2018','2019','2020','2021']


# varName = 'dmg'
# varName = 'nodata'


# sector_ID_list.sort()

# save_nc = False 
# save_tif = False
# # set directory to save output
# path2data = os.path.join(homedir,'Data/NERD/dmg095_nc/data_sector/') # save dir

# for year in years_list:
#     res='400m' # default
#     if varName == 'dmg-25px':
#         region_data = region_data.rename({'dmg-25px':'dmg'})
#         res='1000m'
#     if int(year) == 1997 or int(year) == 2000:
#         res='1000m'

#     for sector_ID in ['WS']: #sector_ID_list:
#         if sector_ID == 'WIS' : # skip WIS in favor of WIS-a and WIS-b (process in parts due to memory usage)
#             continue 
#         # if sector_ID == 'WS' and res=='400m':
#         #     continue

        
#         ''' --------------
#         Define tileNumbers for selected region
#         ------------------ ''' 
#         ## select tiles
#         tileNums_select = myf.get_tilelist_region(sector_poly, sector_ID, gridTiles=gridTiles)

#         # Skip some tiles of FR and ROSS iceshelves that have nodata for S1 observations
#         if not varName == 'nodata':
#             tileNums_skip = [130,131,146,147,148,158,159,160,167,168,169, 170, 171,177,178,179,180,187,188,189,190,196,197,198,206,207,217,218] # for RV
#             tileNums_skip = tileNums_skip + [62,63,70,71,72,78,79,80,86,87,88,89,95,96,97,98,105,106,107,108,115,116,117,118,132,133,134,135,136,137,138,149,150] # for FR
#             tileNums_select = [tileNum for tileNum in tileNums_select if tileNum not in tileNums_skip]

#         print('--- \nSelected {} sector; {} tiles'.format(sector_ID, len(tileNums_select)))


#         print(f'.. loading data for {year}; {varName}')

#         ''' --------------
#         Load DMG or no-data 
#         ------------------ ''' 

#         if 'dmg' in varName:
#             path2save = os.path.join(path2data,'damage095')
#             if int(year) == 1997:
#                 tilepath_in = os.path.join(homedir,'Data/RAMP/RAMP_tiled/dmg_tiled/dmg095/')
#                 year_subdir=''
#             else:
#                 tilepath_in = os.path.join(homedir,'Data/S1_SAR/tiles/dmg_tiled/dmg095/')
#                 year_subdir = f'{year}-SON'
#         if 'nodata' in varName: 
#             path2save=os.path.join(path2data,'nodata')
#             tilepath_in = os.path.join(homedir,'Data/S1_SAR/tiles/masks/')
#             year_subdir = f'{year}-SON'
#             var='nodata'

#         ## get all files in directory 
#         # year_filelist = os.listdir(os.path.join(tilepath_in,year_subdir ))
#         year_filelist = glob.glob(os.path.join(tilepath_in,year_subdir,'*.tif' ))
#         year_filelist.sort()

#         ## select tiles in region
#         fnames_region = [fname for fname in year_filelist if int(fname.split('.')[0].split('tile_')[1]) in tileNums_select]
#         filelist_region  = [ os.path.join(tilepath_in,year_subdir, fname) for fname in fnames_region ]

#         region_data = (xr.open_mfdataset( filelist_region,  
#                     combine="nested", decode_times=False,
#                     data_vars='minimal', 
#                     coords= 'minimal', 
#                     compat='no_conflicts', #  only values which are not null in both datasets must be equal. The returned dataset then contains the combination of all non-null values
#                     chunks={'y':'auto','x':'auto','band':1}, # add chucnking info for dask
#                     parallel=True,
#                     ).isel(band=0).drop('band')
#                     .transpose('y','x')
#                     .rename({'band_data':varName})
#         )


#         ''' ## Fill dmg NaN values as 0  '''
#         region_data = region_data.where(~np.isnan(region_data),other=0 ) # no-dmg = 0


        
#         ''' --------------------------------------
#         Small fixes to data for netcdf/tiff saving
#         ------------------------------------------ '''
#         try:
#             region_data = region_data.drop('spatial_ref')
#         except: pass
#         try:
#             del region_data[varName].attrs['grid_mapping']
#         except: pass

#         if not region_data.rio.crs:
#             # print('.. setting CRS to 3031')
#             region_data.rio.write_crs(3031,inplace=True)
        
#         ''' ## drop 'time' dimension  '''
#         if len(region_data.dims) > 2:
#             print(region_data.dims)
#             region_data= region_data.isel(band=0).drop('band')

#         ## Small fixes to file 
#         region_data.astype(float)[varName].rio.write_nodata(np.nan, encoded=True, inplace=True) 
#         # reorder dimensions ( need (y,x) without 3rd time dimension to save to netCDF that QGis can read)
#         data_da = region_data[varName].transpose('y','x')

        
#         ''' --------------------------------------
#         Clip to ice shelf
#         ------------------------------------------ '''
#         if 'dmg' in varName:
#             print('.. clippinig to ice shelves')
#             iceshelf_year = ishelf_dict[year] # iceshelf_dflist[yidx]
#             ## CLIP data to iceshevles 
#             # using DROP=TRUE all pixels outside of iceshelf boundary are set to NaN. 
#             data_da  = data_da.rio.clip( 
#                                 iceshelf_year.geometry, iceshelf_year.crs, 
#                                 drop=True, invert=False)
#             # print(data_da)


#         ''' --------------------------------------
#         Save netCDF/GeoTIFF
#         ------------------------------------------ '''

#         ## Check if data exitsts
#         nc_filename = f'{varName}_sector-{sector_ID}_{year}-SON_{res}.nc' #nc_base + '_' + str(year_part) + '.nc'
#         tiff_file = f'{varName}_sector-{sector_ID}_{year}-SON_{res}.tif' 
#         already_exists = os.path.isfile( os.path.join( path2save, nc_filename ) )
        
#         if save_nc:
#             if not os.path.isfile( os.path.join( path2save, nc_filename ) ):
#                 print('.. Saving to nectdf {} '.format(nc_filename))
#                 ## do the saving
#                 # data_da.to_netcdf(os.path.join(path2save,nc_filename),mode='w',format='NETCDF4')
#                 delayed_obj = data_da.to_netcdf(os.path.join(path2save,nc_filename),mode='w',format='NETCDF4',compute=False)
#                 with ProgressBar():
#                     results = delayed_obj.compute()
                
#         if save_tif: 
#             if not os.path.isfile( os.path.join( path2save, tiff_file ) ):
#                 print('.. Saving to geotiff ', tiff_file)
#                 # save it, now with CRS and as Cloud Optimized Geotiff
#                 data_da.rio.to_raster( os.path.join(path2save, tiff_file),driver="COG") 

