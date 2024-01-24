import xarray as xr
import numpy as np
import os
import geopandas as gpd
import dask 

# import postProcessFunctions as myf
import myFunctions as myf

# ''' -----
# Define which processing step to take

# 1. do_save_annual            : Read an save annual data for all selected parameters as netcdf (damage, rema ice height, its_live velocity; calculate strain rates)
# 2. do_load_and_save_multiyear: Read annual data, calculate temporal variables, store as single netcdf

# NB: every processing step is done for a specified region (generating separate netcdf files) to better handle memory and file sizes. See futher down in code.
# Note: in hindsight it would have made more sense to make individual netcdf files per variable (and then Antarctic wide) instead of region-files with multiple variables. But now it's already working :)
# ---------'''
# do_load_and_save_singlevar_multiyear = False # make nc for single variables 2015-2018 (part1) or 2019-2021 (part 2) combined ve
# do_save_variables_multiyear_p1_p2 = False  # save single variables for 2015-2018 (part1) or 2019-2021 (part 2) combined 
# ## Save sectors;
# do_save_var_annual = True # save single variable, single year (to make an AIS wide netcdf eventually, per year)
# do_combine_sector_parts = False 

# if do_save_var_annual:
#     print("SELECTED: save annual values of (single) variable for specified sector to netCDF")
# elif do_combine_sector_parts:
#     print("SELECTED: load and combine split sector files, save to netCDF")
# elif do_save_variables_multiyear_p1_p2:
#     print("SELECTED: load and combine annual values for a single variable, save to netCDF")


''' -----
Set paths
---------'''

## Local
homedir = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/'
## VRlab
# homedir = '/net/labdata/maaike/'

tilepath_dmg = os.path.join(homedir,'Data/S1_SAR/tiles/dmg_tiled/dmg095/')
# tilepath_dmg = os.path.join(homedir,'Data/RAMP/RAMP_tiled/dmg_tiled/')
# tilepath_rema = os.path.join(homedir,'Data/REMA/tiles/')
# tilepath_velocity = os.path.join(homedir,'Data/ITS_LIVE/tiles/velocity/')

path2data = os.path.join(homedir,'Data/NERD/data_predictor/')


''' --------------
Get Shapefiles 
------------------ '''
# geojson
gridTiles_geojson_path = os.path.join(homedir,'Data/tiles/gridTiles_iceShelves_EPSG3031.geojson')
gridTiles = gpd.read_file(gridTiles_geojson_path)

# ## regions of interest for AIS
# roi_path = os.path.join(homedir, 'QGis/data_NeRD/plot_insets_AIS_regions.shp')
# roi_poly = gpd.read_file(roi_path)

## region_ID_list = roi_poly['region_ID'].to_list()

# ## redefined: SECTORS for AIS
# sector_path = os.path.join(homedir, 'QGis/data_NeRD/plot_insets_AIS_sectors.shp')
# sector_poly = gpd.read_file(sector_path)
# sector_ID_list = sector_poly['sector_ID'].to_list()

        
def save_nc_variable( path2save, varName, year_part,sector_ID, data_to_save=None): #,region_sector='sector'):
    
    # if region_sector == 'region':
    #     nc_base= 'data_region-' + region_ID + '_'+varName 
    # elif region_sector == 'sector':
    nc_base= 'data_sector-' + sector_ID + '_'+varName 

    # if varName == 'strain':
    #     nc_base = nc_base + '_1px'
    # if 'dmg' in varName:
    #     path2save = os.path.join(path2save,'damage')

    # if len(year_part) > 1: # multi-year
    #     if '2015' in year_part or 'part1' in year_part:
    #         nc_filename = nc_base + '_part1.nc' # part 1 for 2015-2018
    #     elif '2019' in year_part or 'part2' in year_part:
    #         nc_filename = nc_base + '_part2.nc' # part 2 for 2019-2021
    #     if '2015' in year_part and '2019' in year_part:
    #         nc_filename = nc_base + '_all.nc' # all years 2015-2021
    # else: # single year
    if isinstance(year_part, list):
        nc_filename = nc_base + '_' + year_part[0] + '.nc'
    elif isinstance(year_part, int) or isinstance(year_part, str):
        nc_filename = nc_base + '_' + str(year_part) + '.nc'


    # if varName == 'rema':
    #     nc_filename = nc_base + '_' + '0000.nc'

    ## Check if data exitsts
    already_exists = os.path.isfile( os.path.join( path2save, nc_filename ) )
    
    ## Save data
    if data_to_save is not None:
        if already_exists:
            print('.. netCDF for {} already exists ({}); continue without saving'.format(varName, nc_filename))
        else:
            print('.. Saving {} to nectdf {} '.format(varName, nc_filename))
            ## do the saving
            data_to_save.to_netcdf(os.path.join(path2save,nc_filename),mode='w',format='NETCDF4') 

    return nc_filename, already_exists


years_list = ['2015']#,'2016','2017','2018','2019','2020','2021']

variables_to_save = ['dmg095']
# variables_to_save = ['dmg-25px']
# variables_to_save = ['nodata']

# sector_ID_list.sort()
region_ID = 'AIS'
tileNums_all = list(np.arange(0,314))
ais_grid = myf.make_ais_grid( 400 )

## select tiles
tileNums_DML = [229,231,232,235,236,239,240,242,243,246,247,248,252,253,255,258,263,264,270,271,278,279,280,288,289,290,112,122,123,124,125,126,139,140,141,142,143,151,152,153,154,155,161,162,163,172,173,181,182,183,191,192,199,200,208,209,219,220,223,224,227,228]
tileNums_other = [x for x in tileNums_all if x not in tileNums_DML]

for year in years_list:

    path2data = os.path.join(homedir,'Data/NERD/dmg095_nc/') # save dir


    ''' --------------
    Load DMG 
    ------------------ ''' 

    # Load dmg
    # if 'dmg095' in variables_to_save or 'dmg-25px' in variables_to_save:
    # get the varname
    # if 'dmg095' in variables_to_save:
    #     Dname = 'dmg'
    # elif 'dmg-25px' in variables_to_save:
    #     Dname = 'dmg-25px'
    if len(variables_to_save) == 1:
        Dname = variables_to_save[0]

    print('.. loading tiles DML')
    print(tilepath_dmg)
    year_subdir = f'{year}-SON'

    ## get dmg files in directory 
    year_filelist = os.listdir(os.path.join(tilepath_dmg,year_subdir ))
    year_filelist.sort()

    ## select tiles in DML region
    fnames_DML = [fname for fname in year_filelist if int(fname.split('.')[0].split('tile_')[1]) in tileNums_DML]
    filelist_DML  = [ os.path.join(tilepath_dmg,year_subdir, fname) for fname in fnames_DML ]
    print('tiles in DML:',len(filelist_DML))

    ## select all other tiles
    fnames_other = [fname for fname in year_filelist if int(fname.split('.')[0].split('tile_')[1]) in tileNums_other]
    filelist_other  = [ os.path.join(tilepath_dmg,year_subdir, fname) for fname in fnames_other ]
    print('other tiles :',len(filelist_other))

    

    ''' --------------------------------------
    Load data
    nb: need to load tiles from DML region separately and combine by reprojecting to a predefined antarctic grid
        as otherwise xarray doesnt patch the coordinates correctly.
        This occured because the gridTiles of DML are not connected to the rest of the tiles (there's a gap).
    ------------------------------------------ '''
    ## Load DML region
    region_DML = (xr.open_mfdataset( filelist_DML,  
                    # combine="by_coords", decode_times=False,
                    combine="nested", decode_times=False,
                    data_vars='minimal', 
                    coords= 'minimal', 
                    # compat='broadcast_equals', #  all values must be equal when variables are broadcast against each other to ensure common dimensions.
                    compat='no_conflicts', #  only values which are not null in both datasets must be equal. The returned dataset then contains the combination of all non-null values
                    chunks={'y':'auto','x':'auto','band':1}, # add chucnking info for dask
                    ).isel(band=0).drop('band')
                    # .assign_coords(time=int(year)).expand_dims(dim=dict(time=int(year)),axis=-1)
                    .transpose('y','x')#,'time')
                    .rename({'band_data':Dname})
    )
    print('-- DML')
    print(region_DML.dims)#, region_ds_dmg.vars)
    # print(region_DML)

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        data_ais_1 = myf.reprj_regions_to_ais_grid(ais_grid, region_DML)

        print(data_ais_1.dims)
        # print(data_ais_1.assign_coords(time=int(year)).expand_dims(dim=dict(time=int(year)),axis=-1) ) 
        # print(data_ais_1)
        
        ## Load rest of tiles
        region_ds = (xr.open_mfdataset( filelist_other,  
                        combine="nested", decode_times=False,
                        data_vars='minimal', 
                        coords= 'minimal', 
                        compat='no_conflicts', #  only values which are not null in both datasets must be equal. The returned dataset then contains the combination of all non-null values
                        chunks={'y':'auto','x':'auto','band':1}, # add chucnking info for dask
                        ).isel(band=0).drop('band')
                        # .assign_coords(time=int(year)).expand_dims(dim=dict(time=int(year)),axis=-1)
                        .transpose('y','x')#,'time')
                        .rename({'band_data':Dname})
        )

        print('-- Rest')
        print(region_ds.dims)#, region_ds_dmg.vars)
        # print(region_ds)

        data_ais_2 = myf.reprj_regions_to_ais_grid(ais_grid, region_ds)

        ## Combine the two regional sets of AIS gridtiles
        # xr.combine_first() defaults to non-null values in the calling object, and fills holes with called object. Effecitvely patching the new region to the first
    
        data_ais = data_ais_1.combine_first(data_ais_2) #.assign_coords(time=int(year)) 

        print('-- AIS \n')
        # print(region_ds.dims)
        print(region_ds)
        

        ''' --------------------------------------
        Small fixes to data
        ------------------------------------------ '''
        ## Fill dmg NaN values as 0 
        data_ais = data_ais.where(~np.isnan(data_ais),other=0 ) # no-dmg = 0
        # if Dname == 'dmg-25px':
        #     data_ais = data_ais.rename({'dmg-25px':'dmg'})

        if not data_ais.rio.crs:
            print('.. setting CRS to 3031')
            region_ds = region_ds.drop('spatial_ref')
            region_ds.rio.write_crs(3031,inplace=True)
            del data_ais[Dname].attrs['grid_mapping']


        ## Small fixes to file
        data_ais.astype(float)[Dname].rio.write_nodata(np.nan, encoded=True, inplace=True)
        # reorder dimensions (due to reprojection they were (x,y,t) and I need (y,x) to save to netCDF that QGis can read)
        data_da = data_ais[Dname].transpose('y','x')

        ''' --------------------------------------
        Save netCDF/GeoTIFF
        ------------------------------------------ '''

        ## Check if data exitsts
        nc_filename = f'{Dname}_AIS_{year}-SON.nc' #nc_base + '_' + str(year_part) + '.nc'
        tiff_file = f'{Dname}_AIS_{year}-SON.tiff' 
        already_exists = os.path.isfile( os.path.join( path2data, nc_filename ) )
        
        save_nc = True 
        save_tif = True
        if save_nc:
            if not os.path.isfile( os.path.join( path2data, nc_filename ) ):
                print('.. Saving to nectdf {} '.format(nc_filename))
                ## do the saving
                data_da.to_netcdf(os.path.join(path2data,nc_filename),mode='w',format='NETCDF4')
            else:
                print('.. netCDF for {} already exists ({}); continue without saving'.format(Dname, nc_filename))

        if save_tif: 
            if not os.path.isfile( os.path.join( path2data, tiff_file ) ):
                print('.. Saving to geotiff ', tiff_file)
                data_da.rio.to_raster( os.path.join(path2data, tiff_file),driver="COG") # save it, now with CRS and as Cloud Optimized Geotiff
        




# ---------------------------------





# # ## Check if files exsits ; otherwise skip loading and saving
# # varFiles_saved=0
# # for varName in variables_to_save:
# #     nc_filename, already_exists = save_nc_variable(path2data, varName, years,region_ID,  region_sector=roi_type )
# #     if already_exists:
# #         varFiles_saved += 1
# # if varFiles_saved == len(variables_to_save):
# #     print('All variables {} already saved for {} year {}'.format(variables_to_save,region_ID, year))
# #     continue

# # if do_save_var_annual:
# ''' --------------
# Select region/sector and corresponding tilenumbers to export

# Available Sectors:
# 'ASE', 
# 'BSE', 
# 'EIS' --> split in 'EIS-1','EIS-2', 
# 'RS', 
# 'WIS'--> split in 'WIS-1','WIS-2', || UPDATE: split in WIS-a and WIS-b
# 'WS' --> split in 'WS-1', 'WS-2'

# Variables to save:
# 'dmg','vx','vy','rema'
# 'dmg-25px'
# 'fractures' for Pang2023 MOA detected fractures
# ------------------ '''

# # years_list = ['2015','2016','2017','2018']
# # years_list = ['2019','2020','2021']
# # years_list = ['2021'] 
# # years_list = ['1997'] 
# # tilepath_dmg = os.path.join(homedir,'Data/RAMP/RAMP_tiled/dmg_tiled/')
# # years_list = ['2004','2009','2014'] # for Pang2023 fractures

# years_list = ['2015','2016','2017','2018','2019','2020','2021']

# # variables_to_save = ['dmg','vx','vy','rema']
# # variables_to_save = ['vx','vy','rema']
# variables_to_save = ['dmg']
# # variables_to_save = ['dmg-25px']
# # variables_to_save = ['nodata']
# # variables_to_save = ['fractures'] ## MOA data for Pang2023 fractures

# sector_ID_list.sort()

# for year in years_list:
#     years=[year]

#     for region_ID in ['WIS-a', 'WIS-b']: # sector_ID_list[-3:]: 
#         path2data = os.path.join(homedir,'Data/NERD/data_predictor/data_sector/') # save dir
#         roi_type = 'sector'
#         # tileNums_select = myf.get_tilelist_region(sector_poly, region_ID, gridTiles=gridTiles, roi_type=roi_type)

#         ### Process Weddell Sea sector in 2parts or single part || only required for velocity & strain variables
#         if any( [var in ['vx','vy','v','strain'] for var in variables_to_save]): 
#             if 'EIS' in region_ID or 'WIS' in region_ID or 'WS' in region_ID: 
#                 if not '-' in region_ID:
#                     print('Skip EIS/WIS/WS large-sector:', region_ID)
#                     continue 
#         elif 'WIS' in region_ID:
#             if region_ID not in ['WIS-a','WIS-b']:
#                 print('Skipping {} - Need to process WIS in A and B part (netcdf projction error otherwise)'.format(region_ID))
#                 continue
#         else:
#             if '-' in region_ID:
#                 print('Processing full sectors; skip part-sector ',region_ID)
#                 continue   

#         ## select tiles
#         tileNums_select = myf.get_tilelist_region(sector_poly, region_ID, gridTiles=gridTiles, roi_type=roi_type)

#         ## Check if files exsits ; otherwise skip loading and saving
#         varFiles_saved=0
#         for varName in variables_to_save:
#             nc_filename, already_exists = save_nc_variable(path2data, varName, years,region_ID,  region_sector=roi_type )
#             if already_exists:
#                 varFiles_saved += 1
#         if varFiles_saved == len(variables_to_save):
#             print('All variables {} already saved for {} year {}'.format(variables_to_save,region_ID, year))
#             continue
        
#         # Skip some tiles of FR and ROSS iceshelves that have nodata for S1 observations
#         if not varName == 'nodata':
#             tileNums_skip = [130,131,146,147,148,158,159,160,167,168,169, 170, 171,177,178,179,180,187,188,189,190,196,197,198,206,207,217,218] # for RV
#             tileNums_skip = tileNums_skip + [62,63,70,71,72,78,79,80,86,87,88,89,95,96,97,98,105,106,107,108,115,116,117,118,132,133,134,135,136,137,138,149,150] # for FR
#             tileNums_select = [tileNum for tileNum in tileNums_select if tileNum not in tileNums_skip]

#         print('--- \nSelected {} region; {} tiles'.format(region_ID, len(tileNums_select)))

#         varFiles_saved=0
#         for varName in variables_to_save:
#             nc_filename, already_exists = save_nc_variable(path2data, varName, years , region_ID)
#             if already_exists:
#                 varFiles_saved += 1
#         if varFiles_saved == len(variables_to_save):
#             print('All variables are already saved for selected years; {} for {}'.format(variables_to_save, years))
#             continue

#         print('.. loading data for', year)
#         print('Saving variables: ', variables_to_save)

#         ''' --------------
#         Load DMG 
#         ------------------ ''' 

#         # Load dmg
#         if 'dmg' in variables_to_save or 'dmg-25px' in variables_to_save:
#             # get the varname
#             if 'dmg' in variables_to_save:
#                 Dname = 'dmg'
#             elif 'dmg-25px' in variables_to_save:
#                 Dname = 'dmg-25px'

#             region_ds_dmg  = myf.load_tiles_region_multiyear(  tilepath_dmg, tileNums_select , years_to_load=years , varname= Dname )
#             ## Fill dmg NaN values as 0 
#             region_ds_dmg = region_ds_dmg.where(~np.isnan(region_ds_dmg),other=0 ) # no-dmg = 0
#             if Dname == 'dmg-25px':
#                 region_ds_dmg = region_ds_dmg.rename({'dmg-25px':'dmg'})

#             # print(region_ds_dmg.dims)#, region_ds_dmg.vars)
#             # print(region_ds_dmg)
#             # raise RuntimeError
#             # ## Save
#             # path2var = os.path.join(path2data, 'damage')
#             nc_filename, already_exists = save_nc_variable(path2data, Dname, years, region_ID, region_ds_dmg,region_sector=roi_type ) 
            
#         else: # load only single year to use as reference grid
#             region_ds_dmg  = myf.load_tiles_region_multiyear(  tilepath_dmg, tileNums_select , years_to_load=['2021'] , varname='dmg' )



#         # raise RuntimeError('Stop here, for dev')
#         # ''' --------------
#         # Load mask / frac
#         # ------------------ '''
#         # # if any( [True for var in variables_to_save if var in ['nodata'] ] ):
#         # if 'nodata' in variables_to_save:
#         #     tilepath_masks = os.path.join(homedir,'Data/S1_SAR/tiles/masks/')
#         #     var='nodata'

#         #     ## Load tiles
#         #     region_data  = myf.load_tiles_region_multiyear(  
#         #                     tilepath_masks, tileNums_select , 
#         #                     years_to_load=[year] , 
#         #                     varname= var )
#         #     print('..resolution:', region_data.rio.resolution())
#         #     ## Resample to same grid as dmg (400m or 1000m)
#         #     if region_data.rio.resolution()[0] == 400:
#         #         ## load dummy dmg at 1000m resolution
#         #         region_ds_dmg  = myf.load_tiles_region_multiyear(  tilepath_dmg, tileNums_select , years_to_load=['2021'] , varname='dmg-25px' )
#         #     elif region_data.rio.resolution()[0] == 1000:
#         #         ## load dummy dmg at 1000m resolution
#         #         region_ds_dmg  = myf.load_tiles_region_multiyear(  tilepath_dmg, tileNums_select , years_to_load=['2021'] , varname='dmg-25px' ).rename({'dmg-25px':'dmg'})
#         #     region_data = myf.reproject_match_grid(region_ds_dmg['dmg'], region_data)

#         #     ## Save
#         #     path2var = os.path.join(path2data, var)
#         #     nc_filename, already_exists = save_nc_variable(path2var, var , years, 
#         #                                                     region_ID, region_data, region_sector=roi_type ) 




# # import glob
# # import dask
# # import matplotlib.pyplot as plt
# # if do_combine_sector_parts:
# #     ''' Some sectors were too large (esp. to save a velocity-variables)
# #     So these were split in two parts. Combine them here. 
# #     'EIS' --> split in 'EIS-1','EIS-2', 
# #     'WIS'--> split in 'WIS-1','WIS-2', 
# #     'WS' --> split in 'WS-1', 'WS-2'
# #     '''
# #     years_list = ['2019','2020','2021']
# #     years_list = ['2015','2016','2017','2018','2019','2020','2021']

# #     variables_to_combine = ['dmg','vx','vy']

# #     path2data = os.path.join(homedir,'Data/NERD/data_predictor/data_sector/')
# #     roi_type = 'sector'
# #     for sector_ID in ['WS']: #  ['EIS' , 'WIS','WS']: 
# #         print('Combining data for ', sector_ID)
# #         for variable_to_combine in variables_to_combine: 
# #             for year in years_list:
# #                 print('--', year)
# #                 # check if file already exists; then skip (incorporated in function)
# #                 _, already_exists = save_nc_variable(path2data, variable_to_combine, [year] , sector_ID)

# #                 nc_file1 = 'data_sector-'+sector_ID+'-1_'+variable_to_combine +'_'+year+'.nc'
# #                 nc_file2 = 'data_sector-'+sector_ID+'-2_'+variable_to_combine +'_'+year+'.nc'

# #                 region_p1 = xr.open_dataset(os.path.join(path2data, nc_file1), chunks={})
# #                 region_p2 = xr.open_dataset(os.path.join(path2data, nc_file2), chunks={}) # specify chuncks, to load as dask

# #                 # print('..', region_p1[variable_to_combine].shape, region_p2[variable_to_combine].shape)

# #                 with dask.config.set(**{'array.slicing.split_large_chunks': True}):
# #                     # step 1: concatenate (to new dimension)
# #                     # step 2: flatten the concat dim (e.g. taking max)
# #                     region_ds = xr.concat([region_p1, region_p2],dim=['y','x']).max(dim='concat_dim')

# #                 # print('.. Merged to single dataset ',  region_ds[variable_to_combine].shape)
# #                 # print('.. resolution  {}m'.format(region_ds.rio.resolution()))
# #                 ''' --------
# #                 SAVE DATA TO NETCDF 
# #                 ------------'''
# #                 nc_file_new= 'data_sector-'+sector_ID+'_'+variable_to_combine +'_'+year+'*.nc'
# #                 save_nc_variable(path2data, variable_to_combine, [year],sector_ID,  
# #                                             region_ds, region_sector=roi_type)
        
        