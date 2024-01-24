import xarray as xr
import numpy as np
import os
import geopandas as gpd
import dask 

# import postProcessFunctions as myf
import myFunctions as myf
from dask.diagnostics import ProgressBar

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
sector_path = os.path.join(homedir, 'QGis/data_NeRD/AIS_outline_sectors.shp')
sector_poly = gpd.read_file(sector_path)
sector_ID_list = sector_poly['sector_ID'].to_list()

# print(sector_ID_list)
        

years_list = ['2015']#,'2016','2017','2018','2019','2020','2021']

variables_to_save = ['dmg095']
# variables_to_save = ['dmg-25px']
# variables_to_save = ['nodata']



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

ishelf_dict = { '1997':iceshelf_df_1997,
                '2015':iceshelf_df_2015,
                '2015':iceshelf_df_2016,
                '2015':iceshelf_df_2017,
                '2015':iceshelf_df_2018,
                '2015':iceshelf_df_2019,
                '2015':iceshelf_df_2020,
                '2015':iceshelf_df_2021,
}

# ## Check if files exsits ; otherwise skip loading and saving
# varFiles_saved=0
# for varName in variables_to_save:
#     nc_filename, already_exists = save_nc_variable(path2data, varName, years,region_ID,  region_sector=roi_type )
#     if already_exists:
#         varFiles_saved += 1
# if varFiles_saved == len(variables_to_save):
#     print('All variables {} already saved for {} year {}'.format(variables_to_save,region_ID, year))
#     continue

if save_var_annual:
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

    # years_list = ['2015','2016','2017','2018']
    # years_list = ['2019','2020','2021']
    # years_list = ['2021'] 
    # years_list = ['1997'] 
    # tilepath_dmg = os.path.join(homedir,'Data/RAMP/RAMP_tiled/dmg_tiled/')
    # years_list = ['2004','2009','2014'] # for Pang2023 fractures

    years_list = ['2015']#,'2016','2017','2018','2019','2020','2021']

    variables_to_save = ['dmg']
    varName = 'dmg'
    # variables_to_save = ['dmg-25px']
    # variables_to_save = ['nodata']
    # variables_to_save = ['fractures'] ## MOA data for Pang2023 fractures

    sector_ID_list.sort()

    save_nc = True 
    save_tif = True

    for year in years_list:
        years=[year]
        
        for sector_ID in sector_ID_list:
            if sector_ID == 'WIS' or sector_ID == 'WS': # skip WIS in favor of WIS-a and WIS-b (process in parts)
                continue 
            path2data = os.path.join(homedir,'Data/NERD/data_predictor/data_sector/') # save dir
            # roi_type = 'sector'

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
            Load DMG 
            ------------------ ''' 

            if 'dmg' in varName:
                # region_data  = myf.load_tiles_region_multiyear(  tilepath_dmg, tileNums_select , years_to_load=years , varname= varName )
                path2save = os.path.join(path2data,'damage095')
                year_subdir = f'{year}-SON'

                ## get all files in directory 
                year_filelist = os.listdir(os.path.join(tilepath_dmg,year_subdir ))
                year_filelist.sort()

                ## select tiles in DML region
                fnames_region = [fname for fname in year_filelist if int(fname.split('.')[0].split('tile_')[1]) in tileNums_select]
                filelist_region  = [ os.path.join(tilepath_dmg,year_subdir, fname) for fname in fnames_region ]

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

                res='400m'
                if varName == 'dmg-25px':
                    region_data = region_data.rename({'dmg-25px':'dmg'})
                    res='1000m'

                ''' ## Fill dmg NaN values as 0  '''
                region_data = region_data.where(~np.isnan(region_data),other=0 ) # no-dmg = 0


            ''' --------------
            Load mask / frac
            ------------------ '''
            if 'nodata' in varName: 
                tilepath_masks = os.path.join(homedir,'Data/S1_SAR/tiles/masks/')
                path2save=os.path.join(path2data,'nodata')
                var='nodata'

                ## Load tiles
                region_data  = myf.load_tiles_region_multiyear(  
                                tilepath_masks, tileNums_select , 
                                years_to_load=[year] , 
                                varname= var )
                print('..resolution:', region_data.rio.resolution())
                ## Resample to same grid as dmg (400m or 1000m)
                if region_data.rio.resolution()[0] == 400:
                    ## load dummy dmg at 400 resolution
                    region_ds_dmg  = myf.load_tiles_region_multiyear(  tilepath_dmg, tileNums_select , years_to_load=['2021'] , varname='dmg' )
                    res='400m'
                elif region_data.rio.resolution()[0] == 1000:
                    ## load dummy dmg at 1000m resolution
                    region_ds_dmg  = myf.load_tiles_region_multiyear(  tilepath_dmg, tileNums_select , years_to_load=['2021'] , varname='dmg-25px' ).rename({'dmg-25px':'dmg'})
                    res='1000m'
                region_data = myf.reproject_match_grid(region_ds_dmg['dmg'], region_data)

            
            ''' --------------------------------------
            Small fixes to data for netcdf/tiff saving
            ------------------------------------------ '''

            if not region_data.rio.crs:
                print('.. setting CRS to 3031')
                region_data = region_data.drop('spatial_ref')
                region_data.rio.write_crs(3031,inplace=True)
                del region_data[varName].attrs['grid_mapping']

            
            ''' ## drop 'time' dimension  '''
            if len(region_data.dims) > 2:
                print(region_data.dims)
                region_data= region_data.isel(band=0).drop('band')

            # print(region_data.dims)#, region_ds_dmg.vars)

            ## Small fixes to file
            region_data.astype(float)[varName].rio.write_nodata(np.nan, encoded=True, inplace=True)
            # reorder dimensions ( need (y,x) without 3rd time dimension to save to netCDF that QGis can read)
            data_da = region_data[varName].transpose('y','x')

            
            ''' --------------------------------------
            Clip to ice shelf
            ------------------------------------------ '''
            if 'dmg' in varName:
                print('.. clippinig to ice shelves')
                iceshelf_year = ishelf_dict[year] # iceshelf_dflist[yidx]
                ## CLIP data to iceshevles 
                # using DROP=TRUE all pixels outside of iceshelf boundary are set to NaN. 
                data_da  = data_da.rio.clip( 
                                    iceshelf_year.geometry, iceshelf_year.crs, 
                                    drop=True, invert=False)
                # print(data_da)


            ''' --------------------------------------
            Save netCDF/GeoTIFF
            ------------------------------------------ '''

            ## Check if data exitsts
            nc_filename = f'{varName}_sector-{sector_ID}_{year}-SON_{res}.nc' #nc_base + '_' + str(year_part) + '.nc'
            tiff_file = f'{varName}_sector-{sector_ID}_{year}-SON_{res}.tiff' 
            already_exists = os.path.isfile( os.path.join( path2save, nc_filename ) )
            
            if save_nc:
                if not os.path.isfile( os.path.join( path2save, nc_filename ) ):
                    print('.. Saving to nectdf {} '.format(nc_filename))
                    ## do the saving
                    # data_da.to_netcdf(os.path.join(path2save,nc_filename),mode='w',format='NETCDF4')
                    delayed_obj = data_da.to_netcdf(os.path.join(path2save,nc_filename),mode='w',format='NETCDF4',compute=False)
                    with ProgressBar():
                        results = delayed_obj.compute()
                    
            if save_tif: 
                if not os.path.isfile( os.path.join( path2save, tiff_file ) ):
                    print('.. Saving to geotiff ', tiff_file)
                    # save it, now with CRS and as Cloud Optimized Geotiff
                    data_da.rio.to_raster( os.path.join(path2save, tiff_file),driver="COG") 


# import glob
# import dask
# if combine_sector_parts:
#     ''' Some sectors were too large (esp. to save a velocity-variables)
#     So these were split in two parts. Combine them here. 
#     'EIS' --> split in 'EIS-1','EIS-2', 
#     'WIS'--> split in 'WIS-1','WIS-2', 
#     'WS' --> split in 'WS-1', 'WS-2'
#     '''
#     years_list = ['2019','2020','2021']
#     years_list = ['2015','2016','2017','2018','2019','2020','2021']

#     variables_to_combine = ['dmg','nodata']

#     path2data = os.path.join(homedir,'Data/NERD/data_predictor/data_sector/')
#     roi_type = 'sector'
#     for sector_ID in ['WS','WIS']: #  ['EIS' , 'WIS','WS']: 
#         print('Combining data for ', sector_ID)
#         for variable_to_combine in variables_to_combine: 
#             for year in years_list:
#                 print('--', year)
#                 # check if file already exists; then skip (incorporated in function)
#                 _, already_exists = save_nc_variable(path2data, variable_to_combine, [year] , sector_ID)

#                 nc_file1 = 'data_sector-'+sector_ID+'-1_'+variable_to_combine +'_'+year+'.nc'
#                 nc_file2 = 'data_sector-'+sector_ID+'-2_'+variable_to_combine +'_'+year+'.nc'

#                 region_p1 = xr.open_dataset(os.path.join(path2data, nc_file1), chunks={})
#                 region_p2 = xr.open_dataset(os.path.join(path2data, nc_file2), chunks={}) # specify chuncks, to load as dask

#                 # print('..', region_p1[variable_to_combine].shape, region_p2[variable_to_combine].shape)

#                 with dask.config.set(**{'array.slicing.split_large_chunks': True}):
#                     # step 1: concatenate (to new dimension)
#                     # step 2: flatten the concat dim (e.g. taking max)
#                     region_ds = xr.concat([region_p1, region_p2],dim=['y','x']).max(dim='concat_dim')

#                 # print('.. Merged to single dataset ',  region_ds[variable_to_combine].shape)
#                 # print('.. resolution  {}m'.format(region_ds.rio.resolution()))
#                 ''' --------
#                 SAVE DATA TO NETCDF 
#                 ------------'''
#                 nc_file_new= 'data_sector-'+sector_ID+'_'+variable_to_combine +'_'+year+'*.nc'
#                 save_nc_variable(path2data, variable_to_combine, [year],sector_ID,  
#                                             region_ds, region_sector=roi_type)
        
        