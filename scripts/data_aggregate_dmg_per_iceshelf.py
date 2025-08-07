
import os
import rioxarray as rioxr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import glob
import xarray as xr

import rasterio as rio

import pandas as pd 

# import seaborn as sns
import argparse

# Import user functions
import myFunctions as myf 
import dask
from joblib import Parallel, delayed
import time

homedir = '/Users/.../'

tilepath_dmg = os.path.join(homedir,'Data/S1_SAR/tiles/dmg_tiled/')
tilepath_dmg = os.path.join(homedir,'Data/RAMP/RAMP_tiled/dmg_tiled/')
path2savefig = os.path.join(homedir,'Data/NERD/plots_dev/')
path2data = os.path.join(homedir,'Data/NERD/DMG_aggregated/')
gridTiles_geojson_path = os.path.join(homedir,'Data/tiles/gridTiles_iceShelves_EPSG3031.geojson')

path2iceshelves = os.path.join(homedir, 'Data/SHAPEFILES/annual_iceshelf_polygons/revised_measures_greene/')
iceshelf_path_meas = os.path.join(homedir, 'Data/SHAPEFILES/IceShelf_Antarctica_v02.shp')
sector_path = os.path.join(homedir, 'Data/SHAPEFILES/plot_insets_AIS_sectors.shp')
    

''' --------------
Get Shapefiles 
------------------ '''
# geojson
gridTiles = gpd.read_file(gridTiles_geojson_path)

# measures ice shelves
iceshelf_poly_meas = gpd.read_file(iceshelf_path_meas)

## SECTORS for AIS
sector_poly = gpd.read_file(sector_path)
sector_ID_list = sector_poly['sector_ID'].to_list()

sector_ID_list


''' --------------
Annual ice shelf polygons
------------------ '''

iceshelf_flist = glob.glob(path2iceshelves + '*.shp')
iceshelf_flist = [os.path.basename( filepath) for filepath in iceshelf_flist]
iceshelf_flist.sort()
iceshelf_flist

# annual ice shelves
iceshelf_df_1997 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_1997.75.shp' ) )
iceshelf_df_2000 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2000.75.shp' ) )
iceshelf_df_2015 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2015.2.shp' ) )
iceshelf_df_2016 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2016.2.shp' ) )
iceshelf_df_2017 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2017.2.shp' ) )
iceshelf_df_2018 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2018.2.shp' ) )
iceshelf_df_2019 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2019.2.shp' ) )
iceshelf_df_2020 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2020.2.shp' ) )
iceshelf_df_2021 = gpd.read_file(os.path.join(path2iceshelves, 'iceshelf_polygon_measures_greene_2021.2.shp' ) )

iceshelf_dflist = [iceshelf_df_1997,iceshelf_df_2015,iceshelf_df_2016,
                  iceshelf_df_2017,iceshelf_df_2018,iceshelf_df_2019,
                  iceshelf_df_2020,iceshelf_df_2021]
        

ishelf_dict = { '1997':iceshelf_df_1997,'2000':iceshelf_df_2000,'2015':iceshelf_df_2015,
                '2016':iceshelf_df_2016,'2017':iceshelf_df_2017,
                '2018':iceshelf_df_2018,'2019':iceshelf_df_2019,
                '2020':iceshelf_df_2020,'2021':iceshelf_df_2021,
}        


def parse_cla():
    """
    Command line argument parser

    Accepted arguments:
        Required:
        --sector (-s)     : Abbreviation of sector to process. 
                            Options: ('ASE', 'BSE', 'EIS', 'RS', 'WIS-a','WIS-b', 'WS')
        
        Optional:
        --timeframe (-t)  : Specify to process 1997 and 2021 damage values only
        --resolution      : Specify which damage maps to load. Options: 400m or 1000m 
        --year            : process only a single year. If not specified, all years will be loaded.

    :return args: ArgumentParser return object containing command line arguments
    """
    
    # Required arguments
    parser = argparse.ArgumentParser()
       
    parser.add_argument("--sector","-s",help='Specify sector to process',type=str, 
                            choices=('ASE', 'BSE', 'EIS', 'RS', 'WIS-a','WIS-b', 'WS') )    
    
    # Optional arguments
    group2 = parser.add_mutually_exclusive_group(required=False)
    group2.add_argument("--timeframe",'-t',
                    help="Specify 'longterm' to process 1997 and 2021 at same 1000m output resolution",
                         type=str, choices=('longterm','moa'))   
    group2.add_argument("--year","-y",help='Specify year to process (if not specified: all years) (at 400m res)',type=str )  
    parser.add_argument("--dtype","-d",help='Specify dmg type to process (default ''dmg095'')',type=str, required=False,
                            choices=('dmg','dmg095') , default='dmg095') 
    
    # parser.add_argument("--overlap", help='Specify anything to toggle.',type=str, required=False ) 
    parser.add_argument('--resolution',help='Resolution of dmg maps',type=str,required=False,
                            choices=('400m','1000m'))
    parser.add_argument('--masktype','-mt',help='Specify the way of masking nodata pixels',
                            type=str,choices=('annual','any','15-21','2021'),required=False)

    
    parser.add_argument('--strict',help='calculate stricter dmg values (for uncertainty), set to True or False ',type=str,required=False,
                            default='False')

    args = parser.parse_args()
    return args 



''' --------------------------------------
 SELECT REGION, LOAD DATA
------------------------------------------ '''

def main( region_ID, sector_ID, dmg_type = 'dmg095', 
                timeframe=None, year=None, resolution='400m',
                masktype=None,strict_type=None ):

    region_ID = sector_ID  ## leftover from previous versions
    path2nc = os.path.join(homedir, 'Data/NERD/data_predictor/data_sector/')
    path2nc = os.path.join(homedir, 'Data/NERD/dmg095_nc/data_sector/')

    # 1 Get sector polygon
    region_polygon = sector_poly[sector_poly['sector_ID'] == region_ID]
    
    if timeframe is None: # all years
        
        if resolution == '400m':
            filelist_dmg =  glob.glob( os.path.join(path2nc,'damage', 'data_sector-'+region_ID+'_dmg_2*.nc') )
            filelist_dmg = [file for file in filelist_dmg if '1997' not in file] 
            region_ds = (xr.open_mfdataset(filelist_dmg )
                    .drop('spatial_ref')
                    .rio.write_crs(3031,inplace=True)
            ) 
            
            ## load RAMP detected dmg, and reproject to 400m grid of S1 data
            region_ramp = xr.open_dataset(os.path.join(path2nc,'damage', 'data_sector-'+region_ID+'_dmg_1997.nc' ))
            # reproject to 400m grid
            region_ramp = myf.reproject_match_grid(region_ds, region_ramp)
            
            ## Add to dataset
            region_ds = xr.concat([region_ramp, region_ds],dim='time') 

        elif resolution == '1000m': ## All data at RAMP 1000m resolution
            print('--- Loading 1000m data ---- ')
            filelist_dmg =  glob.glob( os.path.join(path2nc,'damage/', 'data_sector-'+region_ID+'_dmg_1000m_*.nc') )
            filelist_dmg = [file for file in filelist_dmg if '1997' not in file] 
            filelist_dmg.sort()
            ## stored as (y,x) netcdfs without (time) dimension. need to open as combine=nested
            region_ds = (xr.open_mfdataset(filelist_dmg , combine='nested',concat_dim='time')
                    .drop('spatial_ref')
                    .rio.write_crs(3031,inplace=True)
            ) 
            # fix stupid dimension and cooridnate tranpose mismatch
            region_ds = region_ds.transpose('y','x','time')[[ "y", "x", "time", "dmg095"]] 
            # print(region_ds)

            ## ## load RAMP detected dmg
            region_ramp = xr.open_dataset(os.path.join(homedir,'damage/', 'data_sector-'+region_ID+'_dmg_1997.nc' ))
            
            '''## update august2024: load RAMP mamm 2000'''
            ds = xr.open_dataset(os.path.join(path2nc,'damage/', 'dmg_sector-'+region_ID+'_2000-SON_1000m.nc' )) ## is actually dmg095 but name is dmg 
            # add time-dimension to xarray.DataArray
            region_mamm = xr.DataArray( data = np.expand_dims(ds['dmg'],-1),  # (y,x) to (y,x,1)
                            coords={'y': (ds["y"]), 'x': (ds["x"]), 'time':([2000])},
                            name='dmg095', attrs=ds.attrs, indexes=ds.indexes # copy other properties
                            ).to_dataset(name='dmg095').rio.write_crs(3031,inplace=True) 
            region_mamm = myf.reproject_match_grid(region_ramp, region_mamm).transpose('y','x','time')[[ "y", "x", "time", "dmg095"]] 
            
            ## Add to dataset
            ## region_ds = xr.concat([region_ramp[['dmg095']], region_ds],dim='time') 
            region_ds = xr.concat([region_ramp[['dmg095']], region_mamm[['dmg095']], region_ds],dim='time') 
            print(region_ds)
                

    elif timeframe == 'longterm': # RAMP versus S1
        ''' ----------------
        Load damage for 1997 (100m 10px) and 2021 (40m 25px) and update threshodls accordingly
        -- make sure S1 1000m resolution is same ramp 1000m res grid
        --------------------'''
        
        ## load S1 detected dmg (has no-dmg filled as 0)
        nc_file = 'data_sector-'+region_ID+'_dmg-25px_2021.nc'
        region_ds = xr.open_dataset(os.path.join(path2nc,'damage', nc_file) ) # has no-dmg filled as 0
 
        ## load RAMP detected dmg
        region_ramp = xr.open_dataset(os.path.join(path2nc,'damage','data_sector-'+region_ID+'_dmg_1997.nc' ))

        ## Add to dataset
        region_ds = xr.concat([region_ramp, region_ds],dim='time') 

    if year:
        year_list = [int(year)]
    else:
        year_list = list(region_ds.time.values) # can be either all years 2015-2021, or 1997&2021 for when processing lonogterm lowres
        print('year_list is ', year_list)

    ''' -----------
    Update stricter dmg threshold for uncertainty ranges
    --------------'''
    ## update dmg threshold for uncertainty
    if strict_type is not None:

        path2save = os.path.join(os.path.join(homedir,'Data/NERD/data_organise/',f'stricter-{strict_type}'))
        ## B
        if strict_type == 'pct05':
            print('-- updating dmg threshold by removing lowest 5pct pxs') 
            annual_threshold = [0.000945, 0.000953, 0.000971, 0.000985, 0.000989, 0.000984, 0.000985, 0.000981, 0.000985] ## values calculated in GEE
            time_values         = [1997,    2000,       2015,        2016,  2017,       2018,   2019,       2020,   2021]
        ## B
        if strict_type == 'bias':
            print('-- updating dmg threshold by removing sensor bias')
            annual_threshold = [ 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0015, 0.001 ] 
            time_values         = [1997,    2000,       2015,        2016,  2017,       2018,   2019,       2020,   2021]

        # Update each time slice with the respective threshold (just mask out pixels, do not change Dsignal)
        slicelist = []
        for t, threshold in zip(time_values, annual_threshold):
                tslice = region_ds.sel(time=t).where(region_ds.sel(time=t)>threshold,other=0)
                slicelist.append(tslice) 
        region_ds = xr.concat(slicelist,dim='time')  

    ''' --------------------------------------
    Load masks: all years
    Masks have value 1 for px with NODATA and value 0 for px with valid data (strange maybe)
    ------------------------------------------ '''

    ## load from netcdf
    # nc_file = 'data_sector-'+region_ID+'_nodata_2021.nc'
    region_masks = (xr.open_mfdataset( glob.glob( os.path.join(path2nc,'nodata', 'data_sector-'+region_ID+'_nodata_*.nc') ))
                        .drop('spatial_ref')
                        .rio.write_crs(3031,inplace=True)
    ) 
    # # ## Resample mask to same grid as dmg 
    # # -- for masks, resampling does something weird wth coords. Can only coontinue with dataArray insteaad of dataSet
    region_masks = myf.reproject_match_grid(region_ds[dmg_type], region_masks['nodata'])
    ymax = len(region_masks.time.values)

    '''
    ## Deduce origin of most-recent data coverage
    Region_masks has binary mask for where nodata/valid data; (0=valid, 1=nodata)
    --> convert this to 0 (nodata) and 2015-2021 to specify which year valid data comes from
    --> then I can identify using .max() to which year I should fill with
    # '''
    data_available_year=[]
    for yr in region_masks.time.values:
        region_year_mask = xr.where(region_masks.sel(time=yr) ==0, ## locate valid pxs
                                    yr, # fill with year value 
                                    np.nan # nan where nodata 
                                    )
        data_available_year.append(region_year_mask)
    data_available_year = xr.concat(data_available_year, dim='time')
    
    if masktype == '2021': # ('annual','any','15-21','2021')
    # if timeframe == 'longterm':
        print('.. applying 2021 nodata mask  (NB: only used for 1997 and 2021 lowres data)')
        save_subdir = '_aggregated_with_nodataMask_2021_1000m'

        ''' --------------------------------------
        Apply masks: apply 2021 mask to both 2021 and 1997
        ------------------------------------------ '''
        # ## Option 1: Apply mask to BOTH 1997 (ramp) AND 2021 (S1) (select nodata mask of 2021)
        region_ds = region_ds.where(region_masks.sel(time=2021) == 0) # if the mask has value 0 it means VALID data (this is counter intuitive, forgive me)


    elif masktype == '15-21':
        print('.. applying 15-21 nodata masks')

        ''' --------------------------------------
        Apply masks: where S1 has nodata for all years
        -- apply mask to 1997; and also 2015-2021 
            (for the latter, this does not actually remove information; 
            but counts px values differently as it properly excludes nodata pixels)
        -- NB: this is skipped here, as I'll fill it with a nodatavalue furhter down
        ------------------------------------------ '''
        save_subdir = '_aggregated_with_nodataMask_15-21'
        print('.. Applying S1 mask to RAMP')
        ymax = len(region_masks.time.values)
        region_ds = region_ds.where( np.isnan(data_available_year).sum(dim='time') < ymax ) # puts nan in out-of-tile areas (where mask is NaN instead of 0/1)
    
    elif masktype == 'annual':
        print('.. applying annual nodata masks (for RAMP and MOA: apply 15-21 mask')

        ''' --------------------------------------
        Apply masks ANNUAL: 
        - 1997: apply a single mask that covers areas where none of S1-years hahve datta (2015-2021)
        - MOA 2004;2009;2014: apply a single mask that covers areas where none of S1-years hahve datta (2015-2021)
        - 2015-2021: apply annual mask of thhat year. 
        ------------------------------------------ '''
        save_subdir = '_aggregated_with_nodataMask_annual'
        
        region_years_masked = []
        for year in year_list:
            if year in [1997, 2000, 2004, 2009, 2014]: 
                print('.. Applying S1 15-21 mask to RAMP/MOA ({})'.format(year))
                ymax = len(region_masks.time.values)
                # all of them have nodata, the sum(mask) == ymax at those areas
                # so all pxiels that have any valid value in the time range, have sum(mask) < ymax
                region_1997 = region_ds.sel(time=year).where( np.isnan(data_available_year).sum(dim='time') < ymax ) # puts nan in out-of-tile areas (where mask is NaN instead of 0/1)
                region_other = region_ds.drop_sel(time=year)
                region_ds = xr.concat([region_1997, region_other],dim='time')
                region_years_masked.append(region_1997)
            else: 
                print('.. Applying {} mask to S1'.format(year))
                region_year = region_ds.sel(time=year).where(region_masks.sel(time=year) == 0)  # if the mask has value 0 it means VALID data (this migh be counter intuitive, forgive me)
                region_years_masked.append(region_year)
        region_ds = xr.concat ( region_years_masked, dim='time' )
        
    elif masktype == 'any':
        print('.. applying ANY (strict) nodata mask (all data)')
        ''' --------------------------------------
        Apply masks STRICT: 
        - all years: apply mask that covers area where ANYyear has nodata -- so that nodata area is the same allways
                     This means that some ice shelves will be removed. But it also removes all artifical jumps.
        ------------------------------------------ '''
        save_subdir = '_aggregated_with_nodataMask_any'
        
        region_years_masked = []
        for year in year_list:
            # Mask if all of them have nodata, the sum(mask) == ymax at those areas; 
            # All pxiels that have any valid value in the time range, have sum(mask) < ymax
            # Masks for ANY px having nodata: the sum(mask) == 0
            print('.. Applying strict any(nodata) mask to {}'.format(year))
            region_year = region_ds.sel(time=year).where( np.isnan(data_available_year).sum(dim='time') == 0 )  # if the mask has value 0 it means VALID data (this migh be counter intuitive, forgive me)
            region_years_masked.append(region_year)
        region_ds = xr.concat ( region_years_masked, dim='time' )
    
    else:
        raise ValueError('Handle aggregation without nodataMask not implemented')

    
    ''' --------------------------------------
    Fill nodata: 
    Mask areas where all years have nodata, and fill those with -999 
        - Then these pixels are counted as 'nodata' in aggregated values
        - only relevant for some iceshelves like ROSS and Amery
    ------------------------------------------ '''
    
    ''' 2. Mask areas where all years have nodata, and fill those with -999 '''
    ## Mask wehere all years have nodata (only relevant for some iceshelves like ROSS)
    # And fill with -999
    region_ds =  region_ds.where( np.isnan(data_available_year).sum(dim='time') < ymax , -999) # mask and fill in 1 step


    ''' Interpolate NaN values of Sentinel-1 data (temporal)
    Update: changed order. First filled all-yr-nodata with -999, then interpolate (so interpolation is presumably faster)'''
    interpol_yrs = False
    fill2val = False
    # if interpol_yrs:
    #     print('..interpolating temporal gaps')
    #     start = time.time()
    #     ## INTERPOLATE S1-NAN VALUES: 
    #     # interpolates nodata-areas between available years (temporal) 
    #     # Extents/extrapolates for where only one-sided info. 
    #     region_ds_s1 = region_ds.drop_sel(time=1997).chunk(dict(time=-1)) 
    #     region_ds_s1 = region_ds_s1.interpolate_na(dim='time',method='linear',fill_value="extrapolate")
    #     region_ds = xr.concat([region_ds.sel(time=1997), region_ds_s1],dim='time')

    #     save_subdir = '_aggregated_with_nodataMask_annual_interpol'
    #     end = time.time()
    #     print(".. done with interpolation " + str( np.round((end - start)/60 ,1) ) + 'min')
    # # else:
    # #     save_subdir = '_aggregated_with_nodataMask_annual'
        
    if fill2val:
        ''' 1b '''
        save_subdir += '_fill2values'
        region_ds = region_ds.fillna(-888)
        print('.. filled nodata values with -888 ')

    ''' --------------------------------------
    SELECT ICESHELVES IN REGION - per year
    ------------------------------------------ '''

    path2save = os.path.join(homedir,'Data/NERD/dmg_aggregated/') 
    path2save = os.path.join(path2save, save_subdir) # for datamask approach

    print('Saving to: ', path2save)

    for yidx, year in enumerate(year_list):
        print('--')
        print(year)

        if timeframe:
            df_filename = 'aggregated_dmg_per_iceshelf_' + region_ID +  '-' + str(year) + '_1000m.shp'
        else:
            df_filename = 'aggregated_dmg_per_iceshelf_' + region_ID +  '-' + str(year) + '_' + resolution + '.shp'

        # check if file exists:
        ## Chekc if predicted-dmg file already exists
        if os.path.isfile( os.path.join(path2save,df_filename ) ):
            print('File {} already exists  -- continue'.format(df_filename))
            continue

        # 2a. select annual polygon file
        iceshelf_year = ishelf_dict[str(year)]

        # 3. get annual ice shelves for current region
        ishelf_region = gpd.sjoin( iceshelf_year ,region_polygon)

        # 4. calculate values per ice shelf per year
        iceshelf_names = list(ishelf_region['NAME'].unique())
        iceshelves_stack=[]
        for iceshelf_name in iceshelf_names: 
            start = time.time()
            # print(iceshelf_name)
            
            # Select single ice shelf from dataframe
            ishelf_gpd = ishelf_region[ishelf_region['NAME'] == iceshelf_name]    

            ## set up dataframe with correct columns: 
            ## use MEASURES dataset for metadata as it has single entries for iceshelves
            ## (nb: aggregated values ARE calculated based on the annually evolving polygons)
            ishelf_current_year = iceshelf_poly_meas[iceshelf_poly_meas['NAME'] == iceshelf_name].copy()
            if len(ishelf_current_year) == 0: # name not found in MEAUSRES dataset
                # copy entry from measures-greene
                ishelf_current_year = ishelf_region[ishelf_region['NAME'] == iceshelf_name].copy()

            if iceshelf_name == 'Fox':
                if region_ID == 'BSE':
                    ishelf_current_year = ishelf_current_year[ishelf_current_year['Regions'] == 'West']
                elif region_ID == 'WA' or region_ID == 'EIS':
                    ishelf_current_year = ishelf_current_year[ishelf_current_year['Regions'] == 'East']
            if iceshelf_name == 'Ross_West': 
                if 'Ross_East' in iceshelf_names: # if both Ross_west and Ross_east, skip one of them,
                    print('Skipping Ross_West in favor of Ross_East (both have same full Ross geom)')
                    continue
            elif iceshelf_name == 'Ross_East': # Only process Ross-East,  but use the single 'Ross' geometry rather than the measures 'West/East' split                
                ishelf_current_year = ishelf_region[ishelf_region['NAME'] == iceshelf_name].copy() # copy entry from measures-greene
                ishelf_current_year['NAME']='Ross' # reset name to full geom
            if iceshelf_name == 'Getz' and region_ID == 'tbd2':
                continue

            ## Drop some columns
            for col in ['index_right','testField','TYPE','idxMEaS_df','Area','id']:
                try:
                    ishelf_current_year.drop(col,axis=1,inplace=True)
                except:
                    pass
            
            ''' ----------------
            Handle NODATA areas
            -------------------- '''

            ## CLIP data to iceshevles & select year 
            # using DROP=TRUE all pixels outside of iceshelf boundary are set to NaN. 
            # All nodata pixels inside are kept as -999. 
            ishelf_ds  = region_ds.sel(time=[year]).rio.clip( 
                                ishelf_gpd.geometry, ishelf_gpd.crs, 
                                drop=True, invert=False)
            


            ''' ----------------
            Aggregate values to 1d
            -------------------- '''

            ## Aggergate: Stack samples to 1D 
            # By clipping data to iceshelf, all out-of-bound pixels are set to NaN
            # I set all nodata pixels to -999, so these are not dropped as long as they are inside icehself polygon
            # So can safely drop NaN pxs here to reduce data in datafrme.
            with dask.config.set(**{'array.slicing.split_large_chunks': False}): 
                var_ds_stack = ishelf_ds.stack(samples=['x','y']) 
                var_ds_stack  = var_ds_stack.dropna(dim='samples',how='all') 

            # Calculate Spatial average ! Calculate this for dmg>0
            # (also making sure to exclude fillvalue -999)
            ishelf_avg = var_ds_stack.where(var_ds_stack>0).mean(dim='samples',skipna=True) # time series of spatial mean
            ishelf_sum = var_ds_stack.where(var_ds_stack>0).sum(dim='samples',skipna=True) # time series of spatial sum
            ishelf_Npx = var_ds_stack.count(dim='samples')  # px count in current ice shelf (only pxs with valid value)
            count_D   = var_ds_stack.where(var_ds_stack>0 ).count(dim='samples') # 105385
            count_noD = var_ds_stack.where(var_ds_stack==0 ).count(dim='samples') # 388230
            count_nan = var_ds_stack.where(var_ds_stack==-999 ).count(dim='samples') # nodata all years
            count_gap = var_ds_stack.where(var_ds_stack==-888 ).count(dim='samples') # nodata annual gap

            ''' Convert to dataframe (Loads into memory -- can be very slow) '''
            if 'labdata' in homedir: # rename coords otherwise error with reset_index
                # data_pxs_df = data_pxs_df.rename({'x':'x_coord','y':'y_coord'})
                dmg_iceshelf_year_df = var_ds_stack.to_dataframe().droplevel(['x','y']).reset_index('time').drop(['spatial_ref'],axis=1)
            else:
                dmg_iceshelf_year_df = var_ds_stack.to_dataframe().reset_index(level=['time','x','y']).drop(['spatial_ref'],axis=1)
            # print('.. conversion finished :) ')

            ## Discretize dmg
            dmg_classes2 = np.array([-0.001, 0,  0.0125, 0.0625, 0.1625, 0.3125])
            dmg_class_labels = ['no damage', 'low', 'medium','high' ,'very high']
            # Update for correct inclusive righter bin edge. Has not been used for 400m-aggr values.
            dmg_classes2 = np.array([-0.001, 0,  0.0125, 0.0625, 1.1]) 
            dmg_class_labels = ['no damage', 'low', 'medium','high' ]
            dmg_iceshelf_year_df['dmg_discr'] , cut_bin1 = pd.cut(x = dmg_iceshelf_year_df[dmg_type], 
                                            bins = dmg_classes2, labels=dmg_class_labels,
                                            include_lowest = True, # includes lowest bin. NB: values below this bin are dropped (e.g. the -999 nodata value)
                                            retbins=True, right=True) # [(-0.001, 0.025] < (0.025, 0.1] < (0.1, 0.225] < (0.225, 0.4]]

            dCounts = dmg_iceshelf_year_df['dmg_discr'].value_counts()
            # check dmg=0 count
            if not count_noD[dmg_type].values[0] == dCounts['no damage']:
                ## The lowest, 'no damage' bin has edges -0.001 and 0; 
                # so if nodata has a fill value of -999 instead of NaN, these are binned and counted as well
                # and the 'no damage count' is not the same
                print('.. [W]: px count of dmg=0 is different calculated from dataset ({}) than from binned dataframe ({}); this happens when using a fill-value for nodata'.format(
                            count_noD[dmg_type].values[0], dCounts['no damage'], 
                ))
                
            

            ''' --------------------------------------
            Store values per ice shelf
            ------------------------------------------ '''
            
            ishelf_current_year['Dsum']  = ishelf_sum[dmg_type].values # spatial sum (D>0)
            ishelf_current_year['Npx']  = ishelf_Npx[dmg_type].values # number of no-damage px (counts only valid values)
            ishelf_current_year['#DMG']  = count_D[dmg_type].values # number of damage px
            ishelf_current_year['#noDMG'] = count_noD[dmg_type].values # dCounts['no damage'] # number of no-damage px
            if fill2val:
                ishelf_current_year['#nodata-all'] = count_nan[dmg_type].values # number of nodata px ALL YRS
                ishelf_current_year['#nodata-gap'] = count_gap[dmg_type].values # number of nodata px ANNUAL GAP
            else:
                ishelf_current_year['#nodata'] = count_nan[dmg_type].values # number of nodata px
            ishelf_current_year['#lowDMG'] = dCounts['low'] 
            ishelf_current_year['#mediumDMG'] = dCounts['medium'] # number of pixels in medium dmg class
            ishelf_current_year['#highDMG'] = dCounts['high'] 
            # ishelf_current_year['#vHighDMG'] = dCounts['very high']

            # list of ice shelf entrys (rows): for 
            iceshelves_stack.append(ishelf_current_year)

            ## print time processing of iceshelf
            if interpol_yrs:
                end = time.time()
                print(".. Done with " + iceshelf_name + ': '+ str( np.round((end - start)/60 ,1) ) + 'min')

        ''' --------------
        Assemble data into a dataFrame 
        - every entry is an ice shelf feature
        - one dataFrame per year
        ------------------ '''

        # Merge all entrys to single dattaFrame
        iceshelf_new_df = pd.concat(iceshelves_stack)

        iceshelf_new_df.head()


        ''' --------------
        Save dataframe: ICE SHELVES PER REGION, per year
        ------------------ '''
        
        print('Saving to: ', df_filename)
        iceshelf_new_df.to_file(os.path.join(path2save, df_filename), index=False,crs='EPSG:3031')



if __name__ == '__main__':

    ''' ---------------------------------------------------------------------------
    Command Line configuration
    Run script as "python path/to/script.py --sector sector_ID  --....

    -------------------------------------------------------------- '''
    
    # if optional arguments are not specified in command line, they are set to None
    # required arguments will throw a usage error
    args = parse_cla()
    region_ID = None
    sector_ID = args.sector
    dmg_type = args.dtype # if args.dtype is not None else 'dmg095'
    timeframe = args.timeframe
    year = args.year
    resolution = args.resolution
    masktype = args.masktype
    strict_type = args.strict if args.strict != 'False' else None
    
    main(region_ID ,sector_ID, dmg_type=dmg_type, 
            timeframe=timeframe, year=year, 
            resolution=resolution, masktype=masktype, strict_type=strict_type)   
