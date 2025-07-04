import os
import geopandas as gpd
import numpy as np
import glob
import xarray as xr

import pandas as pd 
import re
import argparse

import myFunctions as myf 
import dask
import time



''' --------------
Set Paths
------------------ '''
homedir = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/'



path2iceshelves = os.path.join(homedir,'Data/Greene2022_AIS_coastlines/shapefiles/annual_iceshelf_polygons/revised_measures_greene/')
iceshelf_path_meas = os.path.join(homedir, 'QGis/Quantarctica/Quantarctica3/Glaciology/MEaSUREs Antarctic Boundaries/IceShelf/IceShelf_Antarctica_v02.shp')
sector_path = os.path.join(homedir, 'QGis/data_NeRD/AIS_outline_sectors.shp')


''' --------------
Get Shapefiles 
------------------ '''

# measures ice shelves
iceshelf_poly_meas = gpd.read_file(iceshelf_path_meas)

## redefined: SECTORS for AIS
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

# iceshelf_dflist = [iceshelf_df_1997,iceshelf_df_2015,iceshelf_df_2016,
#                   iceshelf_df_2017,iceshelf_df_2018,iceshelf_df_2019,
#                   iceshelf_df_2020,iceshelf_df_2021]
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
        --sector (-s)     : define wich region to process
        
        Optional:
        --year (-y)        : Specify if to process a single year; process all years if unspecified.
        --resolution (-res): specify which type of processing resolution the input data is from. High res: 400m. Low res: 1000m. Downsampled: 400m donwsampled to 1000m.

    :return args: ArgumentParser return object containing command line arguments
    """
    
    # Required arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector","-s",help='Specify sector to process',type=str, required=True,
                            choices=('ASE', 'BSE', 'EIS', 'RS', 'WIS-a','WIS-b', 'WS', 'WS-a','WS-b') )  
                     

    # Optional arguments
    group2 = parser.add_mutually_exclusive_group(required=False) 
    group2.add_argument("--year","-y",help='Specify year to process. Unspecified means all years',type=str )  
    parser.add_argument('--resolution','-res',help='Processed resolution',type=str,required=False,
                            # choices=('400m','1000m'),
                            default='1000m')
    
    parser.add_argument('--dmgvar','-v',help='dmg varname, defaults to damage095. ',type=str,required=False,
                            # choices=('damage095','damage095_lowerbound','damage095_upperbound'),
                            default='damage095')

    parser.add_argument('--strict',help='calculate stricter dmg values (for uncertainty) ',type=str,required=False,
                            default='False')
    args = parser.parse_args()
    return args 


def setup_iceshelf_df_entry(iceshelf_name, ishelf_region, sector_ID ):
    ishelf_current_year = iceshelf_poly_meas[iceshelf_poly_meas['NAME'] == iceshelf_name].copy()
    if len(ishelf_current_year) == 0: # name not found in MEAUSRES dataset
        # copy entry from measures-greene
        ishelf_current_year = ishelf_region[ishelf_region['NAME'] == iceshelf_name].copy()

    if iceshelf_name == 'Fox':
        if sector_ID == 'BSE':
            ishelf_current_year = ishelf_current_year[ishelf_current_year['Regions'] == 'West']
        elif sector_ID == 'WA' or sector_ID == 'EIS':
            ishelf_current_year = ishelf_current_year[ishelf_current_year['Regions'] == 'East']
    ## Drop some columns
    for col in ['index_right','testField','TYPE','idxMEaS_df','Area','id']:
        try:
            ishelf_current_year.drop(col,axis=1,inplace=True)
        except:
            pass
    return ishelf_current_year

''' --------------------------------------
 SELECT REGION, LOAD DATA
------------------------------------------ '''

def main( sector_ID, year=None, resolution='1000m', dmgvar='damage095' ,strict_type=None):

    # path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/data_sector/')
    path2data = os.path.join(homedir, 'Data/NERD/data_organise/v0_forPlots/netcdf/')
    
    # Path to save dataframe
    # path2save = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/')
    path2save = os.path.join(homedir, 'Data/NERD/data_organise/v0_forPlots/reproduced_newscript/')


    if resolution == '400m':
        ## load S1 400m dmg
        filelist_dmg =  glob.glob( os.path.join(path2data,'damage095',resolution, f'dmg_sector-{sector_ID}_*.nc') )
        filelist_dmg = [file for file in filelist_dmg if '1997' not in file]  # make sure 1997 is not included; as not processed at 400m res
        filelist_dmg.sort()
        
    elif '1000m' in resolution: # resolution == '1000m': 
        filelist_dmg =  glob.glob( os.path.join(path2data,dmgvar,resolution, f'dmg*-{sector_ID}_*.nc') ) ## can be both dmg0995 or damage095_lowerbound
        filelist_dmg.sort()
    else: # just get files from input path
        filelist_dmg =  glob.glob( os.path.join(path2data, f'*-{sector_ID}_*.nc') ) 
        filelist_dmg.sort()
    
    ## tmp for uncertainty (from different dmg maps)
    if 'upperbound' in dmgvar or 'lowerbound' in dmgvar:
        path2save = os.path.join(path2save,dmgvar)

    ## tmp: other uncertainty, calculate here.
    # stricter_type = 'd099'
    # stricter_type = 'pct025'
    if strict_type is not None:
        path2save = os.path.join(path2save,f'stricter-{strict_type}')
    
    
    if year is None: # all years
        ## load list of files
        filenames = [os.path.basename(file) for file in filelist_dmg]
        ## retrieve available years from filenames
        year_list = [int( re.search(r'\d{4}', file).group()) for file in filenames]
    else: # filter filelist for desired year
        filelist_dmg = [file for file in filelist_dmg if year in file]
        year_list=[year]
        if not filelist_dmg:
            raise ValueError(f'Could not find year {year}')

    # read normal dmg095 (v0)
    region_ds = (xr.open_mfdataset(filelist_dmg ,
                    combine='nested', concat_dim='time',
                    compat='no_conflicts',
                    preprocess=myf.drop_spatial_ref)
            .rio.write_crs(3031,inplace=True)
            .assign_coords(time=year_list) # update year values (y,x,time)
    )
    
    # ## read normal dmg095 from function (v0 files)
    ## region_ds = myf.load_nc_sector_years( os.path.join(path2data,dmgvar,resolution) , sector_ID, varName = 'dmg' ,year_list = year_list)
    # print(region_ds.dims, region_ds)
    
    
    # print(region_ds)
    # for file in filelist_dmg:
    #     with xr.open_mfdataset(file) as ds:
    #         print(file, ds.dims)


    ## ## for uncertainty dmg / new batch processed downsampling dmg095
    ## region_ds_ramp = myf.load_nc_sector_years( os.path.join(path2data,dmgvar,resolution) , sector_ID, varName = 'dmg' ,year_list = [1997, 2000])
    ## region_ds = myf.load_nc_sector_years( os.path.join(path2data,dmgvar,resolution) , sector_ID, varName = 'dmg' ,year_list = [2015,2016,2017,2018,2019,2020,2021])
    ## region_ds_ramp = myf.reproject_match_grid(region_ds,region_ds_ramp)
    ## region_ds = xr.concat([region_ds_ramp, region_ds], dim='time')
    ## # print(region_ds)

    print('--- dmg dataset ----')
    year_list = list(region_ds.time.values)
    if len( list(region_ds.data_vars) )>1:
        raise ValueError('Expected 1 variable, got {}'.format( list(region_ds.data_vars) ))
    dmg_type = list(region_ds.data_vars)[0]
    print('year_list: ', year_list)

    ## update dmg threshold for uncertainty
    if strict_type is not None:
        ## apply stricter dmg threshold to dmg map, to get a range of uncertainty
        # (a) use d99 trheshold
        # (b) remove lowest 25pct 
        
        ## A:
        if strict_type == 'd099':
            print('..updating dmg threshold with d099 thresholds')
            annual_threshold = [0.02, 0.029, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046] #pct99 noise thresholds for respective sensor sources (100:10px)
            time_values     = [1997, 2000, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
        ## B
        if strict_type == 'pct25':
            print('..updating dmg threshold by removing lowest 25pct pxs')
            annual_threshold = [ 0.004854, 0.006818, 0.004863, 0.004865, 0.004867, 0.004867, 0.004867, 0.004865, 0.004876 ]
            time_values         = [1997,    2000,       2015,        2016,  2017,       2018,   2019,       2020,   2021]
        ## C
        if strict_type == 'd095': ## re-apply d095 trehshold (after taking median-dmg there are pixels below the threshold)
            print('..updating dmg threshold by re-applying d095 thhreshold')
            annual_threshold = [ 0.016, 0.023, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038 ]
            time_values         = [1997,    2000,       2015,        2016,  2017,       2018,   2019,       2020,   2021]
        ## B
        if strict_type == 'pct05':
            print('..updating dmg threshold by removing lowest 5pct pxs')
            annual_threshold = [0.000945, 0.000953, 0.000971, 0.000985, 0.000989, 0.000984, 0.000985, 0.000981, 0.000985]
            time_values         = [1997,    2000,       2015,        2016,  2017,       2018,   2019,       2020,   2021]

        # Update each time slice with the respective threshold
        slicelist = []
        for t, threshold in zip(time_values, annual_threshold):
                tslice = region_ds.sel(time=t).where(region_ds.sel(time=t)>threshold,other=0)
                slicelist.append(tslice) 
        region_ds = xr.concat(slicelist,dim='time')  

    # ## fill no-damage nan with 0 values, so thhat i only mask nodata-areas (and later also clip to ice shelves) 
    region_ds= region_ds.fillna(0)

    ''' --------------------------------------
    Load masks: all years of Sentinel-1
    Masks have value 1 for px with NODATA and value 0 for px with valid data (strange maybe)
    Masks are only provided for Sentinel1 2015-2021 data; as RAMP has data in all tiles.
    ------------------------------------------ '''

    print('--- mask dataset ----')  
    ## locate files
    path2mask = os.path.join(homedir, 'Data/NERD/dmg095_nc/data_sector/')
    # mask_filelist = glob.glob( os.path.join(path2data,'nodata', f'nodata_sector-{sector_ID}_*_400m.nc') ) # masks only avail at 400m
    # update: read geotiff due to projection erorr in netcdf of 2015
    mask_filelist = glob.glob( os.path.join(path2mask,'nodata','geotiff', f'nodata_sector-{sector_ID}_*_400m.tif') )
    mask_filelist.sort()
    ## retrieve years from filenames
    filenames = [os.path.basename(file) for file in mask_filelist]
    mask_years = [int( re.search(r'\d{4}', file).group()) for file in filenames]
    ## load netcdf
    region_masks = (xr.open_mfdataset( mask_filelist,
                            combine='nested', concat_dim='time',
                            compat='no_conflicts',
                            preprocess=myf.drop_spatial_ref,
                            )
                        .rio.write_crs(3031,inplace=True)
                        .assign_coords(time=mask_years) # update year values (y,x,time)
    ) 
    region_masks = region_masks.rename({'band_data':'nodata'}).isel(band=0) # fix some stuff when reading from tiff instead of netcdf

    ## Resample mask to same grid as dmg 
    # -- for masks, resampling does something weird wth coords. Can only coontinue with dataArray insteaad of dataSet
    region_masks = myf.reproject_match_grid(region_ds[dmg_type], region_masks['nodata'])
    ymax = len(region_masks.time.values) # number of available years

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
        # print(yr, region_year_mask.dims)
        data_available_year.append(region_year_mask)
    data_available_year = xr.concat(data_available_year, dim='time')
    
    print('mask years ', mask_years ) # data_available_year.dims)
       
    ''' --------------------------------------
    Apply masks STRICT: 
    All years: apply mask that covers area where ANYyear has nodata -- so that nodata area is the same allways
    This means that some ice shelves will be removed. 
    # If you want to mask only the areas where all S1-years have nodata, use  sum(mask) == ymax ;
    # If you want to mask all areas where any S1-year has nodata: the sum(mask) == 0
    ------------------------------------------ '''
    
    ## region_years_masked = []
    ## for year in year_list:
    ##     print('.. Applying strict any(nodata) mask {}'.format(year))
    ##     region_year = region_ds.sel(time=year).where( np.isnan(data_available_year).sum(dim='time') == 0 )  # if the mask has value 0 it means VALID data (counter intuitive, i'm sorry)
    ##     region_years_masked.append(region_year)
    ## region_ds = xr.concat ( region_years_masked, dim='time' )
    ## print(region_ds[dmg_type].shape, region_ds.dims)
    
    # count_nodata_px = np.isnan(data_available_year).sum(dim='time') # if the mask has value 0 it means VALID data (counter intuitive, i'm sorry)
    count_nodata_px = (region_masks).sum(dim='time') ## count > 0 if there is any year that has nodata. count==0 means all years have VALID data

    region_ds = region_ds.where( count_nodata_px == 0 )
    


    
    ''' --------------------------------------
    Fill/interpolate annual nodata:
    1. Fill nodata pixels with a different fill value (e.g -999)
        - Then these pixels are counted as 'nodata' in aggregated values
        - only relevant for some iceshelves like ROSS and Amery
    2. (Optional) Interpolate NaN values of Sentinel-1 data (temporal)
        - do not interpolate using RAMP 1997 data; only 2015-2021
        - extrapolate values where only one-sided values available (ie 2015 or 2021)
        - NB: very computationally expensive (needing to interpolate every px on Antarctic scale in time..)
    ------------------------------------------ '''
    
    ''' 1. Fill annual nodata areas with a dstinctive value
        By using count_nodata_px < ymax, all instances of a px having nodata for a certain year are captured
        UPDATE: oops should be count_nodata_px > 0 '''
    ## region_ds =  region_ds.where( count_nodata_px < ymax , -999) # mask and fill in 1 step; 
    region_ds =  region_ds.where( count_nodata_px == 0 , -999) # mask and fill in 1 step; 

    ''' 2. Interpolate NaN values of Sentinel-1 data (temporal)
    Update: changed order. First filled all-yr-nodata with -999, then interpolate (so interpolation is presumably faster)'''
    interpol_yrs = False
    if interpol_yrs:
        print('..interpolating temporal gaps')
        start = time.time()
        ## INTERPOLATE S1-NAN VALUES: 
        # interpolates nodata-areas between available years (temporal) 
        # Extents/extrapolates for where only one-sided info. 
        region_ds_s1 = region_ds.drop_sel(time=1997).chunk(dict(time=-1)) 
        region_ds_s1 = region_ds_s1.interpolate_na(dim='time',method='linear',fill_value="extrapolate")
        region_ds = xr.concat([region_ds.sel(time=1997), region_ds_s1],dim='time')
        save_subdir = '_aggregated_with_nodataMask_annual_interpol'
        end = time.time()
        print(".. done with interpolation " + str( np.round((end - start)/60 ,1) ) + 'min')
    else:
       save_subdir = '_aggregated_with_nodataMask_any/_perSector/' 
       if not os.path.exists(os.path.join(path2save,save_subdir)):
            print('subdir doesnt exsit, saving to original path2save dir')
            save_subdir=''

    # raise RuntimeError('stop -dev')

    ''' --------------------------------------
    SELECT ICESHELVES IN REGION - per year
    ------------------------------------------ '''

    path2save = os.path.join(path2save, save_subdir)
    
    # 1 Get sector polygon
    region_polygon = sector_poly[sector_poly['sector_ID'] == sector_ID]

    # raise RuntimeError('stop -dev')

    for yidx, year in enumerate(year_list):
        print('--')
        print(year)

        # df_filename = 'aggregated_dmg_per_iceshelf_' + sector_ID +  '-' + str(year) + '.shp'
        df_filename = f'aggregated_dmg_per_iceshelf_{sector_ID}-{str(year)}_{resolution}.shp'
        print(df_filename)

        # check if file exists:
        ## Chekc if predicted-dmg file already exists
        if os.path.isfile( os.path.join(path2save,df_filename ) ):
            print('File {} already exists  -- continue'.format(df_filename))
            continue

        # 2a. select annual polygon file
        # iceshelf_year = iceshelf_dflist[yidx]
        iceshelf_year = ishelf_dict[str(year)] 

        # 2b. filter ice shelves for current region
        ishelf_region = gpd.sjoin( iceshelf_year ,region_polygon)

        # 2c. Small correction to avoid duplicate processing
        iceshelf_names = list(ishelf_region['NAME'].unique())
        if 'Ross_East' in iceshelf_names and 'Ross_West' in iceshelf_names: # if both Ross_west and Ross_east, skip one of them,
            print('Skipping Ross_West in favor of Ross_East (both have same full Ross geom)')
            iceshelf_names.remove('Ross_West')
            # print(iceshelf_names)
        
        # 3. calculate values per ice shelf per year
        iceshelves_stack=[]
        for iceshelf_name in iceshelf_names: 
            start = time.time()
            # print(iceshelf_name)
            
            # Select single ice shelf from dataframe
            ishelf_gpd = ishelf_region[ishelf_region['NAME'] == iceshelf_name]    

            '''
            ## set up dataframe with consistent names/ columns: 
            Use MEASURES dataset for metadata as it has consistant, single entries for iceshelves 
            to simplify retreatred ice shelves such as Wordie that retreated into multiple ice shelves wiith different names 
            NB: aggregated values ARE calculated based on the annually evolving polygons; but the saved dataframe will have MEAsures geometry
            '''
            ishelf_current_year = setup_iceshelf_df_entry(iceshelf_name,ishelf_region,sector_ID)

            
            ''' ----------------
            Clip data to annual polygon
            # using DROP=TRUE all pixels outside of iceshelf boundary are set to NaN. 
            # All nodata pixels inside are kept as -999. 
            -------------------- '''
            ishelf_ds  = region_ds.sel(time=[year]).rio.clip( 
                                ishelf_gpd.geometry, ishelf_gpd.crs, 
                                drop=True, invert=False)
            
            ''' ----------------
            Aggregate values to 1d
            
            By clipping data to iceshelf, all out-of-bound pixels are set to NaN
            I set all nodata pixels to -999, so these are not dropped as long as they are inside icehself polygon
            So can safely drop NaN pxs here to reduce size of datafrme.
            -------------------- '''
            ## Aggergate: Stack samples to 1D 
            with dask.config.set(**{'array.slicing.split_large_chunks': False}): 
                var_ds_stack = ishelf_ds.stack(samples=['x','y']) 
                var_ds_stack  = var_ds_stack.dropna(dim='samples',how='all') 

            ## Sum or count values
            ishelf_sum = var_ds_stack.where(var_ds_stack>0).sum(dim='samples',skipna=True) # time series of spatial sum
            # ishelf_Npx = var_ds_stack.count(dim='samples')  # px count in current ice shelf (only pxs with valid value)
            ishelf_Npx = var_ds_stack.where(var_ds_stack!=-999).count(dim='samples') # count valid pxs, defined by !=-999 value (all NaN pxs were already dropped before)
            count_D   = var_ds_stack.where(var_ds_stack>0 ).count(dim='samples') # count damaged pixels
            count_noD = var_ds_stack.where(var_ds_stack==0 ).count(dim='samples') # count no-damaged pixels
            count_nan = var_ds_stack.where(var_ds_stack==-999 ).count(dim='samples') # cound nodata pixels

            ''' Convert to dataframe (Loads into memory -- can be slow) '''
            try:
                dmg_iceshelf_year_df = var_ds_stack.to_dataframe().reset_index(level=['time','x','y']).drop(['spatial_ref'],axis=1)
            except: # if error with reset_index, drop coords first
                dmg_iceshelf_year_df = var_ds_stack.to_dataframe().droplevel(['x','y']).reset_index('time').drop(['spatial_ref'],axis=1)

            ## Discretize dmg
            dmg_classes2 = np.array([-0.001, 0,  0.0125, 0.0625, 0.1625, 0.3125]) # start with negative bounds to get "d=0" as a single class 
            dmg_class_labels = ['no damage', 'low', 'medium','high' ,'very high']
            # Update for correct inclusive righter bin edge.
            dmg_classes2 = np.array([-0.001, 0,  0.0125, 0.0625, 1.1]) 
            dmg_class_labels = ['no damage', 'low', 'medium','high' ]
            dmg_iceshelf_year_df['dmg_discr'] , cut_bin1 = pd.cut(x = dmg_iceshelf_year_df[dmg_type], 
                                            bins = dmg_classes2, labels=dmg_class_labels,
                                            include_lowest = True, # includes lowest bin. NB: values below this bin are dropped (e.g. the -999 nodata value)
                                            retbins=True, right=True) # [(-0.001, 0.025] < (0.025, 0.1] < (0.1, 0.225] < (0.225, 0.4]]

            dCounts = dmg_iceshelf_year_df['dmg_discr'].value_counts()
            
            ''' --------------------------------------
            Store values per ice shelf
            ------------------------------------------ '''

            ishelf_current_year['Dsum']  = ishelf_sum[dmg_type].values # spatial sum (D>0)
            ishelf_current_year['Npx']  = ishelf_Npx[dmg_type].values # number of px in ice shelf
            ishelf_current_year['#DMG']  = count_D[dmg_type].values # number of damage px
            ishelf_current_year['#noDMG'] = count_noD[dmg_type].values # # number of no-damage px. NB: use coount_noD and not dCounts['no damage'] , as dCounts['no damage'] also counts -999 and -888 ndoata values
            ishelf_current_year['#nodata'] = count_nan[dmg_type].values # number of nodata px
            ishelf_current_year['#lowDMG'] = dCounts['low'] 
            ishelf_current_year['#mediumDMG'] = dCounts['medium'] # number of pixels in medium dmg class
            ishelf_current_year['#highDMG'] = dCounts['high'] 

            # list of ice shelf entrys (rows): for 
            iceshelves_stack.append(ishelf_current_year)


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
    Run script as "python path/to/script.py --sector sector_ID
 
    Additional options:
    --resolution   :  read damage netcdfs with assessments at 400m or 1000m resolution (default: 1000m)
    --year         :  Can process a single year. If unspecified, will process all available years.

    -------------------------------------------------------------- '''
    
    # if optional arguments are not specified in command line, they are set to None
    # required arguments will throw a usage error
    args = parse_cla()
    sector_ID = args.sector
    year = args.year
    resolution = args.resolution
    dmgvar = args.dmgvar
    strict_type = args.strict if args.strict != 'False' else None

    main(sector_ID, 
            year=year, 
            resolution=resolution,
            dmgvar=dmgvar, strict_type=strict_type)   
