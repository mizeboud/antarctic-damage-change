import os
import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray as rio # needed for reprojections
import cftime as cft
import joblib 
import configparser
import json

# Import user functions
import myFunctions as myf 

homedir = '/Users/.../'


''' --------------
Load trained RF 
------------------ '''

path2predictor = os.path.join(homedir,'files/')
path2save = os.path.join(homedir,'Data/NERD/data_predictor/dmg_predicted/.../')

''' GridSrearch output '''

# RF trained on orginal data resolution (400m)
# model_name = 'RF_bestEstimator_gridSearch_2024-06-25T19:37.joblib'
model_name = 'RF_gridSearch_bestEstimator.joblib'

''' GridSearch output '''
## Downsampled 20x20, dres=pct095
# configFile = 'RFgcv_166399.ini' # renamed file
configFile = 'RF_gridSearch.ini'
iceshelves_testset_unbalanced = ['Borchgrevink','Totten','Nivl','Fimbul','Moscow_University','Hull',
 'Abbot_3' ,'LarsenD', 'Tucker', 'Harmon_Bay', 'Abbot_1', 'Brahms', 'Voyeykov',
 'Garfield' ,'Hummer_Point','PourquoiPas', 'Porter', 'Noll', 'Alison',
 'Withrow', 'Eltanin_Bay', 'Liotard' ,'Falkner','Manhaul']  # names based on MEASURES dataset
search_section = 'GRIDSEARCH'
space_section = 'GRIDSEARCHSPACE'
# predictor_name = model_name.split('_bestEstimator_')[0] + '_res20x20'
predictor_name = 'RFgcv_166399_2024-06-25_0x0'

loaded_rf = joblib.load(os.path.join(path2predictor,model_name))
feature_list = list(loaded_rf.feature_names_in_)


''' --------------
## Load relevant config settings
------------------ '''

config = configparser.ConfigParser(allow_no_value=True)
config.read(os.path.join(path2predictor,configFile))

## config for RandomForest randomSearchCV / gridSearchCV training
xvar_list = json.loads(config.get(search_section,"xvar_list")) # use json for reading lists
length_scales = config[search_section]['strain_length_scale_px'].split()
length_scales = [s.replace(',','').replace(' ','') for s in length_scales]  # remove any unintended remaining separatos
length_scales = [s + 'px' for s in length_scales]
dmg_type = config['DATA']['dmg_type']
dtresh_dict = {'dmg':'037', 'dmg095':'053' , 'dmg099':'063'}
dtresh=dtresh_dict[dmg_type]
ksize = int(config['DATA']['downsample_size'])


''' --------------
Get Shapefiles 
------------------ '''
# geojson
gridTiles_geojson_path = os.path.join(homedir,'Data/tiles/gridTiles_iceShelves_EPSG3031.geojson')
gridTiles = gpd.read_file(gridTiles_geojson_path)

# measures ice shelves
iceshelf_path_meas = os.path.join(homedir, 'QGis/Quantarctica/Quantarctica3/Glaciology/MEaSUREs Antarctic Boundaries/IceShelf/IceShelf_Antarctica_v02.shp')
iceshelf_poly_meas = gpd.read_file(iceshelf_path_meas)

# sectors of interest for AIS

sector_path = os.path.join(homedir, 'QGis/data_NeRD/AIS_outline_sectors.shp')
sector_poly = gpd.read_file(sector_path)
sector_ID_list = sector_poly['sector_ID'].to_list()
sector_ID_list.sort()
sector_ID_list

sector_ID_list = ['ASE', 'BSE', 'EIS', 'RS', 'WIS-a', 'WIS-b', 'WS']




''' Prediction function'''
def predict_dmg(loaded_rf, data_ds, feature_list, select_years=None):

    ''' --------------
    Select few years of data / all data
    ------------------ '''
    if select_years is not None:
        ## select range of years
        data_ds = data_ds.sel(time=[cft.DatetimeNoLeap(yr,1,1) for yr in select_years])

    '''## Get mask -- if based on dataSet, every variable will get its own mask.
    - (1) mask all variables separately to binary 1 for nan-values, 0 for non-nan
    - (2) sum the masks and get a single binary nan-maks for the whole dataset'''

    nan_mask = xr.where( np.isnan(data_ds) , 1, 0 ).to_array().sum("variable") # condition, values_if_true, values_if_false;  # stacks all variablesas new dimension, and then sums)
    # nan_mask = xr.where( nan_mask > 0 , 1, 0 ) # make binary -- also flattens (time,y,x) to (y,x), so do not do this

    '''
    ## Fill nan
    - want to be able to mask NaN data from predictions
    - for this, we use land and ocean mask provided by ISMIP data
    '''

    data_ds = data_ds.fillna(-999)


    ''' --------------
    Make predictions for selected year:
    - Stack x,y data to 1D dataArray (samples, ) per feature
    - Coombine multiple features into a single dataArray (samples, featurues)
    - Usa a dummy dataArray to plug predicted values in (enabling conversion back to 2D map)
    ------------------ '''
    ## stack 2D/3D data as 1D samples
    data_stack_ds = data_ds.stack(samples=['y','x','time']) ## dataset with each feature as a variable, of size (samples, ) for each feature

    for feat in feature_list:
        if feat not in list(data_stack_ds.keys()):
            raise Exception('Feature {} is not in dataset: {}'.format(feat, list(data_stack_ds.keys()) ) )


    ## convert to dataArrray for input (X) and target (y) features
    y_stack_da_dummy = xr.zeros_like(data_stack_ds[feat]) # setup dataArray structure for predicted values
    X_stack_da = (data_stack_ds[feature_list]             # adjust order of features to be expected order for RF
                    .to_array('feature').transpose('samples','feature') # A single dataArray from multiple variables using .to_array(). The stacked dimension will be the 'feature'.
    )
    X_stack_da # Single array of size (samples, N features) with each column corresponding to feature 1, feature 2, ..., feat N


    '''# Predict on stacked xr.DataArray'''

    y_rf_pred = loaded_rf.predict(  X_stack_da ) # is an array ipv dataArray
    y_pred_da = y_stack_da_dummy.copy(data=y_rf_pred).rename('y_predict').astype('float').assign_attrs(standard_name='predicted_dmg',units='') # put back as dataArray

    ## Unstack 
    y_pred_da = y_pred_da.unstack().transpose('time','y','x')

    ## apply nan-mask back to array (it could vary from year-to-year)
    y_pred_da =  y_pred_da.where(nan_mask==0, other=np.nan
                        ).rio.write_nodata(np.nan, encoded=True,inplace=True)  # properly encode nodata, before saving netcdf

    return y_pred_da


''' ----------------------

    LOAD OBSERVATIONAL DATA

------------------------- '''
# year_list=['2015','2016','2017','2018']
year_list=['2015','2016','2017','2018','2019','2020','2021']

ksize=None
ksize_str = '0x0' ## no downsampling of data: work at 400 m grid (highest resolution of damage maps )
# ksize=20; ksize_str='20x20'## downsampling of data to regularised ISMIP grid of 8000 m.

'''-----------------
## Load INPUT VARIABLES
---------------------'''


## data settings
path2data = os.path.join(homedir, 'Data/NERD/data_predictor/data_sector/velocity_rema/')



''' ----------------------
Load data: netCDFs per region, per variable
------------------------- '''

for sector_ID in sector_ID_list[:2]: # ['ASE']: 

    ## Construct save-filename
    fname_dmg_nc = f'data_sector-{sector_ID}_dmg_predicted_{ksize_str}.nc'
    
    ## Process region only if not done so yet
    if os.path.isfile( os.path.join( path2save,fname_dmg_nc ) ) :
        print('File {} already exists, continue'.format(fname_dmg_nc))
        continue
    

    print('----\n Loading netCDF for region ', sector_ID)


    ''' -------------------
    Load all variables from individual netCDF files 
    Expecting data directory to contain netCDFs per sector per training variable per year. 
    ----------------------- '''
    region_ds_varlist=[]
    for var in ['vx','vy']: # base variables to read, from which all other training features are calculated
        region_var = myf.load_nc_sector_years( path2data, sector_ID, varName=var, year_list=year_list) # load all available years
        region_ds_varlist.append(region_var)
    # load rema (only 1 year)
    region_var = myf.load_nc_sector_years( path2data, sector_ID, varName='rema', year_list=['0000']) # , year_list=years_train)
    region_ds_varlist.append(region_var)

    # combine to single dataset
    region_ds = xr.combine_by_coords(region_ds_varlist)
    print('Loaded variables: \n', list(region_ds.keys()) )#, region_ds.coords)
    

    ''' --------------------------------------
    Repeat temporally static variable (REMA) to even out dataset dimension
    This drops time=0
    ------------------------------------------ '''

    region_ds = myf.repeat_static_variable_timeseries( region_ds , 'rema' )

    ''' ## Interpolation of small nan-gaps of REMA data'''
    dx = int(region_ds.rio.resolution()[0]) # np.unique(region_ds['x'].diff('x').values)[0]
    nmax = 6
    # interpolate first X then Y (cannot do both -- so this skews the values. Maarja een mens moet wat.
    interpol = region_ds.chunk(dict(x=-1)).interpolate_na(dim='x', method='linear', limit=None, use_coordinate=True, max_gap=int(nmax*dx), keep_attrs=None)
    region_ds = interpol.chunk(dict(y=-1)).interpolate_na(dim='y', method='linear', limit=None, use_coordinate=True, max_gap=int(nmax*dx), keep_attrs=None)

    ''' ----------------
    Downsample observation data ( 400m to 8000m )
    --------------------'''

    if ksize:
        region_ds = myf.downsample_obs_data(region_ds, ksize)

    ''' ------------
    Calculate velocity and strain for (downsampled) data
    ---------------- '''
    # calculate velocity, strain components and temporal velo/strain change
    for lscale in length_scales:
            
        data_velo_strain, region_ds_roll = myf.calculate_velo_strain_features(region_ds, 
                                                    velocity_names=('vx','vy'), 
                                                    length_scales=[lscale])
        
        ## Fill the first temporal-difference timestep (2015) with the same value as 2015-16  so that this time slice doesnt get dropped later on
        da_delta_2015_new = region_ds_roll.sel(time=2016).assign_coords(time=2015) # select 2016 from dataset and assing it as time=2015
        region_ds_roll = xr.concat([da_delta_2015_new, region_ds_roll],dim='time') # replace it with the copied value

        ## add to dataSet
        region_ds = xr.merge([region_ds, data_velo_strain])
        region_ds = xr.merge([region_ds, region_ds_roll])

    ## Drop irrelevant variables from dataset
    region_ds = region_ds[ xvar_list] 

    ''' ------------------------------
    ## PREDICT
    ----------------------------------'''
    print('.. predicting .. ')
    feature_list_obs = loaded_rf.feature_names_in_ 

    # predict all years at once by setting select_years=None
    y_pred_rf_da_obs = predict_dmg(loaded_rf, region_ds, feature_list_obs, select_years=None)

    ''' --------------
    Save the dataset with all predicted dmg years 
    -- can then do analysis on output later :) 
    ------------------ '''

    ## add attributes:
    attrs_dict_dmg = {
        'title': 'Predicted damage',
        'prediction_model': predictor_name,
        'prediction_features': list(feature_list_obs),
    }
    y_pred_rf_da_obs.attrs = attrs_dict_dmg

    # Convert to xr dataset (to save to netcdf)
    dmg_pred_ds_region = y_pred_rf_da_obs.to_dataset(name='predicted_dmg',promote_attrs=True) 

    print('.. Saving {}'.format(fname_dmg_nc))
    dmg_pred_ds_region.to_netcdf( os.path.join(path2save,fname_dmg_nc ), 
                                mode='w', compute=True, engine='netcdf4')


print('Done')