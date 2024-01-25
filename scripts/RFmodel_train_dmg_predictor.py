# Script that trains&test RandomForest predictor and evaluates using spatial k-fold CrossValidation

import os
import rioxarray as rioxr
import geopandas as gpd
# import matplotlib.pyplot as plt
import numpy as np
# import glob
import xarray as xr
# import matplotlib.patches as mpatches

import rasterio as rio
import pandas as pd 

# import multiprocessing
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_tweedie_deviance
from joblib import Parallel, delayed
# from sklearn.model_selection import KFold


from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import GroupKFold
from joblib import dump, load

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Import user functions
import myFunctions as myf 

# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import PredictionErrorDisplay # needs sklearn >1.1.1 (I had 1.1.1 and updated to 1.2.2)
import time
import datetime
import configparser
import json
import ast
import sys
import glob
import dask
# import warnings
import re

''' ---------------------------
Functions
-------------------------------'''
# Perform spatial CV
# def train_validate_model_spatial(cluster, xvars, yvar, fold_type,
#                                 rf_n = 50, rf_max_depth=None):
#     # For current fold, split train/test                                
#     training_data = data_pxs_sample[data_pxs_sample[fold_type] != cluster]
#     validation_data = data_pxs_sample[data_pxs_sample[fold_type] == cluster]

#     # Random Forest model
#     # rf = RandomForestRegressor(n_estimators=300, n_jobs=-1) # cluster-loop (100s of clusters) with N=10.000 and res=9x9 took 5.5 hours. Will probably be faster for smaller n_estim 
#     rf = RandomForestRegressor( n_estimators= rf_n,  # # region-loop (3) with N=1000 , res 5x5 and n_est=50 takes 2 seconds.
#                                 n_jobs = -1,
#                                 max_depth = rf_max_depth,
#                                 random_state = 42) 
#     rf.fit(training_data[xvars], training_data[yvar])

#     # Prediction
#     pred_obj = rf.predict(validation_data[xvars])
#     pred_test = pd.DataFrame({
#         "cluster_idx": validation_data[fold_type],
#         "dmg": validation_data[yvar],
#         "predRF": pred_obj
#     })

#     return pred_test


# Perform random CV
# def train_validate_model(train_idx, test_idx, xvars, yvar,
#                             rf_n = 50, rf_max_depth=None):
                            
#     training_data = data_pxs_sample.iloc[train_idx]
#     validation_data = data_pxs_sample.iloc[test_idx]

#     # Random Forest model
#     rf = RandomForestRegressor( n_estimators = rf_n, 
#                                 n_jobs = -1,
#                                 max_depth = rf_max_depth,
#                                 random_state = 42)
                                
#     rf.fit(training_data[xvars], training_data[yvar])

#     # Prediction
#     pred_obj = rf.predict(validation_data[xvars])
#     pred_test = pd.DataFrame({
#         "cluster_idx": validation_data[fold_type],
#         "dmg": validation_data[yvar],
#         "predRF": pred_obj
#     })

#     return pred_test


def load_config(configFile):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(os.path.join(configFile))
    
    ## path settings
    path2data = config['PATHS']['path2data']
    path2save = config['PATHS']['path2save']        # full path:   e.g. ee-export_s1_relorbs/path/to/dir
    gridTiles_geojson_path = config['PATHS']['gridTiles_shapefile']
    iceshelf_path_meas = config['PATHS']['iceshelves_shapefile']
    roi_path = config['PATHS']['regions_shapefile']

    ## data settings
    dmg_type = config['DATA']['dmg_type']
    if dmg_type not in ['dmg','dmg095','dmg099']:
        raise ValueError('Dmg type can be ''dmg'', ''dmg095'' or ''dmg099'', received {}'.format(dmg_type))
        
    dtresh_dict = {'dmg':'037', 'dmg095':'053' , 'dmg099':'063'}
    dtresh=dtresh_dict[dmg_type]

    ksize = config['DATA']['downsample_size']
    if ksize is None or ksize == 'None':
        ksize = None
        dres='0x0'
    elif int(ksize) ==0 :
        ksize = None
        dres='0x0'
    else: 
        dres=ksize+'x'+ksize
        ksize = int(ksize) #  int(ksize) == 0:

    years_train   = json.loads(config.get("DATA","years_train"))
    years_exclude = json.loads(config.get("DATA","years_exclude_from_test"))
    fold_type = config['DATA']['groupKFolds_by']
    k_split_outer = int(config['DATA']['k_groupSplit_traintest'])
    k_split_inner = int(config['DATA']['k_groupSplit_trainval'])
    nth_fold_rCV = int(config['DATA']['select_nth_testFold'])

    ## runtime settings
    nb_cores =  int(config['RUNTIMEOPTIONS']['cores'])
    train_randomSearch =  True if config['RUNTIMEOPTIONS']['train_randomSearch'] == 'True' else False
    train_gridSearch =  True if config['RUNTIMEOPTIONS']['train_gridSearch'] == 'True' else False
    # train_singleRF = False # True if config['RUNTIMEOPTIONS']['train_singleRF'] == 'True' else False
    if not train_randomSearch and not train_gridSearch:
        print('No training options are set to True; no training will be performed. Set either train_randomSearch or train_gridSearch to True ')
    if train_randomSearch and train_gridSearch:
        raise ValueError('Set either train_randomSearch OR train_gridSearch to true. Not both')

    # print(config._sections.keys)
    if train_randomSearch:
        search_section='RANDOMSEARCH'
        space_section='RANDOMSEARCHSPACE'
        n_rCV = int(config[search_section]['number_of_fits'])

    if train_gridSearch:
        search_section='GRIDSEARCH'
        space_section='GRIDSEARCHSPACE'
        n_rCV = None

    ## config for RandomForest randomSearchCV / gridSearchCV training
    xvar_list = json.loads(config.get(search_section,"xvar_list")) # use json for reading lists
    length_scales = config[search_section]['strain_length_scale_px'].split()
    length_scales = [s.replace(',','').replace(' ','') for s in length_scales]  # remove any unintended remaining separatos
    length_scales = [s + 'px' for s in length_scales]

    scoring_metric = config[search_section]['scoring_metric'].split() # makes list of single/multi metric input.
    scoring_metric = [s.replace(',','').replace(' ','') for s in scoring_metric]  # remove any unintended remaining separatos
    if len(scoring_metric) == 1:
        scoring_metric = scoring_metric[0] # if single metric, randomSearch does not accept lists
    decision_metric= config[search_section]['decision_metric']

    ## search grid space: (randomSearch / gridSearch)
    space = config._sections[space_section] # reads items to dict; but reads items as string isntead of lists
    
    for key in space.keys():
        space[key] = ast.literal_eval(space[key]) # use ast instead of json, as this has no issues with loading "None" or boolean values

    # -- Print some settings for information
    print('Loaded settings:')
    print('  dmg_type:        {}'.format(dmg_type) )
    print('  dtresh:          {}'.format(dtresh) )
    print('  downsample:      {}'.format(ksize) )
    print('  strain l-scale:  {}'.format(length_scales) )
    print('  years to train:  {}'.format(years_train) )    
    print('  N features:      {}'.format(len(xvar_list)) )
    print('  scoring metric:  {}'.format(scoring_metric ) )
    print('  decision metric: {}'.format(decision_metric ) )
    
    print('{} grid'.format(search_section) )
    pprint(space)

    return dmg_type, dtresh, ksize, length_scales, xvar_list, \
                years_train, years_exclude, train_randomSearch, train_gridSearch, \
                scoring_metric, decision_metric, space, \
                fold_type, k_split_outer, k_split_inner, nth_fold_rCV, n_rCV, nb_cores, \
                path2data, path2save, gridTiles_geojson_path, iceshelf_path_meas, roi_path
                

    
def load_nc_sector_years( path2data, sector_ID, year_list=None, varName=None ):
    ''' Load all/selected annual netCDF files of a variable for one sector'''

    filelist_dir =  glob.glob( os.path.join(path2data, f'*_sector-{sector_ID}_*.nc') )
    filelist_var_all = [file for file in filelist_dir if varName in file]
    filelist_var_all.sort()

    if year_list is None: # all years
        ## load list of files
        filenames = [os.path.basename(file) for file in filelist_var_all]
        ## retrieve available years from filenames
        year_list = [int( re.search(r'\d{4}', file).group()) for file in filenames]
        filelist_var = filelist_var_all.copy()

    else: # filter filelist for desired year
        filelist_var=[]
        for year in year_list:
            filelist_yr = [file for file in filelist_var_all if str(year) in os.path.basename(file)]
            print(filelist_yr)
            if not filelist_yr:
                raise ValueError(f'Could not find year {year}')
            filelist_var.append(filelist_yr)

    ## Open dataset(s)

    try: # read all years at once
        region_ds = (xr.open_mfdataset(filelist_var ,
                    combine='nested', concat_dim='time',
                    compat='no_conflicts',
                    # compat='broadcast_equals', #  all values must be equal when variables are broadcast against each other to ensure common dimensions.
                    preprocess=myf.drop_spatial_ref)
            .rio.write_crs(3031,inplace=True)
            .assign_coords(time=year_list) # update year values (y,x,time)
        )
    except ValueError: # read year by year, then concatenate
        region_list = []
        for file in filelist_var:
            yr = int( re.search(r'\d{4}', os.path.basename(file[0])).group()) 
            print(yr)
            with xr.open_mfdataset(file) as ds:
                try:
                    ds.assign_coords(time=yr)
                except: pass
                region_list.append(ds.rio.write_crs(3031,inplace=True))
        region_ds = xr.concat(region_list,dim='time')  
        print(region_ds.coords) 
    return region_ds
        

def main(configFile):

    ''' ---------------------------
    Set paths, and other config
    -------------------------------'''
    if configFile is None:
        raise NameError('No config file specified. Run script as "python this_script.py /path/to/config_file.ini"')
    
    # config = configparser.ConfigParser(allow_no_value=True)
    # config.read(os.path.join(configFile))
    dmg_type, dtresh, ksize, length_scales, xvar_list, \
            years_train, years_exclude, train_randomSearch, train_gridSearch, \
            scoring_metric, decision_metric, space, \
            fold_type, k_split_outer, k_split_inner, nth_fold_rCV, n_rCV, nb_cores, \
            path2data, path2save, gridTiles_geojson_path, iceshelf_path_meas, roi_path = load_config(configFile)


    # os.makedirs(path2savefig, exist_ok=True) 
   


    ''' ----------------------
    Load data: shapefiles 
    ------------------------- '''
    # geojson
    # gridTiles = gpd.read_file(gridTiles_geojson_path)

    # measures ice shelves
    iceshelf_poly_meas = gpd.read_file(iceshelf_path_meas)

    ## regions of interest for AIS
    roi_poly = gpd.read_file(roi_path)

    sector_ID_list = roi_poly['sector_ID'].to_list()
    sector_ID_list.sort()

    ''' ----------------------
    Load data: netCDFs per region, per variable
    ------------------------- '''

    start_t = time.time()

    ## region_ds_list = []
    data_pxs_gdf_list = []
    # for region_ID in region_ID_list: # region_ID_list[:-2]+[region_ID_list[-1]]:
    for sector_ID in sector_ID_list[:1]: 
    # for index, region_ID in enumerate(region_ID_list): # 7:]):

        print('----\n Loading netCDF for region ', sector_ID)


        ''' -------------------
        Load all variables from individual netCDF files 
        Expecting data directory to contain netCDFs per sector per training variable per year. 
        ----------------------- '''
        region_ds_varlist=[]
        for var in ['vx','vy','dmg']: # base variables to read, from which all other training features are calculated
            region_var = load_nc_sector_years( path2data, sector_ID, varName=var) # load all available years
            region_ds_varlist.append(region_var)
        # load rema (only 1 year)
        region_var = load_nc_sector_years( path2data, sector_ID, varName='rema', year_list=['0000']) # , year_list=years_train)
        region_ds_varlist.append(region_var)
        # combine to single dataset
        region_ds = xr.combine_by_coords(region_ds_varlist)
        print('Loaded variables: \n', list(region_ds.keys()) )

        ''' ------------------------------
        #######
        ####### DATA (PRE)PROCESSING
        #######
        ----------------------------------'''
        

        ''' --------------------------------------
        Repeat temporally static variable (REMA) to even out dataset dimension
        This drops time=0
        ------------------------------------------ '''

        region_ds = myf.repeat_static_variable_timeseries( region_ds , 'rema' )


        ''' ----------------
        Downsample observation data ( 400m to 8000m )
        --------------------'''

        if ksize:
            dx = int(region_ds.rio.resolution()[0])
            dy = int(region_ds.rio.resolution()[1])
            if np.abs(dx) != np.abs(dy):
                print("Warning: x and y resolution are not the same; {} and {} -- code update required".format(np.abs(dx), np.abs(dy) ))

            # with dask.config.set(**{'array.slicing.split_large_chunks': True}): # gives error?
            with dask.config.set(**{'array.slicing.split_large_chunks': False}): ## accept large chunks; ignore warning
                region_ds = myf.downsample_dataArray_withoutD0(region_ds, ksize=ksize, 
                                            boundary_method='pad',downsample_func='mean', skipna=False)
            new_res = ksize*400
            print('.. resolution {}m downsampled to {}m'.format(dx, new_res))
        
        ''' ------------
        Calculate strain for (downsampled) data
        ---------------- '''
        
        for scale_name in length_scales: #[1px, 5px, 15px]:
            
            lscale = int(scale_name.strip('px'))

            # calculate
            emax,  emin, e_eff, strain_components  = myf.calc_nominal_strain(region_ds['vx'], region_ds['vy'], 
                                                                             length_scale_px=lscale , 
                                                                             version2 = True, dx=new_res)
            elon,  etrans,  eshear  = strain_components
            
            # make dataset
            region_ds_strain =  [   emax.to_dataset( name='emax_'+str(lscale)+'px') , 
                                    emin.to_dataset( name='emin_'+str(lscale)+'px') , 
                                    e_eff.to_dataset(name='e_eff_'+str(lscale)+'px') , 
                                    elon.to_dataset( name='elon_'+str(lscale)+'px') , 
                                    etrans.to_dataset(name='etrans_'+str(lscale)+'px') , 
                                    eshear.to_dataset(name='eshear_'+str(lscale)+'px') 
            ]
            region_ds_strain = xr.merge(region_ds_strain)
            print('.. calculated strain variables ', list(region_ds_strain.keys()) )

            ## add to dataset
            region_ds = xr.merge([region_ds, region_ds_strain])


            ''' ------------
            Calculate temporal values 
            ---------------- '''

            ## Calculate difference per year
            ## region_ds_diff = region_ds.drop_sel(time=[0])[['emax_'+scale_name, 'v']].diff(dim='time').rename({'emax_'+scale_name:'deltaEmax','v': 'deltaV'})
            region_ds_diff = region_ds[['emax_'+scale_name, 'v']].diff(dim='time').rename({'emax_'+scale_name:'deltaEmax','v': 'deltaV'})
            region_ds_diff # [2016,2017,2018]; first timestep is dropped

            ## Get rolling-max diff of past 3 years. Set center=False so the window is a trailing window i-2 to i
            ## NB: with min_periods=1, the first year will have the same values as itself
            region_ds_roll = region_ds_diff[['deltaEmax','deltaV']].rolling(time=3, center=False, min_periods=1).max().rename(
                                    {'deltaEmax':'dEmax_'+scale_name,
                                    # 'deltaV':'dV_'+scale_name
                                    }) 

            ## Fill the first temporal-difference timestep (2015) with a value (so that this time slice doesnt get dropped later on)
            ## NB: the fill value for dt_2014-15 with a copy of dt_2015-2016, as filling with 0 would create  artificial data
            da_delta_2015_new = region_ds_roll.sel(time=2016).assign_coords(time=2015) # select 2016 from dataset and assing it as time=2015
            region_ds_roll = xr.concat([da_delta_2015_new, region_ds_roll],dim='time') # replace it with the copied value

            ## add to dataset
            region_ds = xr.merge([region_ds, region_ds_roll])

        ''' --------------------------------------
        Drop irrelevant variables from dataset
            aggregate to 1D 
        As conversion of xr.DataSet to pd.DataFrame is quite memory-expensive, dropping a few variables is preferential
        ------------------------------------------ '''
        region_ds = region_ds[[dmg_type] + xvar_list] # select only variables relevant training
        print('.. variables selected for training dataset: \n', list(region_ds.keys()) )

        ''' Clip to iceshelf '''
        iceshelf_polygon_gpd = iceshelf_poly_meas.drop(['testField','TYPE'],axis=1)
        region_ds  = region_ds.rio.clip( iceshelf_polygon_gpd.geometry, iceshelf_polygon_gpd.crs, drop=False, invert=False)

        ''' Aggregate to 1D '''
        region_ds_1d = region_ds.stack(samples=['x','y']) # (time, samples)

        ''' ----------------
        Convert to dataFrame
        --------------------'''
        print('Stacked {} pixels, convert to dataFrame'.format(len(region_ds_1d.samples)))

        data_pxs_df = region_ds_1d.to_dataframe() # nested dataframe
        data_pxs_df = data_pxs_df.rename(columns={"x": "x_coord", "y": "y_coord"}) # rename the column so that flattening the multi-index in the next step does not give an error (relevant for  pandas version > 1.4.3)

        # Flatten the nested multi-index to just column values -- automatically generates a 'year' value for every sample
        data_pxs_df = data_pxs_df.reset_index(level=['time','x','y']) # 18767504 rows; 
        
        # # Drop spatial ref (has not data acutally) 
        data_pxs_df = data_pxs_df.drop(['spatial_ref'],axis=1)
        
        # For spatial k-fold: do not drop x and y || 
        #   in new pandas version (2.0.1) the x and y were already a column, so i can (still) drop them here in the line below
        #   in old pandas versioin (1.4.3) dropping the x and y here will remove the values completely

        # data_pxs_df = data_pxs_df.drop(['x','y'],axis=1)

        ''' ----------------
        Drop NaN pixels:
        Pandas drops all rows that contain missing values. 
        - This means that if any variable has a NaN value, that px is dropped.
          So make sure to fill NaN values for variables before this step iif needed (filling gaps etc) 
        - Since I have rows for px per year, this means that if I would have clipped the data to annual ice shelf polygons, the number of pixels per year can vary.  
        -------------------- '''

        print('.. dropping {} rows with any NaN value'.format( data_pxs_df.isna().sum(axis='index').max() ))
        data_pxs_df.dropna(subset=xvar_list, axis='index',inplace=True) # Drop rows which contain missing values.


        ''' ----------------
        PREP DATA to do RF regression
        --------------------'''
        
        # # remove all 0 values
        print('.. dropping {} rows with dmg=0 (after downsampling)'.format(len(data_pxs_df.loc[data_pxs_df[dmg_type]==0] )) )
        data_pxs_df = data_pxs_df.loc[data_pxs_df[dmg_type]>0]


        ''' ----------------
        Add index of corresponding ice shelf to each point (to be used for spatial k-fold CV)
        1. Convert dataframe to geopandas dataframe
        2. Perform spatialjoin to iceshelf polygons
        Note: Do this step after dropping all NaN and dmg==0 points, as converting to geopandasDF is a bit slow on large dataFrames.
        --------------------'''

        print('.. converting to geopandas dataframe and identifying corresponding iceshelf per px')

        # 1. create geoDataFrame
        gdf_region = gpd.GeoDataFrame( data_pxs_df, geometry=gpd.points_from_xy( data_pxs_df.x, data_pxs_df.y ), crs='EPSG:3031')

        # 2. Identify to which ice shelf the points belong to, using geopandas spatial-join
        # 'index_right' is the row-idx of the iceshelf dataframe. Rename to 'iceshelf_index' because thats what i was looking for. 
        # 'how=left' means that all values of the left dataset are kept and matched to the right one. All entries on the right-dataset that have no match, are discarded
        gdf_region = gdf_region.sjoin(iceshelf_polygon_gpd,how='left') 
        gdf_region = gdf_region.rename(columns={'index_right':'iceshelf_index'}) #.drop(['x','y','Area'],axis=1)

        print('.. {} px in region-dataframe'.format(len(gdf_region)))


        ''' ----------------
        Put df (back) in list
        --------------------'''
        
        data_pxs_gdf_list.append(gdf_region)


    ''' ------------------------------
    Stack GDFs to get 1 big dataset 
    ----------------------------------'''

    data_df = pd.concat(data_pxs_gdf_list) # dataset with px samples of all ice shelves

    # Report time for loading
    end_t = time.time()
    total_time = end_t - start_t
    print('---\n Loaded data: {:.3f} min'.format(total_time/60))


    ''' ------------------------------

    #######
    ####### RF Training
    #######

    RandomForest:  SPATIAL K-fold CV
    Use spatial CV in Random-search-CV to find estimate of grid parameters
    - Define ranges of hyperparams to test
    - Fit hyperparams and validate using pre-defined (spatial) groups
    - Can do either group-kFold or leave-one-group-out (LOGO) (leave-one-iceshelf-out)
      I choose to do group-kFold since a single ice shelf could be only 10px which would be too small compared to the rest of the dataset
    - Do this for a single train/test split
    ----------------------------------'''


    ''' ------------------------------
    Final data preparations
    ----------------------------------'''


    ## SAMPLING

    ## prinit some info
    Nsamples = len(data_df)
    print("{} total samples ".format(Nsamples))
    
    data_pxs_sample = data_df.sample(int(Nsamples), random_state=42 )

    # again, make sure there are no NaN values (before youo do that, remove unnecessary columns that might contain NaN values)
    try:
        data_pxs_sample = data_pxs_sample.drop(['NAME','Regions','Area'],axis=1)
    except:
        pass

    data_pxs_sample.dropna(axis='index',inplace=True)


    print('Training RF on variables: \n', xvar_list)

    spatial_groups = data_pxs_sample[fold_type] # array for all samples providing which group (a #) it belongs to
    spatial_folds = spatial_groups.unique()
    n_folds = len(spatial_folds) 
    print('Number of groups in data: ',n_folds)

    # configure the cross-validation procedure
    # Initialise the model
    model = RandomForestRegressor(random_state=11)


    ''' ------------------------------
    Configure the (nested) cross-validation procedure
    ----------------------------------'''
    ## For ALL AIS, all ice shelves = 130  that have data ( approx )
    # - groupKFold should have n_split <= n_spatial_groups
    # - since I'm doing nested CV, the inner-cv must have this as well.
    # - also, try to get like a 80/10/10 datasplit?

    cv_inner = GroupKFold(n_splits=k_split_inner)  # the number of distinct groups has to be >= number of folds (n_splits)
    cv_outer = GroupKFold(n_splits=k_split_outer)  # The order the groups are used is always the same (also: doesnt matter)


    ''' ------------------------------------------------------------
    Split test-dataset from data
    - Spatial test-set selection:
        - Split dataset in K-spatial folds (outer fold)
        - Select a single one of the folds as the train/test split
    - Temporal test-set selection:
        - Use years= 2015,2016,2017, 2018 for training 
        - Remove 2019-2020 from testset
        - Test-set then has (a) 2015-2018 for the spatial-testset and (b) 2021 for temporal (all iceshelves)
    - Continue to do hyperparam tuning on the training data, 
        and split that further into train/validation sets (inner folds)
    ----------------------------------------------------------------'''

    ''' -------------
    ## Cut outer fold: select one as train/test data split
    -----------------'''

    ## SPATIAL train/test split
    cv_outer_folds = [fold for fold in cv_outer.split(data_pxs_sample,groups=spatial_groups)] # list with a tuple for each fold with (train_idx, test_idx)

    # select nth outer_fold
    train_idx, test_idx = cv_outer_folds[nth_fold_rCV]
    train_idx #  --> use this for randomSearchCV to split in train/val set

    # For selected fold, split train/test                                
    fold1_traindata = data_pxs_sample.iloc[train_idx]
    fold1_testdata = data_pxs_sample.iloc[test_idx]
    fold1_spatialgroups = spatial_groups.iloc[train_idx]
    fold1_spatialgroups_test = spatial_groups.iloc[test_idx]

    fold_iceshelves_test = data_pxs_sample.iloc[test_idx][[dmg_type,'geometry']].sjoin(iceshelf_poly_meas)['NAME'].unique()
    print('.. Split outer-fold to set test-data apart. Selected fold {}, yielding ice shelves for testdata:\n {} '.format(nth_fold_rCV,fold_iceshelves_test))

    ''' -------------
    ## Further remove data (years) from testset
    -----------------'''

    ## TEMPORAL train/test split:
    print('.. Select years for training set:' , years_train )

    fold1_spatialgroups = fold1_spatialgroups.loc[fold1_traindata['time'].isin(years_train)]
    fold1_traindata = fold1_traindata.loc[fold1_traindata['time'].isin(years_train)]

    fold1_spatialgroups_test = fold1_spatialgroups_test.loc[~fold1_testdata['time'].isin(years_exclude)]
    fold1_testdata  = fold1_testdata.loc[~fold1_testdata['time'].isin(years_exclude)]
    Nsamples_fold = len(fold1_traindata) + len(fold1_testdata)

    
    ## print info about outer folds:

    print('CV outer split={} fold {} -- training/testing {:.0f}/{:.0f}% of samples {}/{}; distinct groups: {}/{}'.format(
                            k_split_outer, nth_fold_rCV, 
                            len(fold1_traindata)/Nsamples_fold*100,  len(fold1_testdata)/Nsamples_fold*100, 
                            len(fold1_traindata),len(fold1_testdata),
                            len(fold1_spatialgroups.unique()), len(fold1_spatialgroups_test.unique())
                            ))

    

    ''' ------------------------------
    Adjust sample imbalance:
    Apply to selected traindata that will be used in the hyperparam tuning
    
    Following these steps:
    (a) discretize damage values into 10 bins
    (b) undersample majority class (which has ~80% of data)
    (c) oversample minority classes
    --  NB: do this AFTER selecting train/test data and Oversampling ONLY on training data
        because otherwise in the oversampling process, test-data is leaked into training set !! Undersampling could be done before
    ----------------------------------'''

    ## Discretize dmg values to identify majority and minority classes
    # print('.. adjusting sample imbalance ')
    # df_discr = fold1_traindata.copy()
    # bins    = np.linspace(0,df_discr[dmg_type].max(), num=11, endpoint=True) # discretize into 10 classes
    # binlabel=['bin-'+str(i) for i in range(0,10) ] 
    # df_discr['dmg_binned'] , cut_bin1 = pd.cut(x = df_discr[dmg_type], bins = bins, labels=binlabel,
    #                                 include_lowest = True, retbins=True, right=True)
    
    # '''FIRST: undersampling of extreme majority class '''
    # bin_counts_input = df_discr['dmg_binned'].value_counts().values
    # n_largest_class = np.max(bin_counts_input)
    # bin_nums_under = {'bin-0':np.min([ bin_counts_input[0], int( n_largest_class*0.7) ]),  # keep x% of largest class 
    #                 'bin-1': bin_counts_input[1],  
    #                 'bin-2': bin_counts_input[2],  
    #                 'bin-3': bin_counts_input[3],  
    #                 'bin-4': bin_counts_input[4],  
    #                 'bin-5': bin_counts_input[5],  
    #                 'bin-6': bin_counts_input[6],  
    #                 'bin-7': bin_counts_input[7],  
    #                 'bin-8': bin_counts_input[8],  
    #                 'bin-9': bin_counts_input[9],  
    #                 }

    # undersample = RandomUnderSampler(sampling_strategy=bin_nums_under)
    # X, y = df_discr.drop(['geometry'],axis=1), df_discr[['dmg_binned']].values.flatten()
    # X_under, y_under = undersample.fit_resample(X, y)

    # ''' ## SECOND: oversampling of minority classes'''

    # bin_counts_input = X_under['dmg_binned'].value_counts().values
    # n_largest_class = np.max(bin_counts_input)

    # bin_nums_over = {'bin-0':np.max([ bin_counts_input[0], int( n_largest_class*1) ]),  # largest class 
    #                 'bin-1': np.max([ bin_counts_input[1], int( n_largest_class*0.8) ]),  # 
    #                 'bin-2': np.max([ bin_counts_input[2], int( n_largest_class*0.7) ]),  # 
    #                 'bin-3': np.max([ bin_counts_input[3], int( n_largest_class*0.6) ]),  # to 20%
    #                 'bin-4': np.max([ bin_counts_input[4], int( n_largest_class*0.5) ]), #  
    #                 'bin-5': np.max([ bin_counts_input[5], int( n_largest_class*0.4) ]), # 
    #                 'bin-6': np.max([ bin_counts_input[6], int( n_largest_class*0.3) ]), # 
    #                 'bin-7': np.max([ bin_counts_input[7], int( n_largest_class*0.2) ]), # | to 10%
    #                 'bin-8': np.max([ bin_counts_input[8], int( n_largest_class*0.1) ]), #  
    #                 'bin-9': np.max([ bin_counts_input[9], int( n_largest_class*0.1) ]), # 
    #                 }   
    # oversample = RandomOverSampler(sampling_strategy=bin_nums_over) # ratio of minority class to majority (only for binary)
    # X_under_over, y_under_over = oversample.fit_resample(X_under, y_under)

    # print('Total samples input                  : ', len(df_discr) ) 
    # print('Total samples under-and-oversampling : ', len(X_under_over) ) 

    # # use oversampled traindata
    # fold1_traindata = X_under_over 
    # fold1_spatialgroups = X_under_over[fold_type] 
    # print('---\n NEW CV-INNER INFO WITH OVERSAMPLED DATA')


    ''' --------------
    ## Inner fold info:
    ------------------ '''

    # info about inner folds:                        
    cv_inner_folds = [fold for fold in cv_inner.split(fold1_traindata, groups=fold1_traindata[fold_type])] # fold1_spatialgroups)] # list with a tuple for each fold with (train_idx, val_idx)
    for fold_num, current_fold in enumerate( cv_inner_folds ):
        train_idx, validation_idx = current_fold
        groups_in_fold_train = fold1_spatialgroups.iloc[train_idx].unique()
        groups_in_fold_test  = fold1_spatialgroups.iloc[validation_idx].unique()
        N_set = len(train_idx) + len(validation_idx)
        print('CV inner split={} fold {} -- training/validation {:.0f}/{:.0f}% of samples {}/{}; distinct groups: {}/{}'.format(
                            k_split_inner, fold_num, 
                            len(train_idx)/N_set* 100 ,len(validation_idx)/N_set*100, 
                            len(train_idx),len(validation_idx),
                            len(groups_in_fold_train), len(groups_in_fold_test)
                            ))

    


    ''' --------------
    ## Start training
    ------------------ '''


    if train_randomSearch:
        ''' ----------------------------------------------------------
        #### RANDOM SEARCH 
        # - for parameter selection
        # - to narrow down hyperparameter searchgrid 
        --------------------------------------------------------------'''
        start_t = time.time()

        # see how many combi's there are
        n_candidates=1 # initialise count
        for key in space.keys():
            n_key = len(space[key])
            n_candidates*=n_key
        
        print('--\n Perform RandomizedSearchCV for {} out of {} candidates:'.format(n_rCV, n_candidates))
        
        ## Search space: defined in config file
        pprint(space)

        rf_Search = RandomizedSearchCV(    estimator = model, 
                                        param_distributions = space, 
                                        scoring=scoring_metric, 
                                        n_iter = n_rCV, #100
                                        cv = cv_inner, 
                                        # refit=True,
                                        refit=decision_metric,          # if you use multi-scoring, the best_estimator still is decided on one of the metrics; defined here.
                                        return_train_score=True, 
                                        verbose=3, random_state=42, 
                                        n_jobs=int(nb_cores/2) ,                            # number of jobs to start parallel. -1 means using all processors. 
                                        pre_dispatch=1,#,nb_cores-2,                        # control the number of jobs that get dispatched during parallel execution (avoid explosion of memory consumption); default 2*n_jobs, so make sure n_jobs is not too big
                                        )  # NB: Warning You cannot nest objects with parallel computing (n_jobs different than 1). 
                                    

        ## Do the randomSearchCV by fitting the randomSearch 'model' ON A SINGLE FOLD (so that i have testedata)
        # rf_randomSearch.fit( data_pxs_sample[xvar_list], data_pxs_sample[dmg_type], groups=spatial_groups)
        rf_Search.fit( fold1_traindata[xvar_list], 
                                fold1_traindata[dmg_type], 
                                groups=fold1_traindata[fold_type])

    if train_gridSearch:
        ''' ----------------------------------------------------------
        #### GRID SEARCH 
        # - to do final hyperparemetr tuning
        # - get final estimator as result immediately (by refit=True)
        --------------------------------------------------------------'''
        start_t = time.time()
        print('--\n Perform GridSearchCV..')

        rf_Search = GridSearchCV(    estimator = model, 
                                        param_grid = space, 
                                        scoring=scoring_metric, 
                                        cv = cv_inner, 
                                        refit=decision_metric,          # if you use multi-scoring, the best_estimator still is decided on one of the metrics; defined here.
                                        return_train_score=True, 
                                        verbose=3, 
                                        n_jobs=int(nb_cores/2) ,                            # number of jobs to start parallel. -1 means using all processors. 
                                        pre_dispatch=2,#,nb_cores-2,                        # control the number of jobs that get dispatched during parallel execution (avoid explosion of memory consumption); default 2*n_jobs, so make sure n_jobs is not too big
                                        )  # NB: Warning You cannot nest objects with parallel computing (n_jobs different than 1). 

        ## Do the gridSearchCV by fitting the model 
        rf_Search.fit( fold1_traindata[xvar_list], fold1_traindata[dmg_type], groups=fold1_traindata[fold_type]) #fold1_spatialgroups)
        time.sleep(5) # wait 5 sec before continuing, to finish printing CV verbose output
        
    # Report time
    end_t = time.time()
    total_time = end_t - start_t
    print('.. finished after {:.3f} min'.format(total_time/60))

    # Get datetime for filenames
    ct = datetime.datetime.now()  # ct stores current time
    ts = ct.isoformat(sep='T', timespec='minutes') # ct.timestamp()  # ts store timestamp of current time

    ## Print best parameters:
    print('-- Best parameter, with best score {:.5f} ({})'.format(rf_Search.best_score_,decision_metric))
    rf_Search_best_params = rf_Search.best_params_
    pprint(rf_Search_best_params)

    ''' ## --Save '''
    ## gridSearch defines best parameters based on training+validation data
    # Can then apply directly using 'test' data / save the RF it to load it later
    rf_Search_best_estimator = rf_Search.best_estimator_

    ## Save esitmator
    efname = 'RF_bestEstimator_{}'.format(ts)
    print('Saving randomSearch bestEstimator to {}'.format(efname) )
    try:
        dump(rf_Search_best_estimator, os.path.join(path2save,efname + ".joblib"))
    except OSError:
        print('.. OSError for saving joblib; pass')
        pass

    print('Done \n ------')



if __name__ == '__main__':
    #  Run script as "python path/to/script.py /path/to/config_file.ini"
        
    # retrieve config filename from command line
    config = sys.argv[1] if len(sys.argv) > 1 else None

    # run script
    main(config)   





