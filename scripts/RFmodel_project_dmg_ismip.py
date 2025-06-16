# Script to apply trained RF on ISMIP data to predict damage
import os
import numpy as np
import xarray as xr
import joblib

# Import user functions
import myFunctions as myf 

import time
import configparser
import glob
import cftime as cft


import argparse


def parse_cla():
    """
    Command line argument parser

    Accepted arguments:
        Required:
        --config (-c)     : Specify file with detailed configuration settings to be used in the script.
                            The settings in the config file are the same for every ISMIP model/experiment
        --model (-m)      : Define which ISMIP model to use. Only models with high resolution (8km) are included for this script
                            Choices are: AWI_PISM, DOE_MALI, ILTS_PIK_SICOPOLIS, JPL1_ISSM, NCAR_CISM
        --experiment (-e) : Define which ISMIP forcing experiment to use
        
        Optional:
        ..       : ..

    :return args: ArgumentParser return object containing command line arguments
    """

    # Required arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",'-c',help="Specify path to config file", type=str, required=True)
    parser.add_argument("--model","-m",help='Specify which ISMIP model to use',type=str, required=True,
                            choices=('AWI_PISM','DOE_MALI','ILTS_PIK_SICOPOLIS','JPL1_ISSM','NCAR_CISM'))
    parser.add_argument("--experiment", "-e", help="Specify which ISMIP experiment to process", required=True,
                            choices=('ctrl_proj_std', 'exp05','exp06','exp07','exp08','all'))                        

    # Optional arguments
    # parser.add_argument("--optional",'-p',help="Optional argument", type=str, required=False)

    args = parser.parse_args()
    return args 

# def load_ismip_experiment(path2ismip, exp_num, length_scales):
def load_ismip_experiment(filelist, length_scales,velocity_varnames=('xvelsurf','yvelsurf') ):
    var_name_vx, var_name_vy = velocity_varnames

    ''' ---------------------------
    Load data
    -------------------------------'''

    ## read all files at once (with deocde_times=True)
    with xr.open_mfdataset(filelist,  
                            combine="by_coords",
                            # decode_times=True, # gives you nice time values, but is not good saving format
                            decode_times=False, # saveable format for netcdf4
                            data_vars='minimal', coords='minimal', 
                            compat='broadcast_equals') as data_ismip_ds:                 

        ## Get number of days from calendar
        # NB: The proleptic_gregorian calander (used by NCAR CISM) includes leap days. 
        # The nubmer of days is used to convert velocity m/s to m/yr, so here it's assumed that using 365 days for all years is an appropriate simplification
        # model_calendar = data_ismip_ds['time'].attrs['calendar'] # 365_day
        try:
            model_calendar = data_ismip_ds.time.attrs['calendar'] 
        except KeyError:
            print('..Did not find property ''calendar'' in attributes. Assume 365-day calendar')
            # print('..Available attributes: ', data_ismip_ds.time.attrs)
            model_calendar = '365_day_assumed'
        if model_calendar == 'proleptic_gregorian':
            print('..Data uses proleptic gregorian calander, which includues leap days. Simplify to 365 days for velocity unit conversion.')
            model_calendar = '365_day_as_simplified_proleptic_gregorian'

        ndays = int(model_calendar.split('_day')[0]) # 365


        ## Drop irrelevant variables (e.g. 'time_bnds' in AWI_PISM and 'lon_bnds','lat_bnds' in NCAR_CISM) 
        # data_ismip_ds = data_ismip_ds.drop('time_bnds')
        data_ismip_ds = data_ismip_ds[[var_name_vx,var_name_vy,'orog']]

        ## For JPL1 and NCAR, fix grid issues:
        if 'JPL1' in filelist[0] or 'NCAR' in filelist[0]:
            print('JPL1/NCAR: update with coordinates from AWI_PISM to grid')
            
            ## Infer path to AWI_PISM model (use arbitrary variable/experiment number for this)
            path_parts = filelist[0].split('/') # AWI_PISM is first of alphabetic filelist
            idx_ismip = path_parts.index('ISMIP6')
            tmp_path = '/'.join(path_parts[:idx_ismip]+['ISMIP6','AWI_PISM'])
            file_tmp = glob.glob(os.path.join(tmp_path,'orog_AIS_AWI_PISM1_exp05.nc') )[0]
            with xr.open_dataset(file_tmp,  decode_times=False) as ds_tmp:
                ## fill awi grid coordinates
                data_ismip_ds = data_ismip_ds.assign_coords(y=ds_tmp['y']) # for JPL1
                data_ismip_ds = data_ismip_ds.assign_coords(x=ds_tmp['x']) # for JPL1
                print('.. Coords after fix: ',list(data_ismip_ds.coords) )
                print('.. Resolution after fix: ',data_ismip_ds.rio.resolution()) 
            
        if 'JPL1' in filelist[0]:
            # Also fix issue where xvelmean has a typo in its dimension; consisting of (ny,x) instead of (y,x) 
            data_ismip_ds["xvelmean"] = data_ismip_ds["xvelmean"].rename({"ny": "y"})
            # And issue where expirment ctrl_proj has no correct attritbutes (copy attrs of exp05)
            tmp_path = '/'.join(path_parts[:idx_ismip]+['ISMIP6','JPL1_ISSM'])
            file_tmp = glob.glob(os.path.join(tmp_path,'orog_AIS_JPL1_ISSM_exp05.nc') )[0]
            with xr.open_dataset(file_tmp,  decode_times=False) as ds_tmp:
                data_ismip_ds.attrs = ds_tmp.attrs

        if 'NCAR' in filelist[0] and 'exp07' in filelist[0]:
            # Error in time array of NCAR_CISM exp07 : copypaste the time array of exp05
            tmp_path = '/'.join(path_parts[:idx_ismip]+['ISMIP6','NCAR_CISM'])
            file_tmp = glob.glob(os.path.join(tmp_path,'orog_AIS_NCAR_CISM_exp05.nc') )[0]
            with xr.open_dataset(file_tmp,  decode_times=False) as ds_tmp:
                data_ismip_ds['time'] = ds_tmp['time']
                data_ismip_ds.attrs = ds_tmp.attrs
        

        ''' --------------
        Convert variables to correct units
        ISMIP data is in m/s; convert to m/yr (RF is trained on velocity from ITS_LIVE, which is m/yr)
        ------------------ '''
        vx_attrs = data_ismip_ds[var_name_vx].attrs
        vx_attrs['units']='m/yr'
        vy_attrs = data_ismip_ds[var_name_vy].attrs
        vy_attrs['units']='m/yr'
        data_ismip_ds[var_name_vx] = (data_ismip_ds[var_name_vx]*60 *60 *24 * ndays).assign_attrs(vx_attrs) # m/s to m/yr
        data_ismip_ds[var_name_vy] = (data_ismip_ds[var_name_vy]*60 *60 *24 * ndays).assign_attrs(vy_attrs) # m/s to m/yr


        ''' --------------
        Calculate features
        ------------------ '''
        
        # calculate velocity, strain components and temporal velo/strain change
        data_velo_strain, region_ds_roll = myf.calculate_velo_strain_features(data_ismip_ds, 
                                                    velocity_names=(var_name_vx,var_name_vy), 
                                                    length_scales=length_scales)
        data_ismip_ds = xr.merge([data_ismip_ds, data_velo_strain])
        data_ismip_ds = xr.merge([data_ismip_ds, region_ds_roll])

        ## set crs
        data_ismip_ds.rio.write_crs(3031,inplace=True)

        return data_ismip_ds

def predict_dmg(loaded_rf, data_ismip_yr, feature_list_ismip, select_years=None):

    ''' --------------
    Select few years of data / all data
    ------------------ '''
    if select_years is not None:
        # select list of years
        data_ismip_yr = data_ismip_yr.sel(time=[cft.DatetimeNoLeap(yr,1,1) for yr in select_years])

    '''## Get mask -- if based on dataSet, every variable will get its own mask.
    - (1) mask all variables separately to binary 1 for nan-values, 0 for non-nan
    - (2) sum the masks and get a single binary nan-maks for the whole dataset'''

    nan_mask = xr.where( np.isnan(data_ismip_yr) , 1, 0 ).to_array().sum("variable") # condition, values_if_true, values_if_false;  # stacks all variablesas new dimension, and then sums)
    # nan_mask = xr.where( nan_mask > 0 , 1, 0 ) # make binary -- also flattens (time,y,x) to (y,x)

    '''
    ## Fill nan
    - RF cannot handle NaN data
    - want to be able to mask NaN data from predictions
    - for this, we use land and ocean mask provided by ISMIP data
    '''

    data_ismip_yr = data_ismip_yr.fillna(-999)


    ''' --------------
    Make predictions for selected year:
    - Stack x,y data to 1D dataArray (samples, ) per feature
    - Coombine multiple features into a single dataArray (samples, featurues)
    - Usa a dummy dataArray to plug predicted values in (enabling conversion back to 2D map)
    ------------------ '''
    ## stack 2D/3D data as 1D samples
    data_stack_ds = data_ismip_yr.stack(samples=['y','x','time']) 

    for feat in feature_list_ismip:
        if feat not in list(data_stack_ds.keys()):
            raise Exception('Feature {} is not in dataset: {}'.format(feat, list(data_stack_ds.keys()) ) )


    ## convert to dataArrray for input (X) and target (y) features
    y_stack_da_dummy = xr.zeros_like(data_stack_ds['orog']) # dataArray structure for predicted values
    X_stack_da = data_stack_ds[feature_list_ismip].to_array('feature').transpose('samples','feature') # Make a single dataArray from multiple variables using .to_array(). The stacked dimension will be 'feature'.
    X_stack_da # (samples, N features)


    '''# Predict on stacked xr.DataArray'''

    y_rf_pred = loaded_rf.predict(  X_stack_da ) # is an array ipv dataArray
    y_pred_da = y_stack_da_dummy.copy(data=y_rf_pred).rename('y_predict').astype('float').assign_attrs(standard_name='predicted_dmg',units='') # put back as dataArray

    ## Unstack 
    y_pred_ismip_da = y_pred_da.unstack().transpose('time','y','x')

    ## apply nan-mask back to array (it could vary from year-to-year depending if ISMIP model alows ice front changes)
    ismip_dmg_da =  y_pred_ismip_da.where(nan_mask==0, other=np.nan
                        ).rio.write_nodata(np.nan, encoded=True,inplace=True)  # properly encode nodata, before saving netcdf

    

    return ismip_dmg_da

def main(configFile, ismip_model, exp_num): 

    ''' ---------------------------
    Set paths, and other config
    -------------------------------'''
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(os.path.join(configFile))

    ## path settings
    path2ismip_base = config['PATHS']['path2ismip_dir']
    path2ismip = os.path.join(path2ismip_base,ismip_model)
    path2save = os.path.join(path2ismip,'damage/')
    path2predictor = config['PATHS']['path2model']   

    ## Predictor settings
    # model_name = 'RF_bestEstimator_randomSearch.joblib'
    # configFile = 'RF_gridSearch.ini'
    predictmodel_file = config['PREDICTIONMODEL']['model_file']
    predictmodel_config = config['PREDICTIONMODEL']['model_config']
    # model_name = 'RF_dmg_predictor_20x20'
    predictor_name = predictmodel_file.split('_bestEstimator_')[0] 

    if 'gridSearch' in predictmodel_file:
        search_section='GRIDSEARCH'
    elif 'randomSearch' in predictmodel_file:
        search_section='RANDOMSEARCH'
    
    ## Read some detailed model_setting from the provided RFmodel-config file
    config_rf = configparser.ConfigParser(allow_no_value=True)
    config_rf.read(os.path.join(path2predictor, predictmodel_config))
    length_scales = config_rf[search_section]['strain_length_scale_px'].split()
    length_scales = [s.replace(',','').replace(' ','') for s in length_scales]  # remove any unintended remaining separatos
    length_scales = [s + 'px' for s in length_scales]

    ## Load RF model
    loaded_rf = joblib.load(os.path.join(path2predictor,predictmodel_file))
    feature_list = list(loaded_rf.feature_names_in_)

    ## fix feature names 
    # - replace 'rema' with 'orog' to match variable name in ismip data
    # - this also makes sure that the feature names are in the same order as the RF expects
    feature_list_ismip = [feature.replace('rema','orog') for feature in feature_list]

    # # -- Print some settings for information
    print('--- \nLoaded settings:')
    print('  ISMIP model :   {}'.format(ismip_model) )
    print('  experiment  :   {}'.format(exp_num) )
    print('  RFpredictor :   {}'.format(predictor_name) )

    if not os.path.isdir(path2save):
        print('.. Creating damage savepath directory')
        os.makedirs(path2save, exist_ok=False)

    if exp_num == 'all':
        experiments = ['ctrl_proj_std','exp05', 'exp06', 'exp07', 'exp08' ]
    else:
        experiments = [exp_num]


    ''' ---------------------------
    Start loading data and making projections using the RF model
    -------------------------------'''
    for exp_num in experiments:
        start_t = time.time()
        ''' ---------------------------
        Load data
        -------------------------------'''

        # print(f'--- \nLoading model data {exp_num}')
        print('--- \nLoading model data {}'.format(exp_num))

        ## Get filelist of ISMIP model-experiment
        filelist = glob.glob(os.path.join(path2ismip,'*'+exp_num+'*.nc') )
        if not filelist: # empty
            if exp_num == 'ctrl_proj_std': # for JPL1_ISSM this is ctrl_proj
                filelist = glob.glob(os.path.join(path2ismip,'*ctrl_proj.nc') )

        ## Check all required variables are available
        var_name_vx = myf.check_presence_of_ismip_variable_file('xvelsurf', filelist)
        var_name_vy = myf.check_presence_of_ismip_variable_file('yvelsurf', filelist)
        _ = myf.check_presence_of_ismip_variable_file('orog', filelist)

        ## Construct save-filename
        fname_dmg_nc = [os.path.basename(file) for file in filelist \
                        if 'orog' in file][0].replace('orog','damage') # e.g. 'damage_AIS_AWI_PISM1_exp05.nc'

        ## Chekc if predicted-dmg file already exists
        if os.path.isfile( os.path.join(path2save,fname_dmg_nc ) ):
            with xr.open_dataset(os.path.join(path2save,fname_dmg_nc )) as ds:
                used_predictor = ds.attrs['prediction_model']
                
                if used_predictor == predictor_name:
                    print('Prediction file {} already exists  -- continue'.format(fname_dmg_nc))
                    continue
                else:
                    raise ValueError('Prediction file {} already exists but predicitons were made with {}' \
                                        ' and not with {} -- move existing file somewhere else.'.format(fname_dmg_nc,used_predictor,predictor_name))
            
        ## load ISMIP data
        data_ismip_ds = load_ismip_experiment(
                            filelist, 
                            length_scales, 
                            velocity_varnames=(var_name_vx, var_name_vy) )

        
        ''' ---------------------------
        Make predictions
        -------------------------------'''
        print('.. Predicting dmg')
        
        ismip_dmg_da = predict_dmg(loaded_rf, data_ismip_ds, feature_list_ismip, select_years=None)
        
        print('.. Done with predicting dmg:', ismip_dmg_da.dims, ismip_dmg_da.shape)

        ''' --------------
        Save the dataset with all predicted dmg years 
        ------------------ '''

        ## add attributes:

        attrs_dict_dmg = {
            'title': 'Predicted damage on ISMIP6 Projections',
            'institution': data_ismip_ds.attrs['institution'],
            'source': data_ismip_ds.attrs['source'],
            'model_abbrev': ismip_model,
            'prediction_model': predictor_name,
            'prediction_features': feature_list_ismip,
        }
        ismip_dmg_da.attrs = attrs_dict_dmg

        ## saving dataArray: Only xarray.Dataset objects can be written to netCDF files, so the xarray.DataArray is converted to a xarray.Dataset object
        
        if not exp_num.startswith('exp'):
            exp_num = 'ctrl'
        var_name = 'D_' + exp_num # name variables D_exp05 etc (better to merge datasets later)
        ismip_dmg_ds = ismip_dmg_da.to_dataset(name=var_name,promote_attrs=True) 
        

        ## Save
        
        print('.. Saving {}'.format(fname_dmg_nc))
        ismip_dmg_ds.to_netcdf( os.path.join(path2save,fname_dmg_nc ), 
                                    mode='w', compute=True, engine='netcdf4')

        # Grab Currrent Time After Running the Code
        end_t = time.time()
        total_time = end_t - start_t
        print('{:.3f} min'.format(total_time/60))



if __name__ == '__main__':

    ''' ---------------------------------------------------------------------------
    Command Line configuration
    Run script as "python path/to/script.py --config path/to/config.ini --model ISMIP_MOD --experiment exp05
    -------------------------------------------------------------- '''
    
    # if optional arguments are not specified in command line, they are set to None
    # required arguments will throw a usage error
    args = parse_cla()
    config = args.config
    ismip_model = args.model
    exp_num = args.experiment
    
    # call main function
    main(config, ismip_model, exp_num)   
