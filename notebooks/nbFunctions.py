# import rioxarray as rioxr
import xarray as xr
import glob
import numpy as np
import os
import geopandas as gpd 
import rasterio as rio
# import warnings
import re 
import dask

# python file with useful functions to import

def get_tilelist_region(region_polys, sector_ID, gridTiles=None):
    ''' This function selects a predefined sector by name ID from the sector-shapefile 'sector_polys'.
    It returns a list of tileNumbers that intersects that region.
    These tileNumbers correspond to the number tile_N in local data filenames
    '''
    if gridTiles is None:  
        try:
            gridTiles_geojson_path = os.path.join('./files/gridTiles_iceShelves_EPSG3031.geojson')
            print('No gridTiles file specified, reading shapefile to geopandas.Dataframe from: \n {}'.format(gridTiles_geojson_path))
            gridTiles = gpd.read_file(gridTiles_geojson_path)
        except:
            raise ValueError('Couldnt find gridTiles shapefile; provide dataframe')

    # -- select polygon for specified sector
    region_df = region_polys.loc[region_polys['sector_ID']==sector_ID]

    # -- Overlap polygon with tiles, to know which files to dwonload
    intersection_gpd = gpd.sjoin(gridTiles, region_df )
    tileNums_select = intersection_gpd['tileNumber'].values
    tileNums_select = [int(tileNum) for tileNum in tileNums_select]
    return tileNums_select


def clip_da_to_iceshelf(data_da, iceshelf_polygon_gpd,drop=False):
     ''' Clip data (xarray.DataArray) to ice shelf polygon file (geopandas.DataFrame) '''
     return data_da.rio.clip( iceshelf_polygon_gpd.geometry, iceshelf_polygon_gpd.crs, drop=drop, invert=False)



def calc_nominal_strain(vx, vy, length_scale_px=1 ,version2 = None , dx=None ):
    '''Calculate nominal strain rate based on x and y velocity grids (data tiles per year)
    
    Input
    -----
    Should be xr.DataArray.
    Version2     : implemented the calculation of effecitve strain and individual strain components. For backwards compability, need to specify this. 
    length_scale : Integer, it represnts the number of pixels to take as length scale. 

    Output
    ------
    emax, emin
    theta_p in degrees'''


    # Infer spatial resolution of grid
    if dx is None:
        # dx = np.unique( vx['x'].diff( dim='x' ))[0]
        # dy = np.unique( vx['y'].diff( dim='y' ))[0]
        # if dx.size > 1:
        #     raise ValueError("Non-uniform y grid")
        # if dy.size > 1:
        #     raise ValueError("Non-uniform y grid")
        dx = int(vx.rio.resolution()[0])
        dy = int(vx.rio.resolution()[1])
        if np.abs(dx) != np.abs(dy):
            raise ValueError("x and y resolution are not the same; {} and {} -- code update required".format(np.abs(dx), np.abs(dy) ))

    ##  Length scale settings 
    px  = length_scale_px   # number of pixels to shift
    res = np.abs(dx)*px     # spatial resolution of gradient in grid


    ## Calculate gradients
    # Currently: gradient with current pixel as cornerpoint
    # To do: gradient with current pixel as centerpoint
    dudx = (vx-vx.roll(x=px) )/res
    dvdy = (vy-vy.roll(y=px) )/res
    dudy = (vx-vx.roll(y=px) )/res
    dvdx = (vy-vy.roll(x=px) )/res
    exx = 0.5*(dudx+dudx)
    eyy = 0.5*(dvdy+dvdy)
    exy = 0.5*(dudy+dvdx)

    ## Calculate principal strains
    emax_xr = (exx+eyy)*0.5 + np.sqrt(np.power(exx-eyy,2)*0.25+np.power(exy,2)) # (x,y,time)
    emax = emax_xr.squeeze() # (x,y)
    emin_xr = (exx+eyy)*0.5 - np.sqrt(np.power(exx-eyy,2)*0.25+np.power(exy,2)) 
    emin = emin_xr.squeeze() 

    ## Calculate effecitve strain rate
    # For shallow ice approximation, vertical shear is neglected. Refer to Emetc et al. 2018
    e_eff = np.sqrt( exx**2 + eyy**2 + exy**2 + exx*eyy  )

    ## Calculate longitudonal, transverse and shear strain: rotate to align with velocity field
    # Function from Alley at al. 2018, who refer to Bindschadler et al 1996
    # Uses flow angle, counter clockwise from x-axis
    alpha = (np.arctan(vy/vx)) # in radians
    elon   = exx * np.cos(alpha)**2 + 2*exy * np.cos(alpha)*np.sin(alpha) + eyy*np.sin(alpha)**2
    etrans = exx * np.sin(alpha)**2 - 2*exy * np.cos(alpha)*np.sin(alpha) + eyy*np.cos(alpha)**2
    eshear = (eyy-exx) * np.cos(alpha)*np.sin(alpha) + exy * ( np.cos(alpha)**2 - np.sin(alpha)**2 )
    strain_components = (elon, etrans, eshear)

    ## Orientation of principal strain

    theta_p = (np.arctan(2*exy/(exx-eyy))/2) # in radians


    # theta_p to degrees
    theta_p_degr = theta_p*360/(2*np.pi)
    # convert to xarray dataArray
    # theta_p_degr = emax_xr.copy(data=theta_p_degr)

    if version2 is None:
        return emax_xr, emin_xr, theta_p_degr
    else:
        return emax_xr, emin_xr, e_eff, strain_components



def aggregate_region_ds_iceshelf( region_ds, varname , iceshelf_polygon_df, drop_clipped=False ):
    ''' 
    Function aggregates the spatial data within the specified (iceshelf) polygons to a single 1D np.array.
    Any spatial information is lost. NaN values are kept, to be able to combine multiple variables within the same region at a later stage.
    
    The output is of shape (Npixels , t) where t is the number of years available in the data.

    To do: implement reading of annual iceshelf polygons (currently working with a single shapefile for all years)

    Input
    -----
    region_ds           : xarray.Dataset of (y,x,time). 
    varname             : name of variable in dataset
    iceshelf_polygon_df : geopandas.DataFrame of polygon to clip data to
    drop_clipped        : If True:  Remove all pixels outside of polygon from array
                          If False: Keep all clipped pixels as NoData in array

    Output
    ------
    data_iceshelf_1d_yr     : numpy array of shape (Nsamples,y). Where y is the number of years in the 'time' dimension of region_ds 
    '''

    # set up empty list
    values_yearList = []

    # get time dimension of dataset
    years = region_ds['time'].values # array of years in dataSet

    # Aggregate annual data
    for year in years: 

        # -- get annual data
        current_yr_da  = region_ds.sel(time=year)[varname] # specifying variable translates xr.dataSet to xr.dataArray
        
        # -- clip to iceshelf
        current_yr_iceshelf  = clip_da_to_iceshelf(current_yr_da,  iceshelf_polygon_df, drop=drop_clipped) 
        
        # -- aggregate values to 1D np.array
        values_yr  = current_yr_iceshelf.values.reshape( -1,1) # extract values (matrix to array)

        # -- store in list
        values_yearList.append(values_yr)  # list with yearly np.array
    
    # merge list of annual values (1D array) to a np.array 
    if len(values_yearList) > 1:
        data_iceshelf_1D_yrs = np.concatenate( values_yearList , axis=1) # shape (Npx, Nyrs)
    else:
        data_iceshelf_1D_yrs = values_yearList[0] # np.expand_dims(values_yearList[0],1) # shape (Npx,1)

    return data_iceshelf_1D_yrs



def fill_nan_cdata(xdata,ydata,cdata,fill_value=-999):
    ''' Remove pixels where x/y data have nodata, but 
    fill pixels where cdata has nodata with a distinct value
    '''

    ###### remove nodata x/y variable
    idx_nan = np.any( np.isnan( np.concatenate( [xdata, ydata] ,axis=1 )), axis=1) # (Nsamples, stack) --> (Nsamples, )
    xdata = xdata[~idx_nan]
    ydata = ydata[~idx_nan]
    cdata = cdata[~idx_nan]

    ### fill nodata of c-variable
    idx_nan_cdata = np.isnan(cdata)
    xdata[idx_nan_cdata] = fill_value
    ydata[idx_nan_cdata] = fill_value
    cdata[idx_nan_cdata] = fill_value

    return xdata, ydata, cdata

def reproject_match_grid( ref_img_da, img_da , resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan):
    ''' Match xarray grid of different spatial resolutions. Input should be dataArray'''

    # Expected order: ('time', 'y', 'x')
    dims = img_da.dims
    ref_img_da = ref_img_da.transpose('time','y','x') # CRS is alreadyy written .rio.write_crs(3031, inplace=True)
    img_da = img_da.transpose('time','y','x')
    
    # -- reproject (even though same crs) and match grid (extent, resolution and projection)
    img_repr_match = img_da.rio.reproject_match(ref_img_da,resampling=resample_method,nodata=nodata_value) # need to specify nodata, otherwise fills with (inf) number 1.79769313e+308

    # advised to update coords to make the coordinates the exact same due to tiny differences in the coordinate values due to floating precision
    img_repr_match = img_repr_match.assign_coords({
        "y": ref_img_da.y,
        "x": ref_img_da.x,
    })
    
    return img_repr_match.transpose(*dims) # transpose dimension order back to original

def repeat_static_variable_timeseries( region_ds , varname_to_repeat ):
    ''' --------------------------------------
    Repeat temporally static variable (REMA; basalmelt) to even out dataset dimension
    ------------------------------------------ '''
    # REMA repeated for every year, to match dataset
    region_da_rema = region_ds.sel(time=0)[varname_to_repeat].drop('time') # only time slice where REMA has data -- (y,x); need to DROP time because it remains as some sort of passive dimension
    region_rema_yrs = region_da_rema.expand_dims(dim=dict(time=region_ds.time.values),axis=2) ## the variable is automatically repeated for the new dimension

    # dataset without REMA and rema's time dimension
    ## region_ds = region_ds.drop_sel(time=0).drop(['rema','basalmelt']) # drops a slice of dimension value but keeps dimension
    region_ds = region_ds.drop_sel(time=0).drop([varname_to_repeat]) # drops a slice of dimension value but keeps dimension

    ## Put it back into dataset
    region_ds[varname_to_repeat] = region_rema_yrs
    ## region_ds['basalmelt'] = region_bmelt_yrs

    return region_ds


def clip_and_aggregate_to_df( region_ds, iceshelf_polygon_gpd):
    ''' --------------
    Aggregate values to 1D: ON xr.DATASET
    - Temporaly static variables should already be repeated, to match datashape
    - Clip 2D to ice shelf (TO DO: clip to annual ice shelf) (2D xarray)
    - Reshape to 1D by xarray.stack (spatial information is retained)
    - Remove all pixels that have NaN value in any one of the variables (TO DO: do this per year, to account for changing ice shelf polygon) 
    - Format arrays to a pandas dataFrama
    --------------'''

    ''' Clip to iceshelf  '''
    region_ds  = region_ds.rio.clip( iceshelf_polygon_gpd.geometry, iceshelf_polygon_gpd.crs, drop=False, invert=False)

    ''' Aggregate to 1D '''
    region_ds_1d = region_ds.stack(samples=['x','y']) # (time, samples)

    ''' Convert to dataFrame '''
    data_pxs_df = region_ds_1d.to_dataframe() # nested dataframe

    # Flatten the nested multi-index to just column values -- automatically generates a 'year' value for every sample
    data_pxs_df = data_pxs_df.reset_index(level=['time','x','y']) # 18767504 rows
    # Drop spatial ref (has not data acutally) 
    data_pxs_df = data_pxs_df.drop(['spatial_ref'],axis=1)
    # For now: drop x and y values; For spatial k-fold: do not drop x and y
    # data_pxs_df = data_pxs_df.drop(['x','y'],axis=1)

    ''' Drop NaN pixels:
    Pandas drops all rows that contain missing values. 
    - This means that if any variable has a NaN value, that px is dropped.
    --> should make sure to fill NaN values for variables before this step (e.g. dmg NaN is set to 0; filling of REMA gaps)
    - Since I have rows for px per year, this means that if I would have clipped the data to annual ice shelf polygons, the number of pixels per year can vary.  
    '''
    data_pxs_df.dropna(axis='index',inplace=True) # Drop rows which contain missing values.
    data_pxs_df.head()

    return data_pxs_df


def downsample_dataArray_withoutD0( region_ds , ksize=3, boundary_method='pad', downsample_func = 'mean', skipna=None,verbose=True):
    ''' --------------
    Downsample data
    - First fill all dmg=0 values with np.nan. such that these are discarded during downsampling
    - Then coarsen the netCDF (downsample resolution)
        - downsampling with median: better suited to omit impact of outliers
        - downsampling with mean:   'general representation of field'. Yields relatively high values for kernels with predominantly low values - in the case of 'dmg' this is what you'd want.
        - downsampling with max:    too much attention to outliers within kernel.
    - Then put dmg=0 back into array (such that when removing 'nan px' from all variables that were clipped to ice shelf polys, these px are retained)
    - Later on, it might be decided that dmg=0 will still be discarded.

    ksize:              Kernel size (widthxheight) that will be downsampled
    boundary_method:    How to handle boundary in xr.DataSet.coarsen()
    downsample_func:    Define if to downsample using kernel mean or median (other not supported yet)
    --------------'''
    if verbose:
        print('..Downsampling data {}x{} pxs'.format(ksize,ksize))

    # Get all dmg variables within array (e.g. 'dmg' and 'dmg095' for updated thresholds)
    all_dmg_vars = [varname for varname in list(region_ds.data_vars) if 'dmg' in varname]
    
    # Fill dmg=0 temporarily with np.nan, so that during downsampling these values are not considered
    for dmg_var in all_dmg_vars:
        region_ds[dmg_var] = region_ds[dmg_var].where(region_ds[dmg_var]>0, other=np.nan)

    if downsample_func == 'mean':
        region_ds_coars  = region_ds.coarsen(x=ksize,y=ksize,boundary=boundary_method).mean() # mean downsampling 

    if downsample_func == 'median':
        region_ds_coars  = region_ds.coarsen(x=ksize,y=ksize,boundary=boundary_method).median() # median downsampling
    if downsample_func == 'max':
        region_ds_coars  = region_ds.coarsen(x=ksize,y=ksize,boundary=boundary_method).max() # max downsampling
    return region_ds_coars


def calculate_velo_strain_features(data_ds, velocity_names=('xvelsurf','yvelsurf'), length_scales=['1px']):
    ''' Calculate velocity from horizontal vx and vy components, multiple strain components and temporal  change of velocity and strain magnitude.
    
    Returns 2 datasets:
    
    data_velo_strain    :   contains velocity magnitude 'v' and 'deltaV' (max annual velocity change with smoothened by 3-yrs trailing window)
    region_ds_roll      :   contains strain components:
                            - max and min principal strains, emax and emin
                            - effective strain, e_eff
                            - longitudonal (elon), transverse (etrans) and shear (eshear) strain
                            - deltaEmax (max annual change of emax smoothened by 3-yrs trailing window)
     '''
    var_name_vx, var_name_vy = velocity_names 

    ''' --------------
    Calculate velocity
    ------------------ '''
    # # velocity magnitude
    region_da_v = ((data_ds[var_name_vx]**2 + data_ds[var_name_vy]**2)**0.5) # .to_dataset(name='v')

    data_ds['v'] = region_da_v

    ''' --------------
    Calculate Strain
    ------------------ '''
    # strain on multiple lenthscale -- version2
    
    for scale_name in length_scales: 
        lscale = int(scale_name.strip('px'))

        # calculate
        emax,  emin, e_eff, strain_components  = calc_nominal_strain(data_ds[var_name_vx], data_ds[var_name_vy], 
                                                                            length_scale_px=lscale , 
                                                                            version2 = True)
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
        
        ## Add minimal (needed for temporal calculations)
        data_ds = xr.merge([data_ds, emax.to_dataset( name='emax_'+str(lscale)+'px') ] )

        ## Velo-strain dataset:
        data_velo_strain = xr.merge([region_da_v.to_dataset(name='v'), region_ds_strain])

        ''' ------------
        Calculate temporal values 
        ---------------- '''

        ## Calculate difference per year (first year is dropped)
        region_ds_diff = data_ds[['emax_'+scale_name, 'v']].diff(dim='time').rename({'emax_'+scale_name:'deltaEmax','v': 'deltaV'})
        # print(region_ds_diff)

        ## Get rolling-max diff of past 3 years. Set center=False so the window is a trailing window i-2 to i
        ## NB: with min_periods=1, the first year will have the same values as itself
        region_ds_roll = region_ds_diff[['deltaEmax','deltaV']].rolling(time=3, 
                                center=False, min_periods=1).max().rename(
                                {'deltaEmax':'dEmax_'+scale_name,
                                # 'deltaV':'dV_'+scale_name
                                }) 

        ## add to dataset
        # data_ds = xr.merge([data_ds, region_ds_roll])
    # return data_ds
    return data_velo_strain, region_ds_roll



def make_regular_grid_for_ds(ds, grid_res):
    x_seq = np.arange(ds.x.min(), ds.x.max()+grid_res, step=grid_res )
    y_seq = np.arange(ds.y.min(), ds.y.max()+grid_res, step=grid_res )

    # dimension in dataset:
    dims = ds.dims
    if 'time' in dims:
        t_seq = ds.time # .values
        grid_dummy = xr.DataArray(
            data=np.ones( (len(t_seq), len(y_seq),len(x_seq) ) ), # dummy values
            dims=["time", "y", "x"],
            coords=dict( # proper coordinate/time values
                time=t_seq, 
                y=y_seq,
                x=x_seq,
            ),
            attrs=dict(
                description="dummy_regular_grid",
            ),
        ).rio.write_crs(3031)

    else:
        grid_dummy = xr.DataArray(
            data=np.ones( (len(y_seq), len(x_seq)) ),
            dims=["y", "x"],
            coords=dict(
                y=y_seq,
                x=x_seq,
            ),
            attrs=dict(
                description="dummy_regular_grid",
            ),
        ).rio.write_crs(3031)


    return grid_dummy


def drop_spatial_ref(ds):
    try:
        ds = ds.drop('spatial_ref')
    except:
        pass
    return ds 



def check_presence_of_ismip_variable_file(varname, filelist):
    if varname == 'xvelsurf':
        varname_subst = 'xvelmean'
    elif varname == 'yvelsurf':
        varname_subst = 'yvelmean'
    elif varname == 'orog':
        varname_subst = None
    else:
        raise ValueError('Expect variable name "xvelsurf", "yvelsurf" or "orog", not "{}"'.format(varname))

    if not any([ (varname in file) for file in filelist ]):
        # use xvelmean instead of xvelsurf, if available (same for yvelsurf)
        if any([ (varname_subst in file) for file in filelist ] ):
            print('Variable {} not available; using substitute variable {}'.format(varname, varname_subst))
            return varname_subst
        else:
            raise NameError('Required variable "{}", neither its substitutde "{}" is found\n{}'.format(varname, varname_subst,filelist))
    else:
        return varname


def make_ais_grid( grid_res):
    xmin, xmax = -3040000, 3040000
    ymin, ymax = -3040000, 3040000
    x_seq = np.arange(xmin, xmax, step=grid_res )
    y_seq = np.arange(ymin, ymax, step=grid_res )

    # ais_dummy = np.ones( (len(x_seq),len(y_seq) ) )
    ais_dummy = np.ones( (len(y_seq),len(x_seq) ) )

    ais_dummy = xr.DataArray(
        # data=np.ones( (len(x_seq),len(y_seq) ) ),
        # dims=["x", "y"],
        data=np.ones( (len(y_seq),len(x_seq) ) ),
        dims=["y", "x"],
        coords=dict(
            y= y_seq,
            x= x_seq,
        ),
        attrs=dict(
            description="DummyAISgrid",
        ),
    ).rio.write_crs(3031)
    return ais_dummy 


def reprj_regions_to_ais_grid(ais_da, img_da):
    img_da.rio.write_crs(3031, inplace=True)
    
    # -- reproject (even though same crs) and match grid (extent, resolution and projection)
    img_repr_match = img_da.rio.reproject_match(ais_da,resampling=rio.enums.Resampling.nearest,nodata=np.nan) # need to specify nodata, otherwise fills with (inf) number 1.79769313e+308

    # advised to update coords
    img_repr_match = img_repr_match.assign_coords({
        "y": ais_da.y,
        "x": ais_da.x,
    })

    return img_repr_match


def update_stricter_dmg_threshold( dmg_da , dmg_threshold ,reduce_or_mask_D='reduce',verbose=False):

    ''' --------------------------------------
    S1: Update dmg threshold to stricter value
    dmg as is: quantiles [0, 25, 50, 75, 1] are at: [0 ,   0.007,  0.012,  0.022,  0.289] -- BASED ON ALL ASE TILES
    tresh: 0.015 is good for plotting
    tresh: 0.007 might (??) be good for boxplots
    tresh: 0.043: new noise trheshold based on max(no-dmg-signal) instead of mean(no-dmg-signal) (new threhsold = 0.08, but need to update original t=0.037 with 0.043 for that)

    Newly calculated dmg threshold:
    Original: tau = 0.037 (based on mean(no-dmg) signal )
    New       tau = 0.053 / 0.063 (based on pct 095/099 no-dmg signal)
    -- should update dmg with   (0.053-0.037= 0.016)
                                (0.063-0.037= 0.026)
    ------------------------------------------ '''


    if reduce_or_mask_D == 'reduce':
        if verbose:
            print("..Applying stricter threshold to dmg values (D = D - threshold): {}".format(dmg_threshold))
        
        # --lower dmg value
        region_da_dmg_prune = dmg_da - dmg_threshold # region_ds_dmg - d_tresh
        region_da_dmg_prune = region_da_dmg_prune.where(region_da_dmg_prune > 0, other=0) # .rename('dmg_prune') # sets other px to 0 -- because I dont want to remove NaN px too early

    elif reduce_or_mask_D == 'mask':
        if verbose: 
            print("..Applying stricter threshold to dmg values (D = D where D < threshold): {}".format(dmg_threshold)) 
        # --do not reduce d-value, only mask low values 
        # (NB: need to be very sure that you want this, as it violates the NERD dmg-signal consistency w.r.t other data sources)
        region_da_dmg_prune = dmg_da.where(dmg_da['dmg'] > dmg_threshold, other=0) # .rename('dmg_prune') # sets other px to 0 -- because I dont want to remove NaN px too early

    return region_da_dmg_prune


def load_nc_sector_years( path2data, sector_ID, year_list=None, varName=None ):
    ''' Load all/selected annual netCDF files of a variable for one sector'''

    ## get filelist of variable for current sector
    filelist_dir =  glob.glob( os.path.join(path2data, f'*_sector-{sector_ID}_*.nc') )
    filelist_var_all = [file for file in filelist_dir if varName in file]
    filelist_var_all.sort()

    if not filelist_var_all:
        print(f'no files found for {varName} at', path2data)

    ## select files for all/specified years 
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
            # print(filelist_yr)
            if not filelist_yr:
                raise ValueError(f'Could not find year {year}')
            filelist_var.append(filelist_yr)

    ## Open dataset(s)

    try: # read all years at once
        region_ds = (xr.open_mfdataset(filelist_var ,
                    combine='nested', concat_dim='time',
                    compat='no_conflicts',
                    preprocess=drop_spatial_ref)
            .rio.write_crs(3031,inplace=True)
            .assign_coords(time=year_list) # update year values (y,x,time)
        )
    except ValueError: # read year by year, then concatenate
        region_list = []
        for file in filelist_var:
            yr = int( re.search(r'\d{4}', os.path.basename(file[0])).group()) 
            # print(yr)
            with xr.open_mfdataset(file) as ds:
                try:
                    ds.assign_coords(time=yr)
                except: pass
                region_list.append(ds.rio.write_crs(3031,inplace=True))
        region_ds = xr.concat(region_list,dim='time')  
        # print(region_ds.coords) 
    return region_ds

def downsample_obs_data(region_ds,ksize):
    dx = int(region_ds.rio.resolution()[0])
    dy = int(region_ds.rio.resolution()[1])
    if np.abs(dx) != np.abs(dy):
        print("Warning: x and y resolution are not the same; {} and {} -- code update required".format(np.abs(dx), np.abs(dy) ))

    # with dask.config.set(**{'array.slicing.split_large_chunks': True}): # gives error?
    with dask.config.set(**{'array.slicing.split_large_chunks': False}): ## accept large chunks; ignore warning
        region_ds = downsample_dataArray_withoutD0(region_ds, ksize=ksize, 
                                    boundary_method='pad',downsample_func='mean', skipna=False)
    new_res = ksize*400
    print('.. resolution {}m downsampled to {}m'.format(dx, new_res))


    # Check if grid resolution is regular (otherwise, adjust)
    if np.abs(int(region_ds.rio.resolution()[0]) ) != np.abs( int(region_ds.rio.resolution()[1]) ):
        print( "x and y resolution are not the same; {} and {} -- resample to regular grid of {}m".format(
                    np.abs(int(region_ds.rio.resolution()[0])), 
                    np.abs(int(region_ds.rio.resolution()[1])), new_res ))
        
        grid_dummy = make_regular_grid_for_ds(region_ds, grid_res=new_res)
        region_ds.rio.write_crs(3031, inplace=True)
        region_ds = reproject_match_grid( grid_dummy, region_ds )         

    return region_ds