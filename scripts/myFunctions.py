import rioxarray as rioxr
import xarray as xr
import glob
import numpy as np
import os
import geopandas as gpd 
import rasterio as rio
import warnings

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


def load_tile_yxt( tilepath , subdir , tileNum, year=None, name='unknown', CRS=3031):
    '''Load a datatile from disk to an xarray.dataArray
    
    Input
    -----
    tilepath    :   Path to data folder containing geotiffs of the variable.
    subdir      :   Subdirectory. Usually, the data is stored in a folder-per-year (e.g. '2016' or '2016-SON')
    tileNum     :   Tile Number to load. Float or Integer
    year        :   The year of the dataset that is loaded. 
                    If None is given, the function first tries to infer the value from the subfolder name and otherwise sets year=0.
                    The value is used to define the temporal dimension of the xr.DataArray
    name        :   Variable name to set in xr.DataArray
                    Also used to filter files, if there are more variables stored in the same directory.
    CRS         :   The projection of the data (number). Default EPSG:3031

    
    Output
    ------
    img         :   An xr.DataArray of the loaded file in shape (y,x,time). Includes CRS.

    '''
    
    # search for tilenumber in directory
    tile_file = glob.glob(os.path.join(tilepath,subdir ,'*_'+str(int(tileNum))+'.tif'))
    if not tile_file:
        raise ValueError('No files found for {}'.format(os.path.join(tilepath,subdir ,'*_'+str(int(tileNum))+'.tif') ))
    elif len(tile_file) > 1: # multiple files present. Further filter search for variable name
        tile_file = [ fname for fname in tile_file if name in fname ] 
    
    if len(tile_file) > 1:
        raise ValueError('Multiple tiles found for {}:\n {}'.format(
                    os.path.join(tilepath,subdir ,'*_'+str(int(tileNum))+'.tif') ,
                    tile_file))
    
    tile_file = tile_file[0] # 1-element list to string

    # if name not in tile_file:
    #     raise ValueError('Found tile in folder, but varname {} does not occur in filename {}'.format(name, os.path.split(tile_file)[1]))

    # read with rioxarray
    img = rioxr.open_rasterio(os.path.join(tilepath , subdir, tile_file)).squeeze() # read geotiff
    # img.rio.write_crs(3031, inplace=True) # make sure it has the correct projection information
    img.name = name
    img.attrs['long_name']=name

    # If the year of the dataset is not specified, infer information of what year the data is, or set to 0
    if year is None:
        try:
            year=int(subdir)
        except:
            year=0

    # add time-dimension to xarray.DataArray
    img = xr.DataArray( data = np.expand_dims(img.data,-1),  # (y,x) to (y,x,1)
                        coords={'y': (img["y"]),
                                'x': (img["x"]),
                                'time':([int(year)])},
                        name=name, 
                        attrs=img.attrs, indexes=img.indexes)  # copy other properties
    img.rio.write_crs(CRS, inplace=True) # make sure it has projection information

    return img


def clip_da_to_iceshelf(data_da, iceshelf_polygon_gpd,drop=False):
     ''' Clip data (xarray.DataArray) to ice shelf polygon file (geopandas.DataFrame) '''
     return data_da.rio.clip( iceshelf_polygon_gpd.geometry, iceshelf_polygon_gpd.crs, drop=drop, invert=False)


def check_theta_p_e11_e22(exx,eyy,exy,emax,theta_p):
    '''Identify if calculated principal strain orientation aligns with e_11 or e_22
    
    Input
    -----
    exx:    strain tensor e_11
    eyy:    strain tensor e_22
    exy:    shear strain tensor e_12=e_21 (1/2 of engineering strain)
    emax:   
    '''
        
    # Construct strain and angle matrices
    E = np.array([ [exx.squeeze(), exy.squeeze() ], 
                   [exy.squeeze(), eyy.squeeze() ] ]) # shape (x y n k) (2, 2, 494, 401) <-- should be (494, 401, 2, 2)

    Q = np.array([ [ np.cos(theta_p.squeeze()), np.sin(theta_p.squeeze())], 
                   [-np.sin(theta_p.squeeze()), np.cos(theta_p.squeeze())] ])  # shape (x y k m) (2, 2, 494, 401) <-- should be (494, 401, 2, 2)      

    Qt = np.transpose(Q, (1, 0, 2, 3)) # (2, 2, 494, 401); transposed the (2,2) axes

    # move axes for np.matmul (from (2,2,x,y) to (x,y, 2, 2 ))
    E = np.moveaxis(E, [0, 1], [-2, -1]) # shape (494, 401, 2, 2); (x y n k)
    Q = np.moveaxis(Q, [0, 1], [-2, -1]) # shape (494, 401, 2, 2); (x y k m)
    Qt = np.moveaxis(Qt, [0, 1], [-2, -1]) #  shape (494, 401, 2, 2); (x y k m) 

    Edot = np.matmul(np.matmul(Q,E),Qt)

    # Update_theta_p to match e11
    e11 = Edot[:,:,0,0]
    e22 = Edot[:,:,1,1] 
    ndec = 6

    # idx where emax=e11, theta_p is correct 
    idx_e11_eq_emax = e11.round(ndec) == emax.values.round(ndec) 

    # idx where emax=e22, theta_p should be + 90 degree
    idx_e22_eq_emax = e22.round(ndec) == emax.values.round(ndec) 

    # if emax is not e11, it should be e22
    idx_correct = ~idx_e11_eq_emax == idx_e22_eq_emax 
    idx_correct[np.isnan(e11)] = True
    # --> check pixels where this is not true
    if not idx_correct.all():
        diff_e11 = np.nan_to_num(np.abs(emax-e11))
        diff_e22 = np.nan_to_num(np.abs(emax-e22)) 
        min_diff =  np.array((diff_e11,diff_e22)).min(axis=0)
        mindiff_e11_or_e22 = np.array((diff_e11,diff_e22)).argmin(axis=0) # 0 of minimum is found in e11; 1 if minimum diff is found in e22
        mindiff_e11_or_e22[idx_correct] = -999                # skip pixels that were already correct

        idx_e11_eq_emax[np.where(mindiff_e11_or_e22 == 0)] = True
        idx_e22_eq_emax[np.where(mindiff_e11_or_e22 == 1)] = True

        idx_correct = ~idx_e11_eq_emax == idx_e22_eq_emax 
        idx_correct[np.isnan(e11)] = True

    return idx_e22_eq_emax


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

    ## to do: Align orientation of principal strain to e_11 or e_22. Check if theta_p aligns with e_11 (then e_11 is emax) or e_22 then update theta_p value + 0.25 pi
    # # idx_e22_eq_emax = check_theta_p_e11_e22(exx,eyy,exy,theta_p)
    # idx_e22_eq_emax = check_theta_p_e11_e22(exx,eyy,exy,emax, theta_p) # theta_p aligns with e_22, aka e_min --> update theta_p value with +0.25 pi 
    # theta_p = theta_p.squeeze().values
    # theta_p[idx_e22_eq_emax] +=  np.pi/4 # 90 degrees is 1/4 pi 
    # # theta_p to degrees
    # theta_p_degr = theta_p*360/(2*np.pi)
    # # convert to xarray dataArray
    # theta_p_degr = emax_xr.copy(data=np.expand_dims(theta_p_degr,axis=-1))

    # theta_p to degrees
    theta_p_degr = theta_p*360/(2*np.pi)
    # convert to xarray dataArray
    # theta_p_degr = emax_xr.copy(data=theta_p_degr)

    if version2 is None:
        return emax_xr, emin_xr, theta_p_degr
    else:
        return emax_xr, emin_xr, e_eff, strain_components

def load_tiles_region_multiyear( tiledVar_path, tile_list , years_to_load=[] , varname='var_name' ):
    '''
    This function loads the 2D data for all the tiles specified in the tile_list to a single xr.DataArray
    If the variable has data for multiple years, these are appended as a 3rd temporal dimension.
    If years_to_load is None or Empty, a single 'year' is loaded, and given value time=0
    
    The data is loaded to a (y,x,t) format.
    NB: data attributes are lost in the merging process. 
    '''
    
    # set up empty list
    var_tileList = []

    for tileNum in tile_list: # [52]

        # load tile data -- for multi years
        if years_to_load: # if list is not empty, load multiple years

            var_tile_yearList = []
            for year in years_to_load: 
                ''' Load annual img to (y,x,t) format xarray.DataArray
                    Store these da's into a list, to combine them into one dataArray after the loop
                '''
                # subdirectory can be in format '2015-SON' or simply '2015', so include a check to its existance
                if os.path.isdir(os.path.join(tiledVar_path, year)):
                    subdir=year
                elif os.path.isdir(os.path.join(tiledVar_path, year+'-SON')):
                    subdir=year+'-SON'

                # load data_yxt, set name of dataset variable
                data_img  = load_tile_yxt(tiledVar_path,  subdir=subdir, 
                                        tileNum = tileNum , year=year, name=varname) 

                # store annual dmg/emax dataArray in list 
                var_tile_yearList.append(  data_img) # list with DAs 

            # Merge list of annual dataArrays into one (returns xr.dataSet because dataArrays are named)
            # TO DO: attributes are lost somewhere here
            tile_ds  = xr.combine_by_coords(var_tile_yearList   , combine_attrs='override')  # stacks years correctly 
            
        else: # only one year to load  
            subdir = ''

            # load data_yxt, set name of dataset variable
            tile_ds  = load_tile_yxt(tiledVar_path,  subdir=subdir, 
                                    tileNum = tileNum , name=varname).to_dataset()
        
        # store current tiledata in list
        var_tileList.append( tile_ds )
    

    # Merge all tiles within region 
    # TO DO: attributes are lost somewhere here

    try:
        region_ds  = xr.merge(var_tileList, combine_attrs='override')
    
    except xr.MergeError:
        print('Merge error; try with compat=''override''')
        region_ds  = xr.merge(var_tileList, combine_attrs='override',compat='override')

    # # ## V2: merge with reproject; to make sure grids are merged correctly
    # var_tileList2=[]
    # tile_ref = var_tileList[0].transpose('time','y','x')
    # for tile in var_tileList[1:]:
    #     tile_matched = tile.transpose('time','y','x').rio.reproject_match(tile_ref,resampling=rio.enums.Resampling.nearest,nodata=np.nan) # need to specify nodata, otherwise fills with (inf) number 1.79769313e+308
    #     # Update coords (advised)
    #     tile_matched = tile_matched.assign_coords({
    #         "y": tile_ref.y,
    #         "x": tile_ref.x,
    #     })
    #     var_tileList2.append(tile_matched)
    # ## region_ds = xr.concat(var_tileList2,dim='time',join='outer',coords='all',combine_attrs='override')#,fill_value=1)
    # # region_ds  = xr.merge(var_tileList2, combine_attrs='override')

    # release linked resources
    region_ds.close()
    
    return region_ds


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


# def remove_nanpx_multivar0( xdata, ydata, cdata):
#     ''' Removes all pixels from data arrays that has a NaN value in either one of the arrays.

#     Input
#     -----
#     xdata and ydata should have the same shape (N, y), and cdata is expected to have shape (N,) or (N,1)
#     If y>1, the datasets are compared row by row (e.g. x[n,1] to y[n,1] and x[n,2] to y[n,2] ).

#     Output
#     ------
#     Returns the data arrays with removed nan-pixels

#     '''
    
#     ########## V1

#     # if len(cdata.shape)>1: # cdata has shape (n,1) or (n,y)
#     #     if cdata.shape[1] > 1:
#     #         raise ValueError('Cdata expected to have shape (n,) or (n,1) but has shape {}'.format(cdata.shape))
#     # else: # convert shape (n,) to (n,1)
#     #     cdata = np.expand_dims(cdata,1)
    
#     # if xdata.shape != ydata.shape:
#     #     raise ValueError('xdata and ydata should have the same shape, but are x:{} and y:{}'.format(xdata.shape,ydata.shape))
    

#     # if len(xdata.shape)>1: # input has shape (n, y); cdata should be (n,1)
#     #     print('x y c shape ', xdata.shape, ydata.shape,cdata.shape)
#     #     if xdata.shape[1] > 1:
#     #         # extend data to have 3rd dim for concatenation
#     #         x2 = np.expand_dims(xdata,2) # n,y,1
#     #         y2 = np.expand_dims(ydata,2) # n,y,1
#     #         c2 = np.expand_dims(cdata,2) # n,1,1
#     #         # repeat cdata to create same dimensions 
#     #         c3 = np.repeat(c2,4,axis=1) # (n,1,1) --> (n,4,1)
#     #         data = np.concatenate([x2,y2,c3],axis=2) # n,y,2
#     #         print('concat x y c', data.shape)

#     #         id = np.any( np.isnan( data ), axis=2 , keepdims=True) # n,y,1
#     #         xdata_clean = np.where( ~id , x2, np.nan  ).squeeze() # n,y
#     #         ydata_clean = np.where( ~id , y2, np.nan  ).squeeze() # n,y
#     #         cdata_clean = np.where( ~id , c3, np.nan  ).squeeze() # (n,y)
#     #         print('xdata remove nan ' , xdata_clean.shape)

#     #     elif xdata.shape[1] == 1: # (n,1)
#     #         data = np.concatenate([xdata,ydata,cdata],axis=1) # n,3
#     #         print('concat x y c, ' , data.shape)
#     #         id = np.any( np.isnan( data ), axis=1 , keepdims=True) # n,1
#     #         xdata_clean = np.where( ~id , xdata, np.nan  )# .squeeze() # n,1
#     #         ydata_clean = np.where( ~id , ydata, np.nan )
#     #         cdata_clean = np.where( ~id , cdata, np.nan )
#     #         print('xdata remove nan ' , xdata_clean.shape)
            
#     # else: # xdata and ydata have shape (n, ); cdata should already be (n,1)
#     #     print('Input shapes x,y,c', xdata.shape, ydata.shape, cdata.shape)
#     #     x2 = np.expand_dims(xdata,1)
#     #     y2 = np.expand_dims(ydata,1)
#     #     data = np.concatenate([x2,y2,cdata],axis=1) # n,3
#     #     print('concat x y c, ' , data.shape)
#     #     id = np.any( np.isnan( data ), axis=1 , keepdims=True) # n,1
#     #     xdata_clean = np.where( ~id , x2, np.nan  ).squeeze() # n,
#     #     ydata_clean = np.where( ~id , y2, np.nan  ).squeeze() # n,
#     #     cdata_clean = np.where( ~id , cdata, np.nan  ).squeeze() # n,

#     ######### V0b

#     # # if any([len(xdata) > 1, len(ydata) > 1, len(cdata)>1]):
#     # #     if any( [xdata.shape[1]>1,ydata.shape[1]>1,cdata.shape[1]>1] ):
#     # #         raise ValueError('Expected data of shape (n,) or (n,1), but has shape x:{}, y:{}, c:{}'.format(xdata.shape,ydata.shape,cdata.shape))
#     # if len(xdata.shape)>1: # cdata has shape (n,1) or (n,y)
#     #     if xdata.shape[1] > 1:
#     #         raise ValueError('xdata expected to have shape (n,) or (n,1) but has shape {}'.format(xdata.shape))  
#     # else: # convert shape (n,) to (n,1)
#     #     xdata = np.expand_dims(xdata,1)

#     # if len(ydata.shape)>1: 
#     #     if ydata.shape[1] > 1:
#     #         raise ValueError('ydata expected to have shape (n,) or (n,1) but has shape {}'.format(ydata.shape)) 
#     # else: # convert shape (n,) to (n,1)
#     #     ydata = np.expand_dims(ydata,1)
        
#     # if len(cdata.shape)>1: 
#     #     if cdata.shape[1] > 1:
#     #         raise ValueError('cdata expected to have shape (n,) or (n,1) but has shape {}'.format(cdata.shape)) 
#     # else: # convert shape (n,) to (n,1)
#     #     cdata = np.expand_dims(cdata,1)

#     ###### v0a
#     idx_nan = np.any( np.isnan( np.concatenate( [xdata, ydata, cdata] ,axis=1 )), axis=1) # (Nsamples, stack) --> (Nsamples, )
#     xdata_clean = xdata[~idx_nan]
#     ydata_clean = ydata[~idx_nan]
#     cdata_clean = cdata[~idx_nan]
#     return xdata_clean, ydata_clean, cdata_clean


def remove_nanpx_multivar(*datasets): # datasets is tuple
    
    # Identify pixels that have NaN value in any of the data input 
    idx_nan = np.any( np.isnan( np.concatenate( datasets ,axis=1 )), axis=1) # (Nsamples, stack) --> (Nsamples, )
 
    datasets_clean = [] # empty list
    for var in datasets: # remove nanpixels from every variable
        var_clean = var[~idx_nan]
        datasets_clean.append(var_clean) # append list

    return tuple(datasets_clean) # make tuple from list of variables    


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
    var_name_vx, var_name_vy = velocity_names 

    ''' --------------
    Calculate Velocity
    ------------------ '''
    # velocity magnitude
    region_da_v = ((data_ds[var_name_vx]**2 + data_ds[var_name_vy]**2)**0.5) # .to_dataset(name='v')

    data_ds['v'] = region_da_v

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

        ## add to dataset
        # data_ds = xr.merge([data_ds, region_ds_strain])
        
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


def load_nc_obs_data( path2data, region_ID, varname=None, parts=['part1'],verbose=True):
    ''' ----------------------
    Load data: netCDFs per region, per variable
    ------------------------- '''
    if verbose:
        print('----\n Loading netCDF for region ', region_ID)

    ## Retrieve list of files for single/multi variable
    if not varname:
        ''' Load all variables from individual netCDF files '''
        filelist = glob.glob(path2data + '*region-'+region_ID+'_*.nc')
        filelist = [file for file in filelist if 'all2' not in file]
        filelist = [file for file in filelist if 'strain' not in file]
        filelist.sort()              
    else: 
        ''' Load files of single variable'''
        filelist = glob.glob(path2data + '*region-'+region_ID+ '_*'+ varname +'*nc')
        filelist.sort()

    ## Filter list for desired 'parts' 
    for part in parts:
        if not part in ['part1','part2','1997']:
            raise ValueError('Selected partition {} not in [''part1'',''part2'',''1997'']'.format(part))

    
    # for part in parts:
    #     files_include = [file for file in filelist if part in file]
    #     # region_filelist += files_include

    #     # region_ds = xr.open_mfdataset( region_filelist,  
    #                 combine="by_coords",decode_times=True,
    #                 data_vars='minimal', 
    #                 coords= 'minimal', 
    #                 compat='broadcast_equals', #  all values must be equal when variables are broadcast against each other to ensure common dimensions.
    #             #   chunks={'y':'auto','x':'auto','time':5}, # add chucnking info for dask
    #                 chunks={'y':2000,'x':2000,'time':5}, # add chucnking info for dask: multitude of downsampling size
    #                 )  
    #     print('Loaded variables: \n', list(region_ds.keys()) )
    # else:

    if len(parts) == 1:
        part=parts[0]
        ''' Load selected variable '''
        # # region_varfile = glob.glob(path2data + '*region-'+region_ID+ '_*'+ varname +'_'+part+'*nc')[0]
        region_varfile = [file for file in filelist if part in file]
        if len(region_varfile)>1:
            raise ValueError('Expected 1 file for part {}, found: {}'.format(part, region_varfile) )
            
        region_varfile = region_varfile[0]
        # load single file
        region_ds = xr.open_dataset( region_varfile )

    else: # combine all parts
        # filelist = glob.glob(path2data + '*region-'+region_ID+ '_*'+ varname +'*nc')
        region_filelist = []
        for part in parts:
            files_include = [file for file in filelist if part in file]
            region_filelist += files_include
        
        with xr.open_mfdataset( region_filelist,  
                combine="by_coords",decode_times=True,
                data_vars='minimal', 
                coords= 'minimal', 
                compat='broadcast_equals', #  all values must be equal when variables are broadcast against each other to ensure common dimensions.
                chunks={'y':2000,'x':2000,'time':1}, # add chucnking info for dask: multitude of downsampling size
                # engine='scipy',
                )  as region_ds:
                print('opening region_ds')

    print('Loaded variables: \n', list(region_ds.keys()) )

    return region_ds




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