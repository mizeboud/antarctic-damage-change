import os
# import rioxarray as rioxr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import glob
import xarray as xr

import rasterio as rio

import pandas as pd 

import seaborn as sns

# Import user functions
# get_tilelist_region, load_tile_yxt, clip_da_to_iceshelf, 
# load_tiles_region_multiyear, aggregate_region_ds_iceshelf , 
# remove_nanpx_multivar, fill_nan_cdata, reproject_match_grid
import myFunctions as myf 
import argparse 

''' --------------
Paths
------------------ '''
# homedir = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/'

# vrlab:
homedir = '/net/labdata/maaike/'

path2savefig = os.path.join(homedir,'Data/NERD/plots_dev/boxviolin/')
path2data = os.path.join(homedir,'Data/NERD/data_predictor/')


''' --------------
Get Shapefiles 
------------------ '''
# geojson
gridTiles_geojson_path = os.path.join(homedir,'Data/tiles/gridTiles_iceShelves_EPSG3031.geojson')
gridTiles = gpd.read_file(gridTiles_geojson_path)

# measures ice shelves
# iceshelf_path_meas = os.path.join(homedir, 'QGis/Quantarctica/Quantarctica3/Glaciology/MEaSUREs Antarctic Boundaries/IceShelf/IceShelf_Antarctica_v02.shp')
iceshelf_path_meas = os.path.join(homedir, 'Data/SHAPEFILES/IceShelf_Antarctica_v02.shp')
iceshelf_poly_meas = gpd.read_file(iceshelf_path_meas)

# #regions of interest for AIS
# roi_path = os.path.join(homedir, 'QGis/data_NeRD/plot_insets_AIS_regions.shp')
roi_path = os.path.join(homedir, 'Data/SHAPEFILES/plot_insets_AIS_regions.shp')
roi_poly = gpd.read_file(roi_path)

region_ID_list = roi_poly['region_ID'].to_list()


## redefined: SECTORS for AIS
# sector_path = os.path.join(homedir, 'QGis/data_NeRD/plot_insets_AIS_sectors.shp')
sector_path = os.path.join(homedir, 'Data/SHAPEFILES/plot_insets_AIS_sectors.shp')
sector_poly = gpd.read_file(sector_path)
# sector_ID_list = sector_poly['sector_ID'].to_list()

# sector_ID_list

''' --------------
Colors
------------------ '''
# get colors
N = 9
# cmap = plt.cm.get_cmap('Pastel1', N) # plt.cm.viridis(N) #  / 10.)
# colors_pastel = cmap(np.linspace(0, 1, N)) # access color values from cmap

# # get colors 
# N = 9
# cmap = plt.cm.get_cmap('Set1', N) # plt.cm.viridis(N) #  / 10.)
# colors_set1 = cmap(np.linspace(0, 1, N)) # access color values from cmap

# cmap(1)

my_palette = sns.color_palette("blend:#7AB,#EDA")
magma_cmap = sns.color_palette("magma_r", as_cmap=True)

magma_palette = sns.color_palette('magma',5) # ['#3b0f70', '#8c2981', '#de4968', '#fe9f6d']
magma_palette_r = sns.color_palette('magma_r',5) # ['#3b0f70', '#8c2981', '#de4968', '#fe9f6d']
sns.set_palette(sns.color_palette("magma_r"))
# print('magma palette', magma_palette)


palette = sns.xkcd_palette(['dark blue', 'light blue', 'gold', 'baby blue'])
bblue_rgb = (0.6352941176470588, 0.8117647058823529, 0.996078431372549)
# print(palette)
# palette

# add baby blue to cmap, for no-dmg 
magma_palette.append(bblue_rgb)
# magma_palette.remove((0.135053, 0.068391, 0.315)) # when Nmagma=6
magma_palette.remove((0.171713, 0.067305, 0.370771)) # when N-magma=5
magma_palette

magma_palette_r.insert(0,bblue_rgb)
# print('magma palette_r : ', magma_palette_r.as_hex())
magma_palette_r

my_palette = sns.color_palette("blend:#7AB,#EDA")
my_palette2 = sns.color_palette('crest')
my_palette2


''' --------------
Plot settings
------------------ '''
# plt.rcParams.update({'font.size': 16})
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
fs=14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


''' --------------
Functions
------------------ '''


def parse_cla():
    """
    Command line argument parser

    Accepted arguments:
        Required: either one of the following
        --region (-r)   :   Abbreviation of region to load, as defined in plot_insets_AIS_regions.shp
        --sector (-s)   :   Abbreviation of sector to load, as defined in plot_insets_AIS_sectors.shp
        
        Optional:
        --dtype     :   Specify which damage threshold option to use. 
                        Defaults to ''dmg095'', which is the threshold based on the 95th percentile noise signal value
        --ksize     :   Option to downsample data; specifies downsampling kernel size. Defaults to no downsampling.

    :return args: ArgumentParser return object containing command line arguments
    """

    # Required arguments
    parser = argparse.ArgumentParser()
    ## Either REGION or SECTOR should be defined:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--region","-r",help='Specify region to process',type=str, 
                            choices=('ASE','BSE','AP','DML','IOS','WA','RV','FR','tbd','tbd2') )    
    group.add_argument("--sector","-s",help='Specify sector to process (all= all AIS)',type=str, 
                            choices=('ASE', 'BSE', 'EIS', 'RS', 'WIS', 'WS','all') )    
    

    # Optional arguments
    parser.add_argument("--dtype","-d",help='Specify dmg type to process (default ''dmg095'')',type=str, required=False,
                            choices=('dmg','dmg095') )    
    parser.add_argument("--ksize","-k",help='Spatial downsampling size (default: none)',type=int, required=False )                  

    args = parser.parse_args()
    return args 


import dask
def load_nc_obs_data( path2data, region_ID, roi_type='region', varname=None ):
    ''' ----------------------
    Load data: netCDFs per region, per variable / all variables
    ------------------------- '''

    print('----\n Loading netCDF for {} : {} '.format(roi_type,  region_ID) )

    if not varname and roi_type == 'region': # load all available variables
            
        ''' Load all variables from individual netCDF files '''
        region_filelist_p1 = glob.glob(path2data + '*'+roi_type+'-'+region_ID+'_*part1.nc')
        region_filelist_p2 = glob.glob(path2data + '*'+roi_type+'-'+region_ID+'_*part2.nc')
        region_filelist_p1.sort()     
        region_filelist_p2.sort()              


        region_ds_p1 = xr.open_mfdataset( region_filelist_p1,  
                    combine="by_coords",decode_times=True,
                    data_vars='minimal', 
                    coords= 'minimal', 
                    compat='broadcast_equals', #  all values must be equal when variables are broadcast against each other to ensure common dimensions.
                #   chunks={'y':'auto','x':'auto','time':5}, # add chucnking info for dask
                    chunks={'y':2000,'x':2000,'time':1}, # add chucnking info for dask: multitude of downsampling size
                    )  
        region_ds_p2 = xr.open_mfdataset( region_filelist_p2,  
                    combine="by_coords",decode_times=True,
                    data_vars='minimal', 
                    coords= 'minimal', 
                    compat='broadcast_equals', #  all values must be equal when variables are broadcast against each other to ensure common dimensions.
                #   chunks={'y':'auto','x':'auto','time':5}, # add chucnking info for dask
                    chunks={'y':2000,'x':2000,'time':1}, # add chucnking info for dask: multitude of downsampling size
                    ) 

        ''' --------------------------------------
        Repeat temporally static variable (REMA; basalmelt) to even out dataset dimension
        This drops time=0
        ------------------------------------------ '''
        region_ds_p1 = myf.repeat_static_variable_timeseries( region_ds_p1 , 'rema' )
        region_ds_p2 = myf.repeat_static_variable_timeseries( region_ds_p2 , 'rema' )
        
        ''' Combine part1 and part 2'''
        region_ds = xr.concat([ region_ds_p1, region_ds_p2],dim='time')

    if not varname and roi_type == 'sector': # load all available variables
        print('.. loading sector files, saved as annual files per variable')
        # for varname in ['dmg','vx','vy','rema']:
        ''' Load all files of variable individual netCDF files '''
        region_filelist_var = glob.glob(path2data + '*'+roi_type+'-'+region_ID+'*.nc')     
        ## region_filelist_dmg = glob.glob(path2data+'damage/' + '*'+roi_type+'-'+region_ID+'*.nc')  
        ## region_filelist_vars = glob.glob(path2data+'velocity_rema/' + '*'+roi_type+'-'+region_ID+'*.nc')   
        ## region_filelist_var = region_filelist_dmg+region_filelist_vars
        # print(region_filelist_var)

        region_ds = xr.open_mfdataset( region_filelist_var,  
                    combine="by_coords",decode_times=True,
                    data_vars='minimal', 
                    coords= 'minimal', 
                    compat='broadcast_equals', #  all values must be equal when variables are broadcast against each other to ensure common dimensions.
                #   chunks={'y':'auto','x':'auto','time':5}, # add chucnking info for dask
                    chunks={'y':2000,'x':2000,'time':1}, # add chucnking info for dask: multitude of downsampling size
                    )  

        ''' --------------------------------------
        Repeat temporally static variable (REMA; basalmelt) to even out dataset dimension
        This drops time=0
        ------------------------------------------ '''
        region_ds = myf.repeat_static_variable_timeseries( region_ds , 'rema' )

    else:
        ''' Load selected variable '''
        region_varfile = glob.glob(path2data + '*region-'+region_ID+ '*'+ varname +'*nc')[0]
        region_ds = xr.open_dataset( region_varfile )

    return region_ds 


def preprocess_obs_data(region_ds, ksize=None):
    ''' ------------------------------
    #######
    ####### DATA (PRE)PROCESSING
    #######
    ----------------------------------'''
    varnames = list(region_ds.data_vars)
    if 'dmg' in varnames:
        ''' ----------------
        Add stricter dmg threshold value to dataArray
        --------------------'''
        region_ds['dmg095']= myf.update_stricter_dmg_threshold( region_ds['dmg'] , 0.016 ,reduce_or_mask_D='reduce') # update to no-dmg pct095 threshold tau=0.053
        # #region_ds['dmg099']= myf.update_stricter_dmg_threshold( region_ds['dmg'] , 0.026 ,reduce_or_mask_D='reduce') # update to no-dmg pct099 threshold tau=0.063



    ''' ## Interpolation of small nan-gaps of REMA data
    '''
    dx = int(region_ds.rio.resolution()[0]) # np.unique(region_ds['x'].diff('x').values)[0]
    nmax = 6
    # interpolate first X then Y (cannot do both -- so this skews the values. Maarja een mens moet wat.
    interpol = region_ds.chunk(dict(x=-1)).interpolate_na(dim='x', method='linear', limit=None, use_coordinate=True, max_gap=int(nmax*dx), keep_attrs=None)
    region_ds = interpol.chunk(dict(y=-1)).interpolate_na(dim='y', method='linear', limit=None, use_coordinate=True, max_gap=int(nmax*dx), keep_attrs=None)


    ''' ----------------
    Downsample data, 
    --------------------'''
    # # region_ds_high_res.append(region_ds) # store high resolution dataset to check downsampling..

    if ksize:
        dx = int(region_ds.rio.resolution()[0])
        dy = int(region_ds.rio.resolution()[1])
        if np.abs(dx) != np.abs(dy):
            print("Warning: x and y resolution are not the same; {} and {} -- code update required".format(np.abs(dx), np.abs(dy) ))

        # with dask.config.set(**{'array.slicing.split_large_chunks': True}): # gives error
        with dask.config.set(**{'array.slicing.split_large_chunks': False}): ## accept large chunks; ignore warning
            region_ds = myf.downsample_dataArray_withoutD0(region_ds, ksize=ksize, 
                                        boundary_method='pad',downsample_func='mean', skipna=False)
        new_res = ksize*400
    else:
        new_res=400
        # print('.. resolution {}m downsampled to {}m'.format(dx, new_res))
    
    # Check if grid resolution is regular (otherwise, adjust)
    if np.abs(int(region_ds.rio.resolution()[0]) ) != np.abs( int(region_ds.rio.resolution()[1]) ):
        print( "x and y resolution are not the same; {} and {} -- resample to regular grid of {}m".format(
                    np.abs(int(region_ds.rio.resolution()[0])), 
                    np.abs(int(region_ds.rio.resolution()[1])), new_res ))
        
        grid_dummy = myf.make_regular_grid_for_ds(region_ds, grid_res=new_res)
        region_ds = myf.reproject_match_grid( grid_dummy, region_ds ) 

    ''' ------------
    Calculate velocity and strain for (downsampled) data
    ---------------- '''
    
    # Calculate velocity and strain components
    region_velo_strain, region_variables_dt = myf.calculate_velo_strain_features(region_ds, velocity_names=('vx','vy'), length_scales=['1px'])

    ## Fill the first temporal-difference timestep (2015) with a value (so that this time slice doesnt get dropped later on)
    ## NB: the fill value for dt_2014-15 with a copy of dt_2015-2016, as filling with 0 would create an artificial jump
    da_delta_2015_new = region_variables_dt.sel(time=2016).assign_coords(time=2015) # select 2016 from dataset and assing it as time=2015
    region_variables_dt = xr.concat([da_delta_2015_new, region_variables_dt],dim='time') # add artificial dt-2015 to dataset

    ## add to dataset
    region_ds = xr.merge([region_ds, region_velo_strain, region_variables_dt ]).transpose('time','y','x') # make sure everything is in same order


    return region_ds

def stack_ds_to_df(region_ds, iceshelf_polygon_gpd = iceshelf_poly_meas):
    ''' --------------
    Aggregate values to 1D: ON DATASET
    - Temporaly static variables should already be repeated, to match datashape
    - Clip 2D to ice shelf (TO DO: clip to annual ice shelf) (2D xarray)
    - Reshape to 1D by xarray.stack (spatial information is retained)
    - Remove all pixels that have NaN value in any one of the variables (TO DO: do this per year, to account for changing ice shelf polygon) 
    - Format arrays to a pandas dataFrama
    --------------'''

    ''' Clip to iceshelf  '''
    # iceshelf_polygon_gpd = iceshelf_poly_meas # TO DO: update with annual polys? -- doesnt make sense for these plots, as I use "REMA" "VELOCITY" and "BASALMELT" that only have static mask anyway
    # Set drop=True so that converting to dataframe already has fewer px
    region_ds  = region_ds.rio.clip( iceshelf_polygon_gpd.geometry, iceshelf_polygon_gpd.crs, drop=True, invert=False)

    ''' Aggregate to 1D '''
    with dask.config.set(**{'array.slicing.split_large_chunks': False}): # silence warning (running with True gives error lateron)
        region_ds_1d = region_ds.stack(samples=['x','y']) # (time, samples)
    ## region_ds_1d = region_ds.stack(samples=['x','y']) # (time, samples)
    region_ds_1d = region_ds_1d.dropna(dim='samples',how='all') # drop mask values where all timesteps have NaN

    ''' Convert to dataFrame (reads to memory) '''
    data_pxs_df = region_ds_1d.to_dataframe() # nested dataframe
    
    # Drop spatial ref (has not data acutally) 
    data_pxs_df = data_pxs_df.drop(['spatial_ref'],axis=1)

    # Flatten the nested multi-index to just column values -- automatically generates a 'year' value for every sample
    #   if 'labdata' in homedir: # rename coords otherwise error with reset_index
    #     data_pxs_df = data_pxs_df.rename({'x':'x_coord','y':'y_coord'})

    #   data_pxs_df = data_pxs_df.reset_index(level=['time','x','y']) # 18767504 rows ## Error in pandas 1.4.3
    data_pxs_df = data_pxs_df.droplevel(['x','y']).reset_index('time')
    
    # For now: drop x and y values
    #   data_pxs_df = data_pxs_df.drop(['x','y'],axis=1)

    ''' Drop NaN pixels:
    Pandas drops all rows that contain missing values. 
    - This means that if any variable has a NaN value, that px is dropped.
        --> should make sure to fill NaN values for variables before this step (e.g. dmg NaN is set to 0; filling of REMA gaps, etc..)
    - Since I have rows for px per year, this means that if I would have clipped the data to annual ice shelf polygons, the number of pixels per year can vary.  
    '''
    data_pxs_df.dropna(axis='index',inplace=True) # Drop rows which contain ANY missing values.
    return data_pxs_df

def discretize_dmg( data_pxs_df, dmg_type='dmg095', 
        dmg_bins =np.array([-0.001, 0,  0.0125, 0.0625, 0.1625, 0.3125]), 
        bin_labels= ['no damage', 'low', 'medium','high' ,'very high'] 
        ):

    df_discr = data_pxs_df.copy()

    df_discr['dmg_binned'] , cut_bin1 = pd.cut(x = df_discr[dmg_type], bins = dmg_bins, labels=bin_labels,
                                 include_lowest = True, retbins=True, right=True, # includes lowest bin. NB: values below this bin are dropped
                                 ) # [(-0.001, 0.025] < (0.025, 0.1] < (0.1, 0.225] < (0.225, 0.4]]

    '''## Bin values with qcut: define bin width based on quantiles; Discretizes variable into equal-sized buckets
    ## CALCULATE THIS only for dmg>0 values :) '''
    try:
      df_gt0 = df_discr.loc[df_discr[dmg_type] > 0]
      qbin = [0, .25, .5, .75, 1.]
      qlabels = ['q'+str(q) for q in qbin[:-1]]
      df_discr['dmg_quantiles'], cut_binq = pd.qcut(df_gt0[dmg_type], q = qbin, labels = qlabels, 
                                # include_lowest=True, right=True, 
                                retbins = True)
      cut_binq = np.round(cut_binq,4)
    except ValueError:
      # if quantiles yield multiple quantiles with bin-edge 0, resort to different bins
      for q in np.linspace(0,1,100):
          qval = df_discr[dmg_type].quantile(q)
          if qval>0: # find first quantile that thas value other than 0
              break
      qbin = np.linspace(np.round(q,2), 1, 4) # 4 classes of quantiles
      qbin = np.array( [0]+list(qbin) ) # 5 clasess, including 0th-1st quantile
      qlabels = ['q'+str(q) for q in qbin[:-1]]
      df_discr['dmg_quantiles'], cut_binq = pd.qcut(df_discr[dmg_type], q = qbin, labels = qlabels, retbins = True)

    discr_dict = {'dmg_binned': {'bins':dmg_bins, 'bin_labels':bin_labels},
                  'dmg_quantiles': {'bins':cut_binq, 'bin_labels':qlabels}
                    }
    return df_discr, discr_dict


def plot_box_feature_vs_dmg(df_plot, discr_type, dmg_class_labels,  xvar='rema', minmax=None, xlabel=''):

    data_x = []
    data_class_labels=[]
    for dclass in dmg_class_labels:
        var_class = df_plot[df_plot[discr_type] == dclass][xvar]
        print(f'class count: {dclass}:{len(var_class)}')
        if len(var_class) > 0:
            data_x.append(var_class.values)
            data_class_labels.append(dclass)
    print(dmg_class_labels, len(data_x))
    print(data_class_labels, len(data_class_labels))
    data_x # list of data
    dmg_class_labels = data_class_labels

    ##### SET PALETTE
    magma_palette_r = sns.color_palette('magma_r',len(dmg_class_labels)-1) # ['#3b0f70', '#8c2981', '#de4968', '#fe9f6d']
    bblue_rgb = (0.6352941176470588, 0.8117647058823529, 0.996078431372549)
    magma_palette_r.insert(0,bblue_rgb)

    ##### PLOT

    fig, ax = plt.subplots(figsize=(8, 6))

    ### Boxplot

    bp = ax.boxplot(data_x, patch_artist = True, 
                        vert = False, 
                        widths=0.35, #0.2,
                        flierprops={'markersize':1,'color':(0,0,0),'alpha':0.3},
                        # flierprops={'markersize':0.1,'color':(0,0,0),'alpha':0.1},
                        medianprops={'color':(0,0,0)},
                        # showfliers=False,
                        )
    boxplots_colors = magma_palette_r # ['yellowgreen', 'olivedrab']

    # Change to the desired color and add transparency
    for patch, color in zip(bp['boxes'], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)


    ### Axes organisation
    ax.set_yticklabels(dmg_class_labels,fontsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Damage',fontsize=16)
    if minmax is not None:
        ax.set_xlim(minmax)

    ax.set_facecolor( (1.0, 1.0, 1.0, 1.0) ) # white
    ax.grid('on',color=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 1.0) )

    return fig


def plot_boxviolin_feature_vs_dmg(df_plot, discr_type, dmg_class_labels,  xvar='rema', minmax=None, xlabel=''):

    data_x = []
    data_class_labels=[]
    for dclass in dmg_class_labels:
        var_class = df_plot[df_plot[discr_type] == dclass][xvar]
        print(f'class count: {dclass}:{len(var_class)}')
        if len(var_class) > 0:
            data_x.append(var_class.values)
            data_class_labels.append(dclass)
    print(dmg_class_labels, len(data_x))
    print(data_class_labels, len(data_class_labels))
    data_x # list of data
    dmg_class_labels = data_class_labels

    ##### SET PALETTE
    magma_palette_r = sns.color_palette('magma_r',len(dmg_class_labels)-1) # ['#3b0f70', '#8c2981', '#de4968', '#fe9f6d']
    bblue_rgb = (0.6352941176470588, 0.8117647058823529, 0.996078431372549)
    magma_palette_r.insert(0,bblue_rgb)

    ##### PLOT

    fig, ax = plt.subplots(figsize=(8, 6))

    ### Boxplot

    bp = ax.boxplot(data_x, patch_artist = True, 
                        vert = False, 
                        widths=0.2,
                        flierprops={'markersize':0.1,'color':(0,0,0),'alpha':0.1},
                        medianprops={'color':(0,0,0)},
                        # showfliers=False,
                        )
    boxplots_colors = magma_palette_r # ['yellowgreen', 'olivedrab']

    # Change to the desired color and add transparency
    for patch, color in zip(bp['boxes'], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)


    #### Violinplot 
    bw_m= 0.2 # Bandwidth selection strongly influences the estimate obtained from the KDE (much more so than the actual shape of the kernel).
    pts = 1000 # pretty slow if > 1000, without change in KDE
    # pts = 100 # try small for 
    vp = ax.violinplot(data_x, points=pts,  #500 # number of points to define kernel density over
                        showmeans=False, showextrema=False, showmedians=False, 
                        bw_method=bw_m, 
                        vert=False,
                        widths=0.8)
    violin_colors = magma_palette_r 

    # violin visualisations
    for idx, b in enumerate(vp['bodies']):
        m = np.mean(b.get_paths()[0].vertices[:, 0]) # Get the center of the plot
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
        b.set_color(violin_colors[idx]) # Change to the desired color


    ### Axes organisation
    # ax.set_yticklabels(dmg_labels,fontsize=14)
    # print(ax.get_yticks(), ax.get_yticklabels() )
    ax.set_yticklabels(dmg_class_labels,fontsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Damage',fontsize=16)
    # ax.set_title(region_ID, fontsize=16)
    if minmax is not None:
        ax.set_xlim(minmax)

    # ax.set_facecolor( (0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 1.0) ) # gray color from sns
    ax.set_facecolor( (1.0, 1.0, 1.0, 1.0) ) # white
    ax.grid('on',color=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 1.0) )
    # fig.tight_layout()

    return fig


''' ----------------------

Load data: netCDFs per region, per variable; and plot boxviolin

------------------------- '''


# #dmg_type = 'dmg095'
# #region_ID_list # ['ASE', 'BSE', 'AP', 'DML', 'IOS', 'WA', 'RV', 'FR', 'tbd', 'tbd2']
# #ksize=None



''' --------------------------------------
 SELECT REGION, LOAD DATA
------------------------------------------ '''

def main( region_ID, sector_ID, dmg_type = 'dmg095', ksize=None, process_all_AIS=False):
    print('Region: {}; Sector: {}'.format(region_ID, sector_ID))
    if region_ID is not None:
        region_type = 'region'
        print('Loading REGION ', region_ID)
    elif sector_ID is not None:
        region_type = 'sector'
        region_ID = sector_ID
        path2data = os.path.join(homedir,'Data/NERD/data_predictor/_SECTORS/')
        print('loading SECTOR ', sector_ID)
    if process_all_AIS:
        region_type = 'AIS'
        path2data = os.path.join(homedir,'Data/NERD/data_predictor/_SECTORS/')
        print('loading ALL AIS DATA ')

    ''' ------------------------------
    ## Load data
    ----------------------------------'''
    ## region_ds = load_nc_obs_data( path2data, region_ID, roi_type=region_type, varname=None )
    # # print(region_ds)
    if not process_all_AIS:
        region_ds = load_nc_obs_data( path2data, region_ID, roi_type=region_type, varname=None )
        ''' ------------------------------
        ## DATA (PRE)PROCESSING
        - downsample (if selected)
        - calculate velocity and strain components
        ----------------------------------'''
        
        region_ds = preprocess_obs_data(region_ds,  ksize=ksize)

        ''' ------------------------------
        ## Convert to 1d-dataframe
        - This loads data into memory, so takes a while
        ----------------------------------'''
        print('.. convert to dataframe')

        data_pxs_df = stack_ds_to_df(region_ds, iceshelf_polygon_gpd = iceshelf_poly_meas) # no real need to clip to annual polygons, as rema and velocity masks do not align anyway

    else:
        data_pxs_df_list = []
        for sector_ID in ['ASE', 'BSE', 'EIS', 'RS', 'WIS', 'WS']:
            region_ds = load_nc_obs_data( path2data, sector_ID, roi_type='sector', varname=None )

            ''' ------------------------------
            ## DATA (PRE)PROCESSING
            - downsample (if selected)
            - calculate velocity and strain components
            ----------------------------------'''
            
            region_ds = preprocess_obs_data(region_ds,  ksize=ksize)

            ''' ------------------------------
            ## Convert to 1d-dataframe
            - This loads data into memory, so takes a while
            ----------------------------------'''
            print('.. convert to dataframe')

            data_pxs_df = stack_ds_to_df(region_ds, iceshelf_polygon_gpd = iceshelf_poly_meas) # no real need to clip to annual polygons, as rema and velocity masks do not align anyway

            data_pxs_df_list.append(data_pxs_df)
        data_pxs_df = pd.concat(data_pxs_df_list) # combine dataframes of all regions to one large AIS-wide dataframe
        print('---\nCombined to AIS wide dataframe')

    ''' ------------------------------
    ## Discretize data
    ----------------------------------'''
    print('.. discretizing')
    df_discr, discr_dict = discretize_dmg(data_pxs_df, dmg_type=dmg_type, 
                    dmg_bins =np.array([-0.001, 0,  0.0125, 0.0625, 0.1625, 0.3125]), 
                    bin_labels= ['no damage', 'low', 'medium','high' ,'very high'] )
    # df_discr, discr_dict = discretize_dmg(data_pxs_df, dmg_type=dmg_type, 
    #                 dmg_bins =np.array([-0.001, 0, 0.005, 0.01, 0.02, 0.03, 0.04, 1]), ## last binEdge very high, otherwise out-of-bounds are not included
    #                 bin_labels= ['no damage', 'nihil', 'very low', 'low', 
    #                               'moderate', 'high' ,'very high'] )

    df_discr.head()

    
    ''' ------------------------------
    ## Plot boxviolin and save 
    ----------------------------------'''
    print('.. plotting')

    ### Make absolote of strain/velocity change -- nee deze niet
    # df_discr['deltaV'] = df_discr['deltaV'].abs()
    # df_discr['dEmax_1px'] = df_discr['dEmax_1px'].abs()
    # minmax_dE   = [-0.005,np.max([ df_discr['dEmax_1px'].quantile(0.99), 0.05 ])]
    # minmax_dV   = [-10,np.max([ df_discr['deltaV'].quantile(0.99), 300])]
    ## without absolute:
    minmax_dE   = [np.min([ df_discr['dEmax_1px'].quantile(0.01), -0.05 ]),
                    np.max([ df_discr['dEmax_1px'].quantile(0.99), 0.05 ])]
    minmax_dV   = [np.min([ df_discr['deltaV'].quantile(0.01), -300 ]),
                    np.max([ df_discr['deltaV'].quantile(0.99), 300])]

    minmax_rema = [-5, np.max([df_discr['rema'].quantile(0.99) , 100])] # either 100, or larger
    minmax_velo = [-50,np.max([df_discr['v'].quantile(0.99) , 1000]) ] # either 1000, or larger
    minmax_emax = [-0.05,np.max([df_discr['emax_1px'].quantile(0.99), 0.1 ])]
    minmax_emin = [-0.5,0.1]

    ## strain compontents not as absolutes
    # minmax_eff = [-0.2, 0.2] # ,np.min([df_discr['e_eff_1px'].quantile(0.99), 1 ])]
    # minmax_elon = [-0.2, 0.2]#[-0.05,np.min([df_discr['elon_1px'].quantile(0.99), 1 ])]
    # minmax_etrans = [-0.2, 0.2]#[-0.05,0.05]
    # minmax_eshear = [-0.2, 0.2]#[-0.05,np.min([df_discr['eshear_1px'].quantile(0.99), 1 ])]


    ### Make absolote of strain components
    df_discr['elon_1px'] = df_discr['elon_1px'].abs()
    df_discr['etrans_1px'] = df_discr['etrans_1px'].abs()
    df_discr['eshear_1px'] = df_discr['eshear_1px'].abs()
    # new plots with new minmax for e_eff and absolute values for strain components
    minmax_eff = [-0.01, 0.2] # np.max([df_discr['e_eff_1px'].quantile(0.99), 0.2 ])]
    minmax_elon = [-0.01, 0.2] 
    minmax_etrans = [-0.01, 0.1]
    minmax_eshear = [-0.01, 0.1]

    ## Select variables to plot
    flist       = [ 'rema',   'emax_1px',       'dEmax_1px',     'v',        'deltaV' ,        'e_eff_1px' ,        'elon_1px',         'etrans_1px',       'eshear_1px'] 
    labelname   = [ 'height', 'principal strain','strain change', 'velocity', 'velocity change',  'effective strain' ,'longit. strain', 'transverse strain','shear strain']  
    minmax_list = [minmax_rema, minmax_emax,     minmax_dE,      minmax_velo, minmax_dV, minmax_eff , minmax_elon, minmax_etrans, minmax_eshear ]

    # flist       = [ 'e_eff_1px' ,       'elon_1px',         'etrans_1px',       'eshear_1px'] 
    # labelname   = [ 'effective strain' ,'longit. strain', 'transverse strain','shear strain']  
    # minmax_list = [ minmax_eff ,        minmax_elon,         minmax_etrans,     minmax_eshear ]

    plot_boxviolin=True
    plot_boxOnly = False
    for xvar, minmax, xlabel in zip(flist, minmax_list ,labelname):
        
        
        if ksize is None:
            ksize=0

        if plot_boxviolin:
            ''' Plot for manually binned dmg'''
            # try:
            discr_type = 'dmg_binned'
            dmg_class_labels = discr_dict['dmg_binned']['bin_labels']

            print(discr_dict)

            fig = plot_boxviolin_feature_vs_dmg(df_discr, discr_type , dmg_class_labels,  
                                        xvar=xvar, minmax=minmax, xlabel=xlabel)

            # figname = 'boxviolin_{}-{}_k{}_{}-{}_{}.png'.format( region_type, region_ID, ksize, dmg_type, discr_type.split('_')[1],xvar )
            figname = f'boxviolin_{region_type}-{region_ID}_k{ksize}_{dmg_type}-{discr_type}_{xvar}.png'

                
            if not os.path.isfile( os.path.join( path2savefig, figname ) ):
                ### SAVE FIGURE
                print("Saving to figure {}".format(figname))
                fig.savefig(os.path.join(path2savefig,figname),bbox_inches='tight')
                plt.close()
            else:
                print('.. manually binned figure already exists -- continue')
            # except:
            #     print('.. some error occured for manual bins..?')
            #     pass

            ''' Plot for quantiled binned dmg'''
        
            discr_type = 'dmg_quantiles'
            dmg_class_labels = discr_dict['dmg_quantiles']['bin_labels']
            fig = plot_boxviolin_feature_vs_dmg(df_discr, discr_type , dmg_class_labels,  
                                        xvar=xvar, minmax=minmax, xlabel=xlabel)

            ## Save Figure
            figname = 'boxviolin_{}-{}_k{}_{}-{}_{}.png'.format( region_type, region_ID, ksize, dmg_type, discr_type.split('_')[1],xvar )
            
            ## Check if data exitsts
            if not os.path.isfile( os.path.join( path2savefig, figname ) ):
                print("Saving to figure {}".format(figname))
                fig.savefig(os.path.join(path2savefig,figname),bbox_inches='tight')
                plt.close()
            else:
                print('.. quantile figure file already exists -- continue')

        if plot_boxOnly:
            ''' Plot for manually binned dmg'''
            
            discr_type = 'dmg_binned'
            dmg_class_labels = discr_dict['dmg_binned']['bin_labels']

            print(discr_dict)

            fig = plot_box_feature_vs_dmg(df_discr, discr_type , dmg_class_labels,  
                                        xvar=xvar, minmax=minmax, xlabel=xlabel)

            # figname = 'boxviolin_{}-{}_k{}_{}-{}_{}.png'.format( region_type, region_ID, ksize, dmg_type, discr_type.split('_')[1],xvar )
            figname = f'boxplot_{region_type}-{region_ID}_k{ksize}_{dmg_type}-{discr_type}_{xvar}.png'

                
            if not os.path.isfile( os.path.join( path2savefig, figname ) ):
                ### SAVE FIGURE
                print("Saving to figure {}".format(figname))
                fig.savefig(os.path.join(path2savefig,figname),bbox_inches='tight')
                plt.close()
            else:
                print('.. manually binned figure already exists -- continue')
    print('Done\n---')


if __name__ == '__main__':

    ''' ---------------------------------------------------------------------------
    Command Line configuration
    Run script as "python path/to/script.py --region region_ID --ksize k --dtype dmg095
    Run script as "python path/to/script.py --sector sector_ID --ksize k --dtype dmg095
    -------------------------------------------------------------- '''
    
    # if optional arguments are not specified in command line, they are set to None
    # required arguments will throw a usage error
    args = parse_cla()
    region_ID = args.region
    sector_ID = args.sector
    dmg_type = args.dtype if args.dtype is not None else 'dmg095'
    ksize = args.ksize
    process_all_AIS=False
    if sector_ID == 'all':
        process_all_AIS = True
        sector_ID = None

    # call main function
    print('settings: dmg {}, ksize {}'.format( dmg_type, ksize))

    main(region_ID, sector_ID, dmg_type , ksize, process_all_AIS=process_all_AIS)   
