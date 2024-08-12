
import os
import geopandas as gpd
import numpy as np
import glob
import pandas as pd 


saving=True


''' --------------
Paths
------------------ '''

homedir = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/'

# path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/')
# path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/damage095_lowerbound/')
# path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/damage095_upperbound/')
# path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/stricter-d099/')
# path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/stricter-d095/')
# path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/stricter-pct25/')
path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/stricter-pct05/')
# path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/damage095_tmp/')

## try to get back original values
path2data = os.path.join(homedir, 'Data/NERD/dmg095_nc/aggregated/_aggregated_with_nodataMask_any/_perSectorARCHIVE/')
path2data = os.path.join(homedir, 'Data/NERD/data_organise/v0_forPlots/reproduced_newscript/')
path2data = os.path.join(homedir, 'Data/NERD/data_organise/v0_forPlots/reproduced_oldscript/') ## jeej goeie
path2data = os.path.join(homedir, 'Data/NERD/data_organise/add_2000/reproduced_oldscript/') ## also 2000
path2data = os.path.join(homedir, 'Data/NERD/data_organise/stricter-pct05/')  ## uncertainty range


''' --------------
Get Shapefiles 
------------------ '''

## redefined: SECTORS for AIS
sector_path = os.path.join(homedir, 'QGis/data_NeRD/AIS_outline_sectors.shp')
sector_poly = gpd.read_file(sector_path)
sector_ID_list = sector_poly['sector_ID'].to_list()

sector_ID_list
sector_IDs= ['ASE', 'BSE', 'EIS', 'RS', 'WIS-a','WIS-b', 'WS'] ## for v0 dmg095; for stricter_uncertainty 
# sector_IDs= ['ASE', 'BSE', 'EIS', 'RS', 'WIS-a','WIS-b', 'WS-a','WS-b'] ## for uncertainty dmg upper-lowerbnds + for new main dmg095

''' --------------
### SELECT SUBDIR
------------------ '''


## -- ANNUAL S1/RAMP: 1000m:
subdir = '_aggregated_with_nodataMask_any/' # strict mask; also done for lowres data 97/21 
# subdir = ''
fname_suffix = '_1000m.shp'
fname_suffix = 'downsampled.shp'
years_list = np.concatenate([np.array([1997]), np.arange(2015,2022)])
years_list = np.concatenate([np.array([1997,2000]), np.arange(2015,2022)])
# years_list = [2000] ## reprocess only single year


''' Update path '''
# path2agg = os.path.join(path2data, subdir, '_perSector/')
# path2save = os.path.join(path2data, subdir )
path2agg = path2data
path2save = path2data
print('Saving to: ', path2save)

# print(path2agg)

## Load file list
df_file_list_all = glob.glob( os.path.join(path2agg, 'aggregated_dmg_per_iceshelf*.shp'))
df_file_list_all.sort()

## select resolution

df_file_list =  [file for file in df_file_list_all if fname_suffix.split('.')[0] in file] 
# print(df_file_list)


''' ---------
### Process
------------ '''


for year in years_list:

    print("Processing {}".format(year))
    iceshelves_df_list=[]
    
    for region_ID in sector_IDs: #  region_ID_list:

        ## Select files from region, for specific year
        df_files = [file for file in df_file_list if region_ID in file]
        df_filename = [file for file in df_files if str(year) in os.path.basename(file)]
        if not df_filename:
            raise  ValueError(f'Found no file {region_ID}-{year}', df_files)
        if len(df_filename) > 1:
            raise ValueError('Found > 1 matches: ', df_filename)
        df_filename = df_filename[0]

        ## read annual data
        df_data = gpd.read_file(os.path.join(path2agg,df_filename))

        ## Store regional data
        iceshelves_df_list.append(df_data)

    '''
    ## Concatenate dataframes
    '''
    df_year_AIS = pd.concat(iceshelves_df_list)
    try:
        df_year_AIS = df_year_AIS.drop(['region_ID','regionNAME','Regions'],axis=1)
    except:
        pass
    
    try:
        df_year_AIS= df_year_AIS.drop(['sector_ID','sectorNAME','x_label','y_label','Regions'],axis=1) # sectorID and sectorNAME etc are all 'none'
        # df_year_AIS= df_year_AIS.drop(['x_label','y_label','Regions'],axis=1)
    except:
        pass
    # df_data = gpd.sjoin(sector_gpd,df_data, how='right').drop(['index_left'],axis=1)

    ### Save to shapefile
    df_filename = 'aggregated_dmg_per_iceshelf_AIS_' + str(year) + fname_suffix
    if saving:
        print('..Saving to: ', df_filename)
        df_year_AIS.to_file(os.path.join(path2save, df_filename), index=False,crs='EPSG:3031')
    else:
        print('.. did not save ',df_filename)


