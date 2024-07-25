
import sys
import os
import ee 
import relorbs
from datetime import date
import argparse 

ee.Initialize()

filepath = '/Users/tud500158/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - TUD500158/PhD/CrevasseDetection/NERD_AIS/logs/'

# start_export = False ## now implemented as dry-run cli


# damage thresholds for pct=0.95 (noDMG signal)
tau_S1_40_10 = 0.053 # was 0.037 for mean(noDMG); 0.053 is pct95(noDMG)
tau_S1_40_25 = 0.041 # was 0.030 for mean(noDMG); 0.041 is pct95(noDMG)
tau_S1_100_10 = 0.038 # was 0.027 for mean(noDMG); 0.038 is pct95(noDMG)



def remove_relorb_bounds(image): #(erode edges)
    # Get the 'relorb_id' property from the image
    match_relorb = image.get('relorb_id')
    
    # Create an Earth Engine ImageCollection filtering by 'system:index'
    s1_relorb = (ee.ImageCollection('COPERNICUS/S1_GRD')
                 .filter(ee.Filter.eq("system:index", match_relorb)))

    # Get the geometry of the matched S1 image and apply the buffer 
    relorb_buffer = s1_relorb.geometry().buffer(-5e3, 1e3)

    # Clip the image to the buffered geometry
    clipped_image = image.clip(relorb_buffer)

    return clipped_image


def parse_cla():
    """
    Command line argument parser

    Accepted arguments:
        Required:
        --asset (-a)        : Specific asset to use. Mutually exclusive with 'variable'.
        
        Optional:

    :return args: ArgumentParser return object containing command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset","-a", help="Specify the path of the asset to export, in Google Earth Engine",type=str, required=True)
    
    # Optional argument
    parser.add_argument("--year", "-y", help="Specify which year to export data. Needed when exporting an asset that is an ImageCollection", type=str, required=False,default=None)
    parser.add_argument("--season","-s", help="Specify season to export: SON or DJF. Defaults to SON.",type=str,choices=('DJF','SON'),default='SON',required=False)
    parser.add_argument('--bucket','-b', help='Specify subdir in main bucket', type=str, required=False)
    parser.add_argument('--scale','-sc', help='Specify export resolution', 
                        type=int, required=False, default=None)

    parser.add_argument('--dry-run', help='Activate a dry-run test: loading of data etc, without starting export', 
                         required=False, action='store_true' ) # type=str,choices=('True','False'),default='False')

    args = parser.parse_args()
    return args 



# def main(year_to_export,season_to_export,variable_to_export):
def main():

    ''' ---------------------------------------------------------------------------
        Configuration
    -------------------------------------------------------------- '''
    # if arguments are not specified in command line, they are set to None
    args = parse_cla()
    asset_name = args.asset
    if asset_name is not None:
        print('Reading asset: ', asset_name)

    season_to_export = args.season
    year_to_export = args.year
    bucket_subdir=args.bucket
    scale = args.scale
    dry_run = args.dry_run 
    start_export = not dry_run 
    if dry_run:
        print('Started dry-run. Not exporting any files.')

    # Load tile libraries, and variable or assets
    gridTiles_iceShelves = ee.FeatureCollection('projects/ee-izeboudmaaike/assets/gridTiles_iceShelves')
    

    ''' ---------------------------------------------------------------------------
    (1)  Get the earth engine image that should be exported
    -------------------------------------------------------------- '''

    if ee.data.getAsset(asset_name)['type']:          # Call 'print' to activate lazy EE computing and test if asset exists
            print( '.. loading of asset succesful') 
    else:
        print('.. Asset doesnt exists')
        raise ValueError('Cant find asset ', asset_name)
        

    if 'dmg' in os.path.basename(asset_name): 
        ## -- load the image collection
        dmgCol = ee.ImageCollection(asset_name) 

        if year_to_export is None:
            raise ValueError('When exporting dmg maps, specify which year to export (--year) in CLI')

        # select season:
        if season_to_export == 'SON':
            start_date = year_to_export + '-09-01'
            end_date = year_to_export + '-11-30'
            print('Selecting dmg for {} to {}'.format(start_date,end_date))  
        elif season_to_export == 'DJF':
            start_date = year_to_export + '-12-01'
            end_date = str(int(year_to_export)+1) + '-03-01'
            print('Selecting dmg for {} to {}'.format(start_date,end_date))  

        # -- define image export location and scale
        bucket_subdir = f'dmg_tiled/{year_to_export}-{season_to_export}/'
        # -- define image base name
        image_name_suffix = f'dmg095_{year_to_export}-{season_to_export}'   ## used for dmg with 95pct threshold


        if 'S1' in os.path.basename(asset_name):
            scale=400
            dmgCol = dmgCol.map(remove_relorb_bounds) # ensure that artifacts at image bounds are removed

        elif 'RAMP' in os.path.basename(asset_name):
            scale=1000
            # # -- define image export location and scale
            # if int(year_to_export) == 2000:
            #     bucket_subdir = 'RAMP/MAMM_tiled/dmg095/'
            #     print('.. set bucket subdir to ', bucket_subdir)

        # -- annual dmg median (imCol to img)
        dmg_selected_year = dmgCol.filterDate(start_date, end_date).median() 
        eeImg = dmg_selected_year.select('dmg')

        

    # elif asset_name is not None: #
    else: # other assets than dmg , e.g.  users/izeboudmaaike/ANT_G0240_0000_vx
        ## NB: expect an image, not an image Collection

        # -- define image and image name
        image_name_suffix =  asset_name.split('/')[-1] # get name of asset from path  (ANT_G0240_0000)

        if 'ANT_' in image_name_suffix: # users/izeboudmaaike/ANT_G0240_0000_vx
            # -- define image export location and scale
            bucket_subdir = 'ITS_LIVE_tiled/' + asset_name.split('/')[-1].replace('_vx','').replace('_vy','') + '/'

            # -- ITS LIVE: v1 files (2015-18) are 240m resolution, the v02 files from 2019-onwards are 120m res. Built check here
            if '240' in image_name_suffix:
                scale=240
            elif '120' in image_name_suffix:
                scale=120
            else:         # Cannot resolve resolution from filename
                scale=400 # Use same resolution as dmg maps
        elif 'nodata' in image_name_suffix: # ../../nodata-S1_SON-2016
            mask_year = image_name_suffix.split('_')[-1] # get SON-2016 from asset name
            # -- define image export location and scale
            bucket_subdir = 'S1_tiled/masks/' + mask_year + '/'  
            scale = 400 # same resolution as dmg maps
        else:
            # expect input from terminal
            if bucket_subdir is None:
                raise ValueError('Provide export bucket (--bucket) in CLI')
            # print('.. Specified export {}m to bucket {}: '.format( scale, bucket_subdir))


        eeImg = ee.Image(asset_name) 

    ''' ---------------------------------------------------------------------------
    (2)        Export images to Cloud Bucket
    -------------------------------------------------------------- '''


    bucket_base = 'ee-export_s1_relorbs'
    CRS='EPSG:3031'

    # // Read tileNumbers in list
    tileNums = gridTiles_iceShelves.aggregate_array('tileNumber').getInfo() # list not sorted
    
    print('.. scale: ', scale)

    for i in range(0,len(tileNums)):

        # get tile from list
        tileN = i; 
        tile_geom = gridTiles_iceShelves.filter(ee.Filter.eq('tileNumber',tileN)).geometry()
        
        # -- Append image name with tile number
        image_name = image_name_suffix + '_tile_' + str(tileN)
        
        # export image
        if i == 0: 
            print('Exporting to bucket {}'.format(bucket_base+'/'+bucket_subdir))
            print('.. example filename: {}'.format(image_name))

        im_task = relorbs.export_img_to_GCS(eeImg,
                                            bucket=bucket_base,
                                            file_name= bucket_subdir + image_name,
                                            scale=scale,CRS=CRS,
                                            export_geometry=tile_geom,
                                            start_task=start_export)
        
    if not start_export:
        print('.. Did not start export tasks; set start_task to True')      
 
    print('.. Done')   

if __name__ == '__main__':
    #  Run script as "python path/to/script.py --year year_to_export --variable variable_to_export --season season_to_export"
    #  OR
    #  Run script as "python path/to/script.py --year year_to_export --asset asset_name"
    #  Run script as "python path/to/script.py --year year_to_export --asset asset_name --bucket bucket_subdir --scale 400"

    # # run script
    main()