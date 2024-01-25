import ee
import relorbs
import configparser
import os
import sys
import time # not needed here
import json

# ee.Authenticate()
ee.Initialize()


def main(configFile):
    '''This function exports Sentinel-1 data from the Google Earth Engine (GEE) to the Google Cloud Storage Bucket (GCS).
    It exports individual relative-orbit files.
    A configuration file is used to specify user input, such as:
    - Which Sentinel-1 subset to use (orbit properties, temporal range, optional spatial range)
    - The path of the GCS bucket
    - Options to clip to coastline.
    '''


    ''' ---------------------------------------------------------------------------
            Configuration
    -------------------------------------------------------------- '''

    if configFile is None:
        raise NameError('No config file specified. Run script as "python this_script.py /path/to/config_file.ini"')
    else:
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(os.path.join(configFile))

    path2files = config['PATHS']['path2relorbList']
    my_bucket = config['PATHS']['gcloud_bucket']        # full path:   e.g. ee-export_s1_relorbs/path/to/dir
    bucket_base   = my_bucket.split('/')[0]             # main bucket: e.g. ee-export_s1_relorbs
    bucket_subdir = my_bucket.replace(bucket_base,'').lstrip('/')    # e.g. path/to/dir/
    if bucket_subdir: # if string is not empty, make sure subdir ends with a trailing "/"
        if bucket_subdir[-1] != '/':
            bucket_subdir += '/' # add trailing "/" if not present

    mode = config['DATA']['mode']
    orbitPass = config['DATA']['orbitPass']
    t_strt = config['DATA']['t_strt']
    t_end = config['DATA']['t_end']
    vismin = int(config['DATA']['img_bounds_min'])
    vismax = int(config['DATA']['img_bounds_max'])
    bnds = config['DATA']['bnds'] 
    CRS = config['DATA']['CRS']
    scale = int(config['DATA']['imRes']) # test scale
    clip_coast = True if config['DATA']['clip_coast'] == 'True' else False
    start_export = True if config['DATA']['start_export'] == 'True' else False
    filter_geometry = None if config['DATA']['AOI'] == 'None' else json.loads(config.get("DATA","AOI"))
    try: 
        tileNums=config['DATA']['tileNums']; 
    except KeyError: 
        tileNums = None
    
    if tileNums is not None:
        tileNums = config['DATA']['tileNums']
        tileNums = [int(t) for t in tileNums.split()]

    try:
        relorb_list_fname = config['PATHS']['fileRelorbList']
    except KeyError: # no pre-existing file specified
        relorb_list_fname = None

    # -- Print some settings for information
    AOI=filter_geometry or tileNums
    relorb_list_rw='Read list ' + relorb_list_fname if relorb_list_fname is not None else 'Create & Write list'
    print('Loaded settings: \n \
        AOI:       {}\n \
        clipCoast: {}\n \
        relorbs:   {}\n \
        bucket:    {}\n \
        orbitPass: {}\n \
        '.format(AOI,clip_coast,relorb_list_rw,bucket_subdir,orbitPass))        

    ''' ---------------------------------------------------------------------------
            Select all images
    -------------------------------------------------------------- '''

    # Filter relorbs from S1 collection
    if t_strt is not None:
        fCol_relorbs_list = relorbs.get_S1_relorb_ids_list(t_strt, t_end, 
                                            bnds=bnds, mode=mode, orbit_pass = orbitPass,
                                            filter_tileNums=tileNums ,filterGeom=filter_geometry) # filter_tileNums=None

    # Logging
    if relorb_list_fname is None: # create new file based on parameter settings

        # if ROI defined, change List name
        if tileNums is not None:
            print('Selected {} relorbs for period {} to {}, and tileNums {}'.format(len(fCol_relorbs_list),t_strt,t_end,tileNums))
            relorb_list_fname = 'List_relorbs_S1_'+mode+'_'+bnds+'_'+t_strt+'_'+t_end+'_tileNums'+str(tileNums)
        elif filter_geometry is not None:
            print('Selected {} relorbs for period {} to {}, in the specified AOI'.format(len(fCol_relorbs_list),t_strt,t_end))
            relorb_list_fname = 'List_relorbs_S1_'+mode+'_'+bnds+'_'+t_strt+'_'+t_end+'_AOI-geometry'
        else:
            print('Selected {} relorbs for period {} to {}, for all ice shelves'.format(len(fCol_relorbs_list),t_strt,t_end))
            relorb_list_fname = 'List_relorbs_S1_'+mode+'_'+bnds+'_'+t_strt+'_'+t_end+ '_'+ str(scale) + 'm'

        # -- save list of relorb img names ; update file version if it already exists
        version=1
        relorb_list_fname_check = relorb_list_fname
        while os.path.isfile(os.path.join(path2files,relorb_list_fname_check+'.txt')):
            relorb_list_fname_check = relorb_list_fname + '_append_' + str(version)
            version += 1 
        relorb_list_fname = relorb_list_fname_check  +'.txt' 
            
        with open(os.path.join(path2files,relorb_list_fname  ), 'w') as f:
            for img_id in fCol_relorbs_list:
                # f.write(str(img_id) + '_' + str(scale) +'m\n')
                f.write(str(img_id) +'\n')
            print('.. Written relorbs to {}'.format(relorb_list_fname))        

        
    ''' ---------------------------------------------------------------------------
            Export images to Cloud Bucket
    -------------------------------------------------------------- '''


    with open(os.path.join(path2files,relorb_list_fname), 'r') as f:
        img_id_load = [line.rstrip('\n') for line in f]

        
    im_task_list = []

    print('.. Export {} relorbs to gcloud bucket {} '.format(len(img_id_load), my_bucket))
    for i in range(0,len(img_id_load)):
        # get img
        imName = img_id_load[i] # select single img id from orbits_to_use
        eeImg = ee.Image('COPERNICUS/S1_GRD/' + imName) 
        # eeImg_meta = eeImg.getInfo() # reads metadata
        
        # -- Erode Edges: buffer img geometry (to be a bit smaller)
        export_geom = eeImg.geometry().buffer(-5e3,1e3)
        
        # buffer coastline
        if clip_coast:
            # print('.. Clipping img to coastline')
            coastline_buffer = relorbs.get_coastline_buffer(size=20e3,err=1e3) 
            eeImg = eeImg.clipToCollection(coastline_buffer)
            
        
        # export filename
        file_name = 'relorb_'+ imName + '_' + str(scale) +'m'
        
        im_task = relorbs.export_img_to_GCS(eeImg.select(bnds),
                                            bucket=bucket_base,
                                            file_name= bucket_subdir + file_name,
                                            vismin=vismin,vismax=vismax,scale=scale,CRS=CRS, #NB: vismin and vismax are not actually used (jan/23)
                                            export_geometry=export_geom,
                                            start_task=start_export)
        
        # store img tasks in list to check status later
        im_task_list.append(im_task)
         
    if not start_export:
        print('.. Did not start export tasks; set start_task to True')      
    print('Done')


    ''' ---------------------------------------------------------------------------
            Images are uploading to gcloud bucket.
            This might take a while.
    -------------------------------------------------------------- '''

if __name__ == '__main__':
    #  Run script as "python path/to/script.py /path/to/config_file.ini"
        
    # retrieve config filename from command line
    config = sys.argv[1] if len(sys.argv) > 1 else None

    # run script
    main(config)   