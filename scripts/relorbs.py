# Functions that are needed to download GEE images to local

import ee
import time


# ee.Authenticate()
ee.Initialize()


# # Load libraries and assets
gridTiles_iceShelves = ee.FeatureCollection('projects/ee-izeboudmaaike/assets/gridTiles_iceShelves')
iceShelves = ee.FeatureCollection('users/izeboudmaaike/ne_10m_antarctic_ice_shelves_polys')


''' ---------- FUNCTIONS ---------- '''


def select_tile_number(tileNum):
    tile = gridTiles_iceShelves.filter(ee.Filter.eq('tileNumber', tileNum)).first()
    return ee.Feature(tile)


def image_add_time_band(image):
    return image.addBands(
        image.metadata('system:time_start').divide(1000 * 60 * 60 * 24 * 365)
        ).unmask()
      # Convert milliseconds from epoch to years to aid in
      # interpretation of the following trend calculation.


## DEV:
def add_relorb_slicenum(image):
    relorb_slice = ee.Number(ee.Image(image).get('relativeOrbitNumber_start')).format('%.0f').cat('_').cat(ee.Number(ee.Image(image).get('sliceNumber')).format('%.0f'))
    return image.set('relorb_slice', relorb_slice)

def get_coastline_buffer(size=20e3,err=1e3):
    def buffer_feature(feature):
        return feature.buffer(size,err)
    
    coastline=ee.FeatureCollection('users/izeboudmaaike/MEaSUREs_AIS_coastline');
    # coastline_buffer = coastline.geometry().buffer(buffer_size,buffer_maxE) # returns ee.Geometry (slow!)
    coastline_buffer = coastline.map(buffer_feature) # returns ee.featureCollection
    return coastline_buffer

def get_S1_relorb_ids_list(t_strt, t_end, bnds='HH', mode='EW',orbit_pass=None, filter_tileNums=None, filterGeom=None ):

    # image collction on all ice shelvess
    imCol = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(t_strt,  t_end)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', bnds))
        .filterMetadata('instrumentMode', 'equals', mode)
        .filterBounds(iceShelves)
        .map(image_add_time_band)
        .map(add_relorb_slicenum)
        )
    
    # select specific orbit pass (update 25/11/22; previously not included)
    if orbit_pass is not None:
        imCol = imCol.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass) )

    # additional filter to subset data
    if filter_tileNums is not None:
        gridTiles_filter = ee.FeatureCollection(ee.List(filter_tileNums).map(select_tile_number)) # list of tiles
        imCol = imCol.filterBounds(gridTiles_filter)
        
    # geometry filter to subset data
    if filterGeom is not None:
        # imCol = imCol.filterBounds(ee.Geometry.Polygon(filterGeom))   
        imCol = imCol.filterBounds(ee.Geometry.MultiPolygon(filterGeom))    

    # all_relorbs =  imCol.aggregate_array('relativeOrbitNumber_start').distinct() # only gets list of relorb_start numbers (e.g '134, 94, 122' etc)

    # Use disstinct() on imCol to get unique relorbs
    # fCol_relorbs = imCol.distinct('relativeOrbitNumber_start') # featCol with imgs names as list
    # fCol_relorbs_list = fCol_relorbs.aggregate_array('system:index').getInfo() # featCol to list

    # UPDATE (25/11/22): use both relativeOrbitNumber and SliceNumber to select images
    fCol_relorbs = imCol.distinct('relorb_slice') # featCol with imgs names as list
    fCol_relorbs_list = fCol_relorbs.aggregate_array('system:index').getInfo() # featCol to list
    
    
    return fCol_relorbs_list



# Check GEE export task status
def status_task_list(task_list):

    for task in task_list: # start inactive tasks
        if task.status()["state"] == "UNSUBMITTED":
            task.start()
        
        
    task_activity = [task.active() for task in task_list]

    start_time = time.time()
    while any(task_activity):
        print('..checking activity: {}/{} active uploads'.format(sum(task_activity),len(task_list)))
        task_activity = [task.active() for task in task_list]
        time.sleep(60) # sleep 60sec
    
    duration=time.time() - start_time
    print('..All tasks completed after {:.0f}sec ({:.1f}min)'.format(duration,duration/60) )
    
    # check for errros after completion
    for task in task_list:
        if task.status()["state"]=="FAILED":
            print("TASK FAILED: {}".format(task.status()["error_message"]))
            print(task.status())
            print("TASK DESCRIPTION: {}".format(task.status()['description']))
        # else:
            # print(task.status()["state"])

def erode_relorb_bounds(eeImg):
    '''
    Function that applies a negative buffer to the img extent, to remove boundary edges containing 0-data rather than nan-data
    '''    
    extent_buffer = ee.Feature(ee.Image(eeImg).geometry().buffer(-5e3,1e3))
    return eeImg.clip(extent_buffer)
       
        
def export_img_to_GCS(eeImg,bucket,file_name,
                      scale,CRS='EPSG:3031',
                      vismin=None,vismax=None,
                      export_geometry=None,
                      start_task=True):
        
        
    if export_geometry is None:
        export_geometry = eeImg.geometry()
        # eeImg = eeImg.clip(export_geometry)
    
        
    # -- ERODE EDGE of relorb [this is done before inputting the img in this fucntion]
    # eeImg = erode_relorb_bounds(eeImg)

    # -- Export the image to Cloud Storage.
    
   
    task_im = ee.batch.Export.image.toCloudStorage(
            image = eeImg,# do not export toByte; then NaN mask will be converted to 0
            description = 'export_'+file_name.split('/')[-1],
            fileNamePrefix = file_name,
            scale= scale,
            bucket= bucket,
            crs=CRS,
            maxPixels=1e10,
            region=export_geometry
    )

    # -- start tasks at GEE editor
    if start_task:
        task_im.start()
        # print('Started Export of {} to bucket {}'.format(file_name,bucket) )
        # print('..Uploading {}'.format(file_name))
        # status(task_im)
    # else:
    #     print('Warning: Export Task not started (set start_task to True).')
              
        
    return task_im
