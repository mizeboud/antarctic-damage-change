# antarctic-damage-change
Created by Dr. Maaike Izeboud (maaike.izeboud@vub.be)

# Intro
This repository provides code accompanying the publication "Damage development on Antarctic ice shelves sensitive to climate warming" by M. Izeboud, S. Lhermitte, S. de Roda Husman, B. Wouters, in review at Nature Climate Change (2024).

# About the project
In this project, we
1. Detect damage on Antarctic ice shelves from SAR remote sensing imagery, using the Normalised Radon Transform Detection method (NeRD; see Izeboud and Lhermitte (2023), Remote Sensing of Environment: https://doi.org/10.1016/j.rse.2022.113359).
2. Damage maps are created for the years 1997, 2000, 2015, 2016, 2017, 2018, 2019, 2020, 2021; obtained from SAR data from the Radarsat mission (1997 and 2000) and Sentinel-1 observations (2015-2021).
3. An estimated relationship between damage and ice dynamical parameters is constructed using a Random Forest Regression (machine learning). This relationship is used to gain insights in the sensitivity of damage development to changing climate conditions until 2100, using data from multiple ice sheet models to represent ice dynamics under various potential future climate scenarios in Antarctica.

# About this repository
This repository provides code relevant to the referenced manuscript. This includes code to reproduce the Random Forest (RF) model development and reproduce figures in the manuscript, but does not include code on the damage detection that has been done. The full code for the used damage detection method, NeRD, is available at [github.com/mizeboud/NormalisedRadonTransform](https://github.com/mizeboud/NormalisedRadonTransform).

- `config_files`: files with settings used in scripts
- `data_demo`: this folder contains some example data relevant to the scripts/notebooks, to give an idea of what it looks like. However, the full datasets are stored at the 4TU Research Data Repository (links added below). The scripts and notebooks in this repository cannot be run without the full datasets.
- `files`: this folder contains necessary files to run code. Including used noise threshold for processing of detected damage, and the trained Random Forest model. 
- `notebooks`: example notebooks of generating figures/plots of the data.
- `scripts`: processing of data, training random forest regression model


# Installation
The code has been developed on ``python version 3.9``. We've experienced version issues when using python 3.10, and higher is untested.

All package requirements are included in the file ``requirements_pipFreeze.txt`` and can be installed with ``pip install -r requirements_pipFreeze.text``. A shorter overview of relevant packages is included in ``requirements.txt``. We've sporadically encountered issues with ``xarray`` and ``geopandas`` versions. In our experience it is necessary to explicitly install ``xarray`` IO backend with ``python -m pip install "xarray[complete]"`` and, with ``geopandas`` version <0.14.4 to pin the ``Fiona`` version to 1.9.6.

# General workflow and code descriptions

### Data file structures
In this study a lot of data was processed on a large spatial domain (the Antarctic ice shelves). For processing speed and computer memory issues and/or file size handling, all data has been stored or processed in smaller (spatial) batches. There were two main approaches used. (1) the Antarctic ice shelves were covered by a grid of 313 square tiles (EPSG:3031) to structure and process the detected damage data. (2) all data used for the RF development and application (detected damage, ice sheet model data) was processed per sector.
![alt text](./gridTiles_sectors.png?raw=true)

### 1. Selecting satellite SAR observations and generating damage maps
1. ``gee_export_S1_relorbs.py`` and ``relorbs.py`` were used to filter and export Sentinel-1 and RAMP satellite observations from the Google Earth Engine (GEE) to a project drive/ google cloud bucket. Filter settings are defined in a config file, ``config_GEE_export_S1_template.ini``
2. The satellite images were processed with NeRD to create damage maps (settings defined in ``config_NERD_image_40m_10px.ini``), which were uploaded to GEE Assets and from there downloaded per tile.
* Sensor bias between RAMP and Sentinel-1 detected damage is calculated in ``calculate_bias_RAMP-S1.ipynb``.

#### 1a. Plotting damage maps
* Damage maps and associated figures were created with QGis; **Manuscript Figure 1a-h, Figure 4a, Extended Data Figure 1, 2, 3**.


### 2. Aggregating data
1. ``data_assemble_tiles_to_netcdf.py`` assembles all data-tiles (.tif) to NetCDF files per sector.
2. ``data_aggregate_dmg_per_ishelf.py`` calculates aggregated values per ice shelf, taking annual masking of no-data areas and updated annual ice front lines into consideration. The aggregation discretizes the continues damage signal into low, medium and high values as well as binary damage/no-damage. These discretized values are used for quantifications and plotting.

##### 2a. Analysing and Plotting aggregated data
* ``plot_boxplot_observations2dmg_sectors.py`` plots the observational parameters against the discretized damage classes (bar plots). **Extended Data Figure 4**
* ``plot_aggregate_dmg_iceshelves_piechart.ipynb`` plot the piecharts in **Manuscript Figure 1i and Extended Data Figure 2**
* ``plot_aggregate_dmg_iceshelves_timeseries.ipynb`` plots the timeseries of **Manuscript Figure 2**

### 3. Random Forest model development and aggregation
1. ```RFmodel_train_dmg_predictor.py``` loads all observational input data (horizontal ice velocity components, surface elevation, detected damage, no-data masks) of all sectors and performs the RF training (settings and hyperparameters in ``RF_gridSearch.ini``). This was done on a computing cluster.
2. ``RFmodel_apply_predictor_[X].py`` loads all ice sheet model data and puts that into the trained RF model to generate damage predictions. Separate files for predicting damage based on the observational data and based on ISMIP-6 project ice sheet model data (using config ``config_RF_predict_ismip.ini``).

##### 3a. Analysing and Plotting RF results
* ``analyse_RF_performance_train-testdata.ipynb``: analyse and visualise the perofrmance of the trained model. **Extended Data Figure 6**.
* ``analyse_RF_param_correlation.ipynb``: analyse the relationship of RF model with synthetic data, **Extended Data Figure 5**.
* ``analyse_ismip_input.ipynb``: analyse ISMIP6 parameters used as input for RF to generate damage projections. **Manuscript Figure 4 d-i and Extended Data Figure 9**
* ``analyse_ismip_predicted.ipynb``: analyse the predicted damage, both timeseries and spatial variations. **Manuscript Figure 4 a-c and Extended Data Figures 7 and 8**


## How to cite
A DOI will be included upon publication of the paper.

## More info

- The Normalised Radon Transform Damage (NeRD) detection method, used to create the damage maps, can be found at  [github.com/mizeboud/NormalisedRadonTransform](https://github.com/mizeboud/NormalisedRadonTransform)
- Supplementary Material for the paper can be found at the 4TU repository with DOI: 10.4121/911e8799-f0dc-42e3-82b4-766ad680a71e
- Annual damage maps are published at the 4TU repository with DOI: 10.4121/70f914ee-b20d-4682-b2ec-54eddcc8569d or viewed in Google Earth Engine (GEE): https://code.earthengine.google.com/b7438668b59e075b21347a9ae8c142e1
