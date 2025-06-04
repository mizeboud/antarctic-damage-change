# antarctic-damage-change
Created by Dr. Maaike Izeboud (maaike.izeboud@vub.be)

## Intro
This repository provides code accompanying the publication "Damage development on Antarctic ice shelves sensitive to climate warming" by M. Izeboud, S. Lhermitte, S. de Roda Husman, B. Wouters, in review at Nature Climate Change (2024).

## About the project
In this project, we
1. Detect damage on Antarctic ice shelves from SAR remote sensing imagery, using the Normalised Radon Transform Detection method (NeRD; see Izeboud and Lhermitte (2023), Remote Sensing of Environment: https://doi.org/10.1016/j.rse.2022.113359).
2. Damage maps are created for the years 1997, 2000, 2015, 2016, 2017, 2018, 2019, 2020, 2021; obtained from SAR data from the Radarsat mission (1997 and 2000) and Sentinel-1 observations (2015-2021).
3. An estimated relationship between damage and ice dynamical parameters is constructed using a Random Forest Regression (machine learning). This relationship is used to gain insights in the sensitivity of damage development to changing climate conditions until 2100, using data from multiple ice sheet models to represent ice dynamics under various potential future climate scenarios in Antarctica.

## About this repository
This repository provides code relevant to the referenced manuscript. This includes code to reproduce the Random Forest (RF) model development and reproduce figures in the manuscript, but does not include code on the damage detection that has been done. The full code for the used damage detection method, NeRD, is available at [github.com/mizeboud/NormalisedRadonTransform](https://github.com/mizeboud/NormalisedRadonTransform).

- config_files: files with settings used in scripts
- data_demo: this folder contains example data to run a few of the scripts/notebooks
- files: this folder contains necessary files to run the included example
- notebooks: example notebooks of generating figures/plots of the data
- scripts: processing of data, training random forest regression model

This repository contains a small example dataset to run the scripts related to the Random Forest model, but the full datasets are stored at the 4TU Research Data Repository (links added below).

## Installation
The code has been developed on ``python version 3.9``. We've experienced version issues when using python 3.10, and higher is untested.

All package requirements are included in the file ``requirements_pipFreeze.txt`` and can be installed with ``pip install -r requirements_pipFreeze.text``. A shorter overview of relevant packages is included in ``requirements.txt``. We've sporadically encountered issues with ``xarray`` and ``geopandas`` versions. In our experience it is necessary to explicitly install ``xarray`` IO backend with ``python -m pip install "xarray[complete]"`` and, with ``geopandas`` version <0.14.4 to pin the ``Fiona`` version to 1.9.6.

## General workflow
In this study a lot of data was processed on a large spatial domain (the Antarctic ice shelves). For processing speed and computer memory issues, all data has been processed in smaller (spatial) batches. There were two main approaches used. (1) the damage detection was structured 
![alt text](https://github.com/mizeboud/antarctic-damage-change/gridTiles.png)

## Codefile descriptions

#### Notebooks
_to do_

#### Scripts
_to do_

## How to cite
A DOI will be included upon publication of the paper.

## More info

- The Normalised Radon Transform Damage (NeRD) detection method, used to create the damage maps, can be found at  [github.com/mizeboud/NormalisedRadonTransform](https://github.com/mizeboud/NormalisedRadonTransform)
- Supplementary Material for the paper can be found at the 4TU repository with DOI: 10.4121/911e8799-f0dc-42e3-82b4-766ad680a71e
- Annual damage maps are published at the 4TU repository with DOI: 10.4121/70f914ee-b20d-4682-b2ec-54eddcc8569d or viewed in Google Earth Engine (GEE): https://code.earthengine.google.com/b7438668b59e075b21347a9ae8c142e1
