# antarctic-damage-change
Created by Maaike Izeboud (m.izeboud@tudelft.nl | @izeboudmaaike )

<span style="color: red;">
This README doc is under construction; more info to be added
</span>

## Intro
This repository provides code accompanying the paper "Antarctic ice shelves vulnerable to damage in future climate warming" by M. Izeboud, S. Lhermitte, S. de Roda Husman, B. Wouters in review at Nature Climate Change (2024).

## About
In this project, we
1. Detect damage on Antarctic ice shelves from SAR remote sensing imagery, using the Normalised Radon Transform Detection method (NeRD; see Izeboud and Lhermitte (2023), Remote Sensing of Environment, https://doi.org/10.1016/j.rse.2022.113359).
2. Damage maps are created for the years 1997, 2000, 2015, 2016, 2017, 2018, 2019, 2020, 2021; obtained from SAR data from the Radarsat mission (1997 and 2000) and Sentinel-1 observations (2015-2021).
3. An estimated relationship between damage and ice dynamical parameters is constructed using a Random Forest Regression. This relationship is used to gain insights in the effects of changing conditions on the evolution of damage, providing insights in effects of potential future climate scenarios on Antarctic ice shelf stability.

## Structure of the repo
This repository provides code to replicate results in the manuscript.
For the full code of the damage detection method, NeRD, see [github.com/mizeboud/NormalisedRadonTransform](https://github.com/mizeboud/NormalisedRadonTransform).

- config_files: files with settings used in scripts
- data_demo: this folder contains example data to run a few of the scripts/notebooks
- files: this folder contains necessary files to run the included example
- notebooks: example notebooks of generating figures/plots of the data
- scripts: processing of data, training random forest regression model


## How to cite
A zotero DOI will be included upon publication of the paper.

## More info

The Normalised Radon Transform Damage (NeRD) detection method, used to create the damage maps, can be found at  [github.com/mizeboud/NormalisedRadonTransform](https://github.com/mizeboud/NormalisedRadonTransform)

Supplementary Material for the paper can be found at the 4TU repository with DOI: 10.4121/911e8799-f0dc-42e3-82b4-766ad680a71e

Annual damage maps are published at the 4TU repository with DOI: 10.4121/70f914ee-b20d-4682-b2ec-54eddcc8569d
