# RAMP starting kit on the bike counters dataset

![GH Actions](https://github.com/ramp-kits/bike_counters/actions/workflows/main.yml/badge.svg)

## Table of Contents
- [Overview of the project](#overview)
- [Dataset Download](#download)
- [Data Analysis](#eda)
- [Estimator](#estimator)
- [Installing Requirements](#requirements)
- [Lauching the web app](#webapp)
- [To go further](#further)

## Overview of the project

This project aims to study and predict bike traffic in Paris and display the results in a web app.  
The goal is to understand bike traffic patterns to optimize the development of Paris' infrastructure.
The study of historical data provides very useful insights about the distribution and the spread of the datapoints.
The data analysis part is visualized using Streamlit. The prediction part is done using different regressors, such as tree-based regressors and Ridge.

### Dataset Download

The parquet files are already in the data folder. You can also download the train and test datasets by clicking on these two links:
 - [train.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/train.parquet)
 - [test.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/test.parquet)

### Data Analysis

An exploratory data analysis is conducted (analysis of distributions, temporal trends, geographical locations) to extact useful insights from the data.
It is the first step towards understanding the underlying trends in the data and identifying the transforms that need to be applied to the inputs of the predictive models.
For instance, one should check the quasi-compatibility between the input distribution and gaussian based models such as linear regression.

### Estimator

The estimator.py file in the submissions/my_submission/ directory provides details about the preprocessing and model fitting steps of the project.

### Installing Requirements

You can install the dependencies with the following command-line:

```bash
pip install -U -r requirements.txt
```

It is recommended to create a new virtual environment for this project. For instance, with conda,
```bash
conda create -n bikes-ramp python=3.9
conda activate bikes-ramp
pip install -r requirements.txt
```

### Usage
The Dockerfile contains instructions including cloning this repository. After building an image, you can run a container for the image and see the web app.

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)

You can find the description of the columns present in the `external_data.csv`
in `parameter-description-weather-external-data.pdf`. For more information about this
dataset see the [Meteo France
website](https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32)
(in French).
