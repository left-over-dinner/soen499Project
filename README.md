# SOEN 499 Project

## Abstract

Our project aims to study the dataset provided by the Montreal bike share company Bixi. In this analysis, we plan to look at the trip histories of commuters from 2014 to 2019 to determine the top starting and ending stations in the city. More specifically, we can find out where and when should be the focus of Bixi's deployments by observing emergent patterns over time (e.g. different zones of high activity at different times of the week or year). Additionally, the analysis of the Bixi dataset will also be used to predict where a rider is likely to end their trip given a starting station, time of day and/or time of year.

<br>

## Introduction

As the growing number of inhabitants have started to put a strain on the transporation networks of major cities, many have turned to alternative modes of transportation. This includes popular offerings such as dockless e-scooters and bike sharing programs like Bixi in Montreal. One issue still remains however: matching the demand for bikes with availability. Companies like Bixi need a way to optimize the deployment of their vehicles by determining where and when is best for them to do so. With this, not only can the company increase their revenue, but users will also benefit from the more consistent availability, and thus alleviate the strain on the other transportation networks. This is exactly what this project seeks to do.

Two objectives are set out for this project. The first is to determine the most favourable starting and ending stations in the city depending on the time of day, day of the week, and time of the year. Such an analysis will reveal where the major focus of deployments should be. The second objective is to predict where a user is likely to end their trip given a set of initial conditions (e.g. starting point, date/time, weather, etc.). This can help in anticipating the availability of free docking stations when the user arrives at their destination.

Some assumptions are made throughout this analysis: although Bixi docking stations are normally fixed, it is assumed that these stations can be moved around for the sake of this optimization problem. Secondly, despite focusing on Bixi's bike share data, it is assumed that the nature of this data can possibly be extrapolated to other forms of micro-mobility vehicles (e.g. e-scooters, e-bikes).

Related analyses have been done on bike share data. Most notably, and most similar to this project, is a [analysis](https://towardsdatascience.com/understanding-bixi-commuters-an-analysis-of-montreals-bike-share-system-in-python-cb34de0e2304) done by Gregoire C-M who sought to reveal the habits of Bixi commuters using Bixi's 2018 dataset. Another [work](https://towardsdatascience.com/exploring-toronto-bike-share-ridership-using-python-3dc87d35cb62) by Yizhao Tan also explores this with the dataset of Toronto's bike sharing program. Another analysis of Paris' Vélib bike sharing program using clustering is also discussed in a [paper](https://hal.archives-ouvertes.fr/hal-01494490/document) by Yunlong Feng, Roberta Costa Affonso and Marc Zolghadri.

<br>

## Dataset

The dataset that will be used is the Bixi trip history data taken from Bixi Montreal’s [open data](https://montreal.bixi.com/en/open-data). The files taken from the website are comprised of two types of CSVs: trip data of all Bixi trips from April to October, and Bixi station information.

The monthly Bixi trip data CSVs are comprised of the following information:
* `start_date` (the datetime of the moment a user has started a trip)
* `start_station_code` (the ID of the Bixi station where the user has started the trip)
* `end_date` (the datetime of the moment a user has ended a trip)
* `end_station_code` (the ID of the Bixi station where the user has ended the trip)
* `duration_sec` (the total duration of the trip in seconds)
* `is_member` (boolean variable to identify whether the user is a member of the Bixi service)

The Bixi station CSV is comprised of the following information:
* `code` (ID of the Bixi station)
* `name` (name of the street/intersection where the Bixi station is located)
* `latitude` (latitude of the Bixi station)
* `longitude` (longitude of the Bixi station)

Trip and Bixi station data from 2014 to 2019 will be utilized.


## Methods and Technologies

The first objective is to identify when and which bike stations are highly congested depending on time of day and month. Following this, the goal is to analyze where new stations can be placed in order to reduce congestion in high-traffic areas, and increase availability. In order to do this, K-means clustering will be used to cluster bike stations by location (longitude, latitude). From here, the weight of each point will be adjusted depending on how many trips have started and ended at each respective station. This will result in clusters where the center is skewed towards locations where the traffic is at its highest. Upon identifying the centers of all clusters, potential new locations can be considered to reduce traffic in these high-traffic areas.

The technology that will be used to create the K-means clusters is the [Spark clustering library](https://spark.apache.org/docs/latest/mllib-clustering.html)

The second objective is predicting the likelihood where a Bixi rider will go. For such, a classifier will be used to compare the numerous destinations a rider went to, from an initial point. More precisely, the classifier that will be utilized to predict the destinations is random forest. To build an accurate decision tree, dependent variables like time, day and year, which highly affect the destination will be integrated into building the best the best model.


The technology that will be utilized to implement this will be the [Scikit-learn random forest library](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
