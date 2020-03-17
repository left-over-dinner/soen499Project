# SOEN 499 Project

## Abstract

Our project aims to study the dataset provided by the Montreal bike share company Bixi. In this analysis, we plan to look at the trip histories of commuters from 2014 to 2019
and use that data to predict where users are likely to end their trips. Information of their starting point, the time of day and/or time of year will also be used to aid in the predictions. By doing so, patterns of movement from different areas of the city can be discovered, and thus adjustments can be made to better improve the flow and availability of this mode of alternative transport.

<br>

## Introduction

As the growing number of inhabitants have started to put a strain on the transporation networks of major cities, many have turned to alternative modes of transportation. This includes popular offerings such as dockless e-scooters and bike sharing programs like Bixi in Montreal. One issue still remains however: matching the demand for bikes with availability. In order to improve on this issue, companies like Bixi need a way of understanding the traffic flows of their network (i.e. where users start and end their trips, and at what times) so that they can make better accomodations to distribute their vehicles and docking stations at the appropriate times and therefore increase availability.

The objective of this project is use Bixi's past trip history datasets to predict where a user is likely to end their trip, given a set of initial conditions. These can include the starting point of their trip, the time of day, the day of the week or year, or possibly even the weather. With this, information can be deduced to help with anticipating the availability of free docking stations when the user arrives at their destination. This prediction problem will be evaluated as either a multi-class classification problem (where each station is a class label), or as a regression problem where a coordinate is determined. The performance of both will be compared to see which kind of predictor yields better results. Additionally, classification and regression will both be implemented with Random Forests and simple Decision Trees to see which model has the better performance.

#### Related Work

Currently, there are no related works that try to predict the end destinations of users in a bike-sharing network. However, there exists numerous works that explore bike-sharing datasets, works that try to classify the types of users (i.e. casual commuter or subscribing member) in a bike-sharing system, and works that try predicting demand of these services as a regression problem.

Most notably, and most similar to this project, is an [analysis](https://towardsdatascience.com/understanding-bixi-commuters-an-analysis-of-montreals-bike-share-system-in-python-cb34de0e2304) done by Gregoire C-M who sought to reveal the habits of Bixi commuters using Bixi's 2018 dataset. In this, he mainly tries to explore the dataset provided by Bixi, but also attempts to correlate rider demand with daily temperature to make regression predictions with scikit-learn's `LinearRegression` model. He concludes that there is a correlation between increasing temperatures and ridership up until a certain point where demand starts to decrease.

There are multiple other works that tackle the problem of predicting the demand of bike-sharing services. An extensive [paper](https://pdfs.semanticscholar.org/32f7/f3da38fe78fb5237aea70f343cc8c77af92b.pdf) produced by Arnab Kumar Datta details the performance of predicting bike-share usage with Decision Trees, Random Forest and AdaBoost models. The conclusions of this paper reported favourable results for the Random Forest and AdaBoost techniques, which both had lower Root Mean Squared Errors (RMSE) than that of the Decision Tree technique. Another [work](http://ijiset.com/vol2/v2s4/IJISET_V2_I4_195.pdf) by Patil, Musale and Rao, also used Decision Trees, Random Forests and Boosted Decision Tree algorithms to make predictions about bike-share demand. Similarly to the previous paper mentioned, their results point towards Random Forests as a strong performer. Other works with a similar objective exist, but mainly employ other algorithms and techniques to build their regressors.

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

<br>

## Methods and Technologies

The first objective is to identify when and which bike stations are highly congested depending on time of day and month. Following this, the goal is to analyze where new stations can be placed in order to reduce congestion in high-traffic areas, and increase availability. In order to do this, K-means clustering will be used to cluster bike stations by location (longitude, latitude). From here, the weight of each point will be adjusted depending on how many trips have started and ended at each respective station. This will result in clusters where the center is skewed towards locations where the traffic is at its highest. Upon identifying the centers of all clusters, potential new locations can be considered to reduce traffic in these high-traffic areas.

The technology that will be used to create the K-means clusters is the [Spark clustering library](https://spark.apache.org/docs/latest/mllib-clustering.html)

The second objective is predicting the likelihood where a Bixi rider will go. For such, a classifier will be used to compare the numerous destinations a rider went to, from an initial point. More precisely, the classifier that will be utilized to predict the destinations is random forest. To build an accurate decision tree, dependent variables like time, day and year, which highly affect the destination will be integrated into building the best the best model.


The technology that will be utilized to implement this will be the [Scikit-learn random forest library](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
