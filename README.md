# SOEN 499 Project

## Abstract

Our project aims to study the dataset provided by the Montreal bike share company Bixi. In this analysis, we plan to look at the trip histories of commuters from 2014 to 2019 to determine the top starting and ending stations in the city. More specifically, we can find out where and when should be the focus of Bixi's deployments by observing emergent patterns over time (e.g. different zones of high activity at different times of the week or year). Additionally, the analysis of the Bixi dataset will also be used to predict where a rider is likely to end their trip given a starting station, time of day and/or time of year.

<br>

## Introduction

As the growing number of inhabitants have started to put a strain on the transporation networks of major cities, many have turned to alternative modes of transportation. This includes popular offerings such as dockless e-scooters and bike sharing programs like Bixi in Montreal. One issue still remains however: matching the demand for bikes with availability. Companies like Bixi need a way to optimize the deployment of their vehicles by determining where and when is best for them to do so. With this, not only can the company increase their revenue, but users will also benefit from the more consistent availability, and thus alleviate the strain on the other transportation networks. This is exactly what this project seeks to do.

Two objectives are set out for this project. The first is to determine the most favourable starting and ending stations in the city depending on the time of day, day of the week, and time of the year. Such an analysis will reveal where the major focus of deployments should be. The second objective is to predict where a user is likely to end their trip given a set of initial conditions (e.g. starting point, date/time, weather, etc.). This can help in anticipating the availability of free docking stations when the user arrives at their destination.

Some assumptions are made throughout this analysis: although Bixi docking stations are normally fixed, it is assumed that these stations can be moved around for the sake of this optimization problem. Secondly, despite focusing on Bixi's bike share data, it is assumed that the nature of this data can possibly be extrapolated to other forms of micro-mobility vehicles (e.g. e-scooters, e-bikes).

Related analyses have been done on bike share data. Most notably, and most similar to this project, is a [analysis](https://towardsdatascience.com/understanding-bixi-commuters-an-analysis-of-montreals-bike-share-system-in-python-cb34de0e2304) done by Gregoire C-M who sought to reveal the habits of Bixi commuters using Bixi's 2018 dataset. Another [work](https://towardsdatascience.com/exploring-toronto-bike-share-ridership-using-python-3dc87d35cb62) by Yizhao Tan also explores this with the dataset of Toronto's bike sharing program. Another analysis of Paris' Vélib bike sharing program using clustering is also discussed in a [paper](https://hal.archives-ouvertes.fr/hal-01494490/document) by Yunlong Feng, Roberta Costa Affonso and Marc Zolghadri.
