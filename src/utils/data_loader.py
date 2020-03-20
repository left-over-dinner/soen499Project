import os
from functools import reduce
from pyspark.sql import DataFrame


def get_bixi_data(spark, data_directory):
    trip_histories = []

    with os.scandir(data_directory) as entries:
        for entry in entries:
            trips_df = spark.read.csv(f'{entry.path}/OD*.csv', header=True, mode='DROPMALFORMED')
            stations_df = spark.read.csv(f'{entry.path}/Station*.csv', header=True, mode='DROPMALFORMED')

            start_stations_df = stations_df \
                .withColumnRenamed('code', 'start_station_code') \
                .withColumnRenamed('name', 'start_name') \
                .withColumnRenamed('latitude', 'start_latitude') \
                .withColumnRenamed('longitude', 'start_longitude')

            end_stations_df = stations_df \
                .withColumnRenamed('code', 'end_station_code') \
                .withColumnRenamed('name', 'end_name') \
                .withColumnRenamed('latitude', 'end_latitude') \
                .withColumnRenamed('longitude', 'end_longitude')

            combined_df = trips_df \
                .join(start_stations_df, 'start_station_code') \
                .join(end_stations_df, 'end_station_code')

            trip_histories.append(combined_df.drop('start_station_code', 'end_station_code'))

    trip_histories_df = reduce(DataFrame.unionAll, trip_histories)
    return trip_histories_df
