import os
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import month, dayofweek, hour, monotonically_increasing_id 

def get_bixi_data(spark, data_directory):
    trip_histories = []
    stations = []

    with os.scandir(data_directory) as entries:
        for entry in entries:
            if entry.path.endswith('DS_Store'):
                continue

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

            trip_histories.append(combined_df.drop('start_station_code', 'end_station_code', 'duration_sec'))
            stations.append(stations_df.drop('code'))

    trip_histories_df = reduce(DataFrame.unionAll, trip_histories)
    all_stations_df = reduce(DataFrame.unionAll, stations).distinct()
    
    # Split start_date into different features
    trip_histories_df = trip_histories_df \
        .orderBy('start_date') \
        .withColumn('id', monotonically_increasing_id()) \
        .withColumn('month', month('start_date')) \
        .withColumn('day_of_week', dayofweek('start_date')) \
        .withColumn('hour', hour('start_date')) \

    # Cast row columns to appropriate types
    trip_histories_df = trip_histories_df \
        .orderBy('start_date') \
        .withColumn('id', monotonically_increasing_id()) \
        .withColumn('start_date', trip_histories_df.start_date.cast('date')) \
        .withColumn('end_date', trip_histories_df.end_date.cast('date')) \
        .withColumn('is_member', trip_histories_df.is_member.cast('integer')) \
        .withColumn('start_latitude', trip_histories_df.start_latitude.cast('double')) \
        .withColumn('start_longitude', trip_histories_df.start_longitude.cast('double')) \
        .withColumn('end_latitude', trip_histories_df.end_latitude.cast('double')) \
        .withColumn('end_longitude', trip_histories_df.end_longitude.cast('double')) \

    all_stations_df = all_stations_df \
        .withColumn('longitude', all_stations_df.longitude.cast('double')) \
        .withColumn('latitude', all_stations_df.latitude.cast('double'))

    return trip_histories_df, all_stations_df
