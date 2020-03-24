import os
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, hour

def get_bixi_data(spark, data_directory):
    trip_histories = []

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

            #start_date split into different columns
            adjusted_df = combined_df \
                .withColumn('year', year('start_date')) \
                .withColumn('month', month('start_date')) \
                .withColumn('day_of_month', dayofmonth('start_date')) \
                .withColumn('day_of_week', dayofweek('start_date')) \
                .withColumn('hour', hour('start_date'))

            trip_histories.append(adjusted_df.drop('start_station_code', 'end_station_code'))

    trip_histories_df = reduce(DataFrame.unionAll, trip_histories)
    
    trip_histories_df = trip_histories_df \
        .withColumn('start_date', trip_histories_df.start_date.cast('date')) \
        .withColumn('end_date', trip_histories_df.end_date.cast('date')) \
        .withColumn('duration_sec', trip_histories_df.duration_sec.cast('integer')) \
        .withColumn('is_member', trip_histories_df.is_member.cast('integer')) \
        .withColumn('start_latitude', trip_histories_df.start_latitude.cast('double')) \
        .withColumn('start_longitude', trip_histories_df.start_longitude.cast('double')) \
        .withColumn('end_latitude', trip_histories_df.end_latitude.cast('double')) \
        .withColumn('end_longitude', trip_histories_df.end_longitude.cast('double')) \
        .withColumn('year', trip_histories_df.year.cast('integer')) \
        .withColumn('month', trip_histories_df.month.cast('integer')) \
        .withColumn('day_of_month', trip_histories_df.day_of_month.cast('integer')) \
        .withColumn('day_of_week', trip_histories_df.day_of_week.cast('integer')) \
        .withColumn('hour', trip_histories_df.hour.cast('integer'))

    return trip_histories_df
