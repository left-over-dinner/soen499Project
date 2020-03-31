import os
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, hour, monotonically_increasing_id 

def get_bixi_data(spark, data_directory):
    weather_df = load_weather_data(spark, data_directory)
    trip_histories_df, all_stations_df = load_bixi_data(spark, data_directory)

    trip_histories_df = combine_weather_with_trip_histories(trip_histories_df, weather_df)

    trip_histories_df = trip_histories_df \
        .withColumn('id', monotonically_increasing_id()) \
        .withColumn('start_date', trip_histories_df.start_date.cast('date')) \
        .withColumn('end_date', trip_histories_df.end_date.cast('date')) \
        .withColumn('is_member', trip_histories_df.is_member.cast('integer')) \
        .withColumn('start_latitude', trip_histories_df.start_latitude.cast('float')) \
        .withColumn('start_longitude', trip_histories_df.start_longitude.cast('float')) \
        .withColumn('end_latitude', trip_histories_df.end_latitude.cast('float')) \
        .withColumn('end_longitude', trip_histories_df.end_longitude.cast('float')) \
        .withColumn('month', trip_histories_df.month.cast('integer')) \
        .withColumn('day_of_week', trip_histories_df.day_of_week.cast('integer')) \
        .withColumn('hour', trip_histories_df.hour.cast('integer')) \
        .withColumn('temperature', trip_histories_df.temperature.cast('float'))

    all_stations_df = all_stations_df \
        .withColumn('longitude', all_stations_df.longitude.cast('float')) \
        .withColumn('latitude', all_stations_df.latitude.cast('float'))

    return trip_histories_df, all_stations_df

def load_bixi_data(spark, data_directory):
    trip_histories = []
    stations = []

    with os.scandir(data_directory) as entries:
        for entry in entries:
            if entry.path.endswith('DS_Store') or entry.name == 'weather':
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
    
    # Split start_date into different columns
    trip_histories_df = trip_histories_df \
        .orderBy('start_date') \
        .withColumn('id', monotonically_increasing_id()) \
        .withColumn('month', month('start_date')) \
        .withColumn('day_of_week', dayofweek('start_date')) \
        .withColumn('hour', hour('start_date')) \
        .withColumn('year', year('start_date')) \
        .withColumn('day_of_month', dayofmonth('start_date'))

    return trip_histories_df, all_stations_df

def load_weather_data(spark, data_directory):
    try:
        weather_df = spark.read.csv(f'{data_directory}/weather/*.csv', header=True, mode='DROPMALFORMED')
        return weather_df \
            .select(['Year', 'Month', 'Day', 'Time', 'Temp (°C)', 'Weather']) \
            .withColumn('hour', hour('Time')) \
            .withColumnRenamed('Year', 'year') \
            .withColumnRenamed('Month', 'month') \
            .withColumnRenamed('Day', 'day_of_month') \
            .withColumnRenamed('Temp (°C)', 'temperature') \
            .withColumnRenamed('Weather', 'weather')

    except:
        print('No weather data found.')

def combine_weather_with_trip_histories(trip_histories_df, weather_df):
    trip_histories_df = trip_histories_df \
        .join(weather_df, ['year', 'month', 'day_of_month', 'hour'])
    
    return trip_histories_df \
        .drop('year') \
        .drop('day_of_month') \
        .drop('Time') \
