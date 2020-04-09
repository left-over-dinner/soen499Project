import sys
import math
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from functools import reduce

from utils.clustering import cluster_stations
from utils.data_loader import get_bixi_data

from models.classification import DecisionTreeClassifier, RandomForestClassifier
from models.regression import RandomForestRegression, DecisionTreeRegression


DATA_DIRECTORY = '../data'

trip_data, stations = None, None
clustered_data = None

def init_spark():
    spark = SparkSession.builder \
        .appName('BIXI Predictor') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '4g') \
        .getOrCreate()
    return spark

def transform_time_features(trips_df, spark):
    """Encode cyclical hour feature"""
    hour_sin_udf = udf(lambda hour: math.sin(2 * math.pi * hour / 23), FloatType())
    hour_cos_udf = udf(lambda hour: math.cos(2 * math.pi * hour / 23), FloatType())
    
    new_trips_df = trips_df \
        .withColumn('hour_sin', hour_sin_udf('hour')) \
        .withColumn('hour_cos', hour_cos_udf('hour'))

    return new_trips_df

def combine_clusters_with_trips(trip_data, clustered_stations):
    """Assign each trip's start station to a cluster."""
    clusters = clustered_stations.select(['name', 'prediction'])

    trip_data = trip_data \
        .join(clusters, trip_data.start_name == clusters.name) \
        .withColumnRenamed('prediction', 'start_cluster') \
        .drop('name') \
        .join(clusters, trip_data.end_name == clusters.name) \
        .withColumnRenamed('prediction', 'end_cluster') \
        .drop('name')

    return trip_data

def resample_data(trip_data, ratio=1):
    """Randomly undersample clusters relative to the minority cluster using the specified ratio."""
    class_distribution = trip_data.groupBy('end_cluster').count()

    print('\nDistribution of clusters:')
    class_distribution.orderBy('end_cluster').show()

    class_distribution = class_distribution.orderBy('count').collect()
    minority_class = class_distribution[0]
    df_list = [trip_data.where(trip_data.end_cluster == c[0]).sample(False, min(minority_class[1] * ratio / c[1], 1.0)) for c in class_distribution]
    
    resampled = reduce(DataFrame.unionAll, df_list)
    
    print('\nResampled distribution of clusters:')
    resampled.groupBy('end_cluster').count().orderBy('end_cluster').show()

    return resampled

def decision_tree_classification():
    print('\n------Training Decision Tree Classifier------')
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.train_model(clustered_data)

def decision_tree_regressor():
    print('\n------Training Decisition Tree Regression------')
    decision_tree_regression = DecisionTreeRegression()
    decision_tree_regression.train_model(trip_data)

def random_forest_classification():
    print('\n------Training Random Forest Classifier------')
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.train_model(clustered_data)

def random_forest_regressor():
    print('\n------Training Random Forest Regressor------')
    random_forest_regressor = RandomForestRegression()
    random_forest_regressor.train_model(trip_data)


# Arguments for terminal
METHODS = {
    "dtc": decision_tree_classification,
    "dtr": decision_tree_regressor,
    "rfc": random_forest_classification,
    "rfr": random_forest_regressor
}

if __name__ == '__main__':
    methods_to_run = None
    if len(sys.argv) == 1:
        # no argument provided, then run all methods
        # collect all method names to run
        methods_to_run = list(METHODS.keys())
    else:
        # valid argument, collect method name, continue
        if sys.argv[1] in METHODS:
            methods_to_run = [sys.argv[1]]
        # invalid argument, exit
        else:
            print("invalid argument:", sys.argv[1])
            exit(0)

    spark = init_spark()
    print('\n------Loading data------')
    trip_data, stations = get_bixi_data(spark, DATA_DIRECTORY)

    print('\n------Transforming time features------')
    trip_data = transform_time_features(trip_data, spark)
    
    # Cluster stations together if classification needs to be run
    if 'dtc' in methods_to_run or 'rfc' in methods_to_run:
        print('\n------Clustering stations------')
        clustered_stations = cluster_stations(stations)
        clustered_data = combine_clusters_with_trips(trip_data, clustered_stations)

        print('\n------Resampling data------')
        clustered_data = resample_data(clustered_data, 2)
    
    for method_name in methods_to_run:
        METHODS[method_name]()
