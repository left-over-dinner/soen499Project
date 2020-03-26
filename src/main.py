from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from utils.clustering import cluster_stations
from utils.data_loader import get_bixi_data

from models.random_forest_classifier import RandomForestClassifier


def init_spark():
    spark = SparkSession.builder \
        .appName('BIXI Predictor') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '4g') \
        .getOrCreate()
    return spark

def combine_clusters_with_trips(trip_data, clustered_stations):
    clusters = clustered_stations.select(['name', 'prediction'])

    trip_data = trip_data \
        .join(clusters, trip_data.start_name == clusters.name) \
        .withColumnRenamed('prediction', 'start_cluster') \
        .drop('name') \
        .join(clusters, trip_data.end_name == clusters.name) \
        .withColumnRenamed('prediction', 'end_cluster') \
        .drop('name')

    return trip_data

if __name__ == '__main__':
    DATA_DIRECTORY = '../data'

    spark = init_spark()
    trip_data, stations = get_bixi_data(spark, DATA_DIRECTORY)

    print('\n------Clustering stations------')
    clustered_stations = cluster_stations(stations)
    data = combine_clusters_with_trips(trip_data, clustered_stations)

    print('\nDistribution of clusters:')
    data.groupBy('end_cluster').count().orderBy('count').show()

    print('\n------Classifying data with Random Forest Classifier------')
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.train_model(data)
