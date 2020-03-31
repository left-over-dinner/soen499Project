from pyspark.sql import SparkSession
from utils.data_loader import get_bixi_data
from utils.clustering import cluster_stations
from models.decision_tree_classifier import DecisionTreeClassifier
from models.decision_tree_regression import DecisionTreeRegression
from main import transform_time_features, combine_clusters_with_trips 


def init_spark():
    spark = SparkSession.builder \
        .appName('BIXI Predictor') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '4g') \
        .getOrCreate()
    return spark


if __name__ == '__main__':
    DATA_DIRECTORY = '../data'
    spark = init_spark()
    trip_history_data, all_station_data = get_bixi_data(spark, DATA_DIRECTORY)
    unique_stations_count = trip_history_data.select('start_name').distinct().count()
    clustered_stations = cluster_stations(all_station_data)
    trip_clustered_data = combine_clusters_with_trips(trip_history_data, clustered_stations)
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.train_model(trip_clustered_data,unique_stations_count)
    exit(0)
    decision_tree_regression = DecisionTreeRegression()
    decision_tree_regression.train_model(trip_history_data,unique_stations_count)
    
    
