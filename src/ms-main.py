from pyspark.sql import SparkSession
from utils.data_loader import get_bixi_data

from models.decision_tree_classifier import DecisionTreeClassifier
from models.decision_tree_regression import DecisionTreeRegression 


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
    data = get_bixi_data(spark, DATA_DIRECTORY)
    unique_stations = data.select('start_name').distinct().count()
    decision_tree_regression = DecisionTreeRegression()
    decision_tree_regression.train_model(data,unique_stations)
    exit(0)
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.train_model(data)
