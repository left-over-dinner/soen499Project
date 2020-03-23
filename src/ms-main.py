from pyspark.sql import SparkSession
from utils.data_loader import get_bixi_data

from models.random_forest_classifier import RandomForestClassifier


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
    data.select(['year','month', 'day_of_month', 'day_of_week','hour']).show()
    exit(0)
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.train_model(data)
