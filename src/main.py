from pyspark.sql import SparkSession
from utils.data_loader import get_bixi_data


def init_spark():
    spark = SparkSession.builder.appName("BIXI Predictor").getOrCreate()
    return spark


if __name__ == '__main__':
    DATA_DIRECTORY = '../data'

    spark = init_spark()
    df = get_bixi_data(spark, DATA_DIRECTORY)
    df.show()
