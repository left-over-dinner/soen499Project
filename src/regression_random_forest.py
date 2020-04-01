import folium
import random
import math

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from utils.data_loader import get_bixi_data

DATA_DIRECTORY = '../data'


def init_spark():
    spark = SparkSession.builder.appName("BIXI Predictor").getOrCreate()
    return spark

hour_sin_udf = udf(lambda hour: math.sin(2 * math.pi * hour / 23), FloatType())
hour_cos_udf = udf(lambda hour: math.cos(2 * math.pi * hour / 23), FloatType())

spark = init_spark()
data = get_bixi_data(spark, DATA_DIRECTORY)[0]

data = data \
    .withColumn('hour_sin', hour_sin_udf('hour')) \
    .withColumn('hour_cos', hour_cos_udf('hour'))

features_longitude = ['day_of_week', 'hour_sin', 'hour_cos', 'start_longitude']
features_latitude = ['day_of_week', 'hour_sin', 'hour_cos', 'start_latitude']

stuff = data.select('end_longitude', 'end_latitude', 'id')
true_vals = stuff.withColumn('actual_longitude', stuff.end_longitude).withColumn('actual_latitude', stuff.end_latitude)

assembler = VectorAssembler(inputCols=features_longitude, outputCol='features')
dataLongitude = assembler.transform(data)

# Obtain predications for longitude
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dataLongitude)
(trainingDataLongitude, testDataLongitude) = dataLongitude.randomSplit([0.7, 0.3])

rf = RandomForestRegressor(featuresCol="indexedFeatures", labelCol="end_longitude")
pipeline = Pipeline(stages=[featureIndexer, rf])
modelLongitude = pipeline.fit(trainingDataLongitude)

predictions_longitude = modelLongitude.transform(testDataLongitude)

# Obtain predications for latitude
assembler = VectorAssembler(inputCols=features_latitude, outputCol='features')
dataLatitude = assembler.transform(data)

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dataLatitude)
(trainingDataLatitude, testDataLatitude) = dataLatitude.randomSplit([0.7, 0.3])

rf = RandomForestRegressor(featuresCol="indexedFeatures", labelCol="end_latitude")
pipeline = Pipeline(stages=[featureIndexer, rf])
modelLatitude = pipeline.fit(trainingDataLatitude)

predictions_latitude = modelLatitude.transform(testDataLatitude)

# Rename columns and combine predictions
predictions_longitude = predictions_longitude.withColumn('predicted_longitude', predictions_longitude.prediction)
predictions_latitude = predictions_latitude.withColumn('predicted_latitude', predictions_latitude.prediction)

# Calculate root mean squared error on longitude and latitude
evaluator = RegressionEvaluator(labelCol="end_longitude", predictionCol="predicted_longitude", metricName="rmse")
rmse_longitude = evaluator.evaluate(predictions_longitude)
print("Root Mean Squared Error (RMSE) of longitude on test data = %g" % rmse_longitude)
evaluator = RegressionEvaluator(labelCol="end_latitude", predictionCol="predicted_latitude", metricName="rmse")
rmse_latitude = evaluator.evaluate(predictions_latitude)
print("Root Mean Squared Error (RMSE) of latitude on test data = %g" % rmse_latitude)

pred_long = predictions_longitude.select("predicted_longitude", "id")
pred_lat = predictions_latitude.select("predicted_latitude", "id")

predictions = pred_long.join(pred_lat, on=['id'])

results_compiled = predictions.join(true_vals, on=['id']).rdd

Montreal = [45.508154, -73.587450]
montreal_map = folium.Map(
    location = Montreal,
    zoom_start = 12,
    tiles = 'CartoDB positron'
)

stations = results_compiled.take(500)
for station in stations:
    folium.Circle(
        location=[station.actual_latitude, station.actual_longitude],
        radius=2,
        color='#3186cc',
        fill=True
    ).add_to(montreal_map)

stations = results_compiled.take(500)
for station in stations:
    folium.Circle(
        location=[station.predicted_latitude, station.predicted_longitude],
        radius=2,
        color='crimson',
        fill=True
    ).add_to(montreal_map)

montreal_map.save('lat_long_random_forest_regression.html')