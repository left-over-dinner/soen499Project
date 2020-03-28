import folium
import random

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from utils.data_loader import get_bixi_data

DATA_DIRECTORY = '../data'


def init_spark():
    spark = SparkSession.builder.appName("BIXI Predictor").getOrCreate()
    return spark

# features_longitude = ['day_of_week', 'hour', 'duration_sec', 'start_longitude']
# features_latitude = ['day_of_week', 'hour', 'duration_sec', 'start_latitude']
# true_vals = ['end_latitude', 'end_longitude']

spark = init_spark()
data = get_bixi_data(spark, DATA_DIRECTORY)

# assembler = VectorAssembler(inputCols=features_longitude, outputCol='features')
# data = assembler.transform(data)

# # Automatically identify categorical features, and index them.
# # Set maxCategories so features with > 4 distinct values are treated as continuous.
# featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# # Split the data into training and test sets (30% held out for testing)
# (trainingData, testData) = data.randomSplit([0.7, 0.3])

# # Train a RandomForest model.
# rf = RandomForestRegressor(featuresCol="indexedFeatures", labelCol="end_longitude")

# # Chain indexer and forest in a Pipeline
# pipeline = Pipeline(stages=[featureIndexer, rf])

# # Train model.  This also runs the indexer.
# model = pipeline.fit(trainingData)

# Make predictions.
# predictions = model.transform(testData)
# print(predictions.show())

# Select example rows to display.
# predictions.select("prediction", "end_longitude", "features").show(5)

# Select (prediction, true label) and compute test error
# evaluator = RegressionEvaluator(labelCol="end_longitude", predictionCol="prediction", metricName="rmse")
# rmse = evaluator.evaluate(predictions)
# print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# rfModel = model.stages[1]
# print(rfModel)  # summary only


features_longitude = ['day_of_week', 'hour', 'duration_sec', 'start_longitude']
features_latitude = ['day_of_week', 'hour', 'duration_sec', 'start_latitude']

stuff = data.select('end_longitude', 'end_latitude')
true_vals = stuff.withColumn('longitude', stuff.end_longitude).withColumn('latitude', stuff.end_latitude)
print(true_vals.show(10))

assembler = VectorAssembler(inputCols=features_longitude, outputCol='features')
dataLongitude = assembler.transform(data)

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dataLongitude)

(trainingData, testData) = dataLongitude.randomSplit([0.7, 0.3])

rf = RandomForestRegressor(featuresCol="indexedFeatures", labelCol="end_longitude")

pipeline = Pipeline(stages=[featureIndexer, rf])

model = pipeline.fit(trainingData)

predictions_longitude = model.transform(testData)


# assembler2 = VectorAssembler(inputCols=features_latitude, outputCol='features')
# dataLatitude = assembler.transform(data)

# featureIndexer2 = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dataLatitude)

# (trainingData2, testData2) = dataLatitude.randomSplit([0.7, 0.3])

# rf2 = RandomForestRegressor(featuresCol="indexedFeatures", labelCol="end_latitude")

# pipeline2 = Pipeline(stages=[featureIndexer2, rf2])

# model2 = pipeline2.fit(trainingData2)

# predictions_latitude = model2.transform(testData2)

pred_long = predictions_longitude.select("prediction", "end_latitude")
# pred_lat = predictions_latitude.select("prediction")

predictions = pred_long.withColumn('longitude', pred_long.prediction).withColumn('latitude', pred_long.end_latitude)
# predictions = pred_lat.withColumn('latitude', pred_lat.prediction)

print(predictions.show(10))


Montreal = [45.508154, -73.587450]
montreal_map = folium.Map(
    location = Montreal,
    zoom_start = 12,
    tiles = 'CartoDB positron'
)

stations = true_vals.rdd.take(200)
for station in stations:
    folium.Circle(
        location=[station.latitude, station.longitude],
        radius=2,
        color='#3186cc',
        fill=True
    ).add_to(montreal_map)

montreal_map.save('centroids.html')

stations = predictions.rdd.take(200)
for station in stations:
    folium.Circle(
        location=[station.latitude, station.longitude],
        radius=2,
        color='crimson',
        fill=True
    ).add_to(montreal_map)

montreal_map.save('clusters.html')