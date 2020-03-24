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

features = ['duration_sec', 'is_member', 'start_latitude']

spark = init_spark()
data = get_bixi_data(spark, DATA_DIRECTORY)

assembler = VectorAssembler(inputCols=features, outputCol='features')
data = assembler.transform(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestRegressor(featuresCol="indexedFeatures", labelCol="end_latitude")

# Chain indexer and forest in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, rf])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)
print(predictions.show())

# Select example rows to display.
predictions.select("prediction", "end_latitude", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="end_latitude", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rfModel = model.stages[1]
print(rfModel)  # summary only