from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor as DTR
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

class DecisionTreeRegression:
    FEATURES_LONGITUDE = ['day_of_week', 'hour_sin', 'hour_cos', 'start_longitude']
    FEATURES_LATITUDE = ['day_of_week', 'hour_sin', 'hour_cos', 'start_latitude']
    DATASET_SPLIT = [0.8, 0.2]
    VISUALIZE_DATAPOINTS = 500

    def train_model(self, data, unique_stations_count):
        data = StringIndexer(inputCol='start_name', outputCol='indexed_start_name').fit(data).transform(data)

        assembler = VectorAssembler(inputCols=self.FEATURE_COLUMNS, outputCol='features')
        data = assembler.transform(data)
        featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(data)

        (trainingData, testData) = data.randomSplit([0.8, 0.2])
        decision_tree_regression = DTR(featuresCol="indexedFeatures", labelCol="end_latitude", maxBins = unique_stations_count)
        pipeline = Pipeline(stages=[featureIndexer, decision_tree_regression])

        model = pipeline.fit(trainingData)
        predictions = model.transform(testData)
        print(predictions.show())

        predictions.select("prediction", "end_latitude", "features").show(5)
        evaluator = RegressionEvaluator(
            labelCol="end_latitude", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

        rfModel = model.stages[1]
        print(rfModel)  # summary only