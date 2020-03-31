from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor as DTR
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

class DecisionTreeRegression:
    FEATURE_COLUMNS = ['indexed_start_name', 'hour','day_of_week']
    NUM_TREES = 10

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