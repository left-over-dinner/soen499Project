import folium

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

class Regression:
    FEATURES_LONGITUDE = ['day_of_week', 'month', 'hour_sin', 'hour_cos', 'start_longitude', 'temperature']
    FEATURES_LATITUDE = ['day_of_week', 'month', 'hour_sin', 'hour_cos', 'start_latitude', 'temperature']
    DATASET_SPLIT = [0.8, 0.2]
    VISUALIZE_DATAPOINTS = 500
    regression_model, output_file_short_name = None, None

    def train_and_predict_feature(self, data, input_column, label_column):
        assembler = VectorAssembler(inputCols=input_column, outputCol='features', handleInvalid = 'skip')
        data = assembler.transform(data)

        feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
        (training_data, test_data) = data.randomSplit(self.DATASET_SPLIT)

        regression = self.regression_model(featuresCol="indexedFeatures", labelCol=label_column)
        pipeline = Pipeline(stages=[feature_indexer, regression])
        model = pipeline.fit(training_data)

        if label_column == 'end_longitude': 
            return model.transform(test_data).withColumnRenamed('prediction', 'predicted_longitude')
            
        return model.transform(test_data).withColumnRenamed('prediction', 'predicted_latitude')

    def calculate_rmse(self, data, label_column, predicted_column):
        evaluator = RegressionEvaluator(labelCol=label_column, predictionCol=predicted_column, metricName="rmse")
        rmse = evaluator.evaluate(data)
        print("Root Mean Squared Error (RMSE) of " + predicted_column +  " on test data = " + str(rmse))

    def train_model(self, data):
        # Obtain predications for longitude and latitude
        predictions_longitude = self.train_and_predict_feature(data, self.FEATURES_LONGITUDE, 'end_longitude')
        predictions_latitude = self.train_and_predict_feature(data, self.FEATURES_LATITUDE, 'end_latitude')

        # Calculate root mean squared error for predicted longitude and latitude respectively
        self.calculate_rmse(predictions_longitude, 'end_longitude', 'predicted_longitude')
        self.calculate_rmse(predictions_latitude, 'end_latitude', 'predicted_latitude')

        # Combine results
        results_compiled = predictions_longitude.join(predictions_latitude, on=['id'])\
            .withColumnRenamed('end_longitude', 'actual_longitude')\
            .withColumnRenamed('end_latitude', 'actual_latitude')\
            .rdd

        # Print results on a map of Montreal
        Montreal = [45.508154, -73.587450]
        montreal_map = folium.Map(
            location = Montreal,
            zoom_start = 12,
            tiles = 'CartoDB positron'
        )

        stations = results_compiled.take(self.VISUALIZE_DATAPOINTS)
        for station in stations:
            folium.Circle(
                location=[station.actual_latitude, station.actual_longitude],
                radius=2,
                color='#3186cc',
                fill=True
            ).add_to(montreal_map)
            folium.Circle(
                location=[station.predicted_latitude, station.predicted_longitude],
                radius=2,
                color='crimson',
                fill=True
            ).add_to(montreal_map)

        montreal_map.save('lat_long_'+self.output_file_short_name+'_regression.html')

class RandomForestRegression(Regression):
    def __init__(self):
        self.regression_model = RandomForestRegressor
        self.output_file_short_name = "random_forest"

class DecisionTreeRegression(Regression):
    def __init__(self):
        self.regression_model = DecisionTreeRegressor
        self.output_file_short_name = "decision_tree"