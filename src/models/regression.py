import folium
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor as RFR, DecisionTreeRegressor as DTR
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator


VISUALIZE_DATAPOINTS = 500

def map_regression_predictions(results, output_file_name):
    # Print results on a map of Montreal
    Montreal = [45.508154, -73.587450]
    montreal_map = folium.Map(
        location = Montreal,
        zoom_start = 12,
        tiles = 'CartoDB positron'
    )

    stations = results.take(VISUALIZE_DATAPOINTS)
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

    montreal_map.save('lat_long_' + output_file_name + '_regression.html')


class Regressor:
    FEATURES_LONGITUDE = ['month', 'day_of_week', 'hour_sin', 'hour_cos', 'start_longitude']
    FEATURES_LATITUDE = ['month', 'day_of_week', 'hour_sin', 'hour_cos', 'start_latitude']
    DATASET_SPLIT = [0.8, 0.2]

    def __init__(self):
        self.regression_model = None
        self.output_file_short_name = None

    def train_model(self, data):
        # Obtain predications for longitude and latitude
        predictions_longitude = self.train_and_predict(data, self.FEATURES_LONGITUDE, 'end_longitude')
        predictions_latitude = self.train_and_predict(data, self.FEATURES_LATITUDE, 'end_latitude')

        # Calculate root mean squared error for predicted longitude and latitude respectively
        self.calculate_rmse(predictions_longitude, 'end_longitude', 'predicted_longitude')
        self.calculate_rmse(predictions_latitude, 'end_latitude', 'predicted_latitude')

        # Combine results
        results_compiled = predictions_longitude.join(predictions_latitude, on=['id'])\
            .withColumnRenamed('end_longitude', 'actual_longitude')\
            .withColumnRenamed('end_latitude', 'actual_latitude')\
            .rdd

        map_regression_predictions(results_compiled, self.output_file_short_name)

    def train_and_predict(self, data, input_column, label_column):
        assembler = VectorAssembler(inputCols=input_column, outputCol='features', handleInvalid='skip')
        data = assembler.transform(data)

        feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
        training_data, test_data = data.randomSplit(self.DATASET_SPLIT)

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


class RandomForestRegressor(Regressor):
    def __init__(self):
        self.regression_model = RFR
        self.output_file_short_name = "random_forest"

class DecisionTreeRegressor(Regressor):
    def __init__(self):
        self.regression_model = DTR
        self.output_file_short_name = "decision_tree"