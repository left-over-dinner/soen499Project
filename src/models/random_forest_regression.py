import folium

from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

class RandomForestRegression:
    FEATURES_LONGITUDE = ['day_of_week', 'hour_sin', 'hour_cos', 'start_longitude']
    FEATURES_LATITUDE = ['day_of_week', 'hour_sin', 'hour_cos', 'start_latitude']

    def train_model(self, data):
        select_lat_long = data.select('end_longitude', 'end_latitude', 'id')
        actual_lat_long = select_lat_long.withColumn('actual_longitude', select_lat_long.end_longitude).withColumn('actual_latitude', select_lat_long.end_latitude)

        # Obtain predications for longitude
        assembler = VectorAssembler(inputCols=self.FEATURES_LONGITUDE, outputCol='features')
        data_longitude = assembler.transform(data)

        feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data_longitude)
        (training_data_longitude, test_data_longitude) = data_longitude.randomSplit([0.7, 0.3])

        rf = RandomForestRegressor(featuresCol="indexedFeatures", labelCol="end_longitude")
        pipeline = Pipeline(stages=[feature_indexer, rf])
        model_longitude = pipeline.fit(training_data_longitude)

        predictions_longitude = model_longitude.transform(test_data_longitude)

        # Obtain predications for latitude
        assembler = VectorAssembler(inputCols=self.FEATURES_LATITUDE, outputCol='features')
        data_latitude = assembler.transform(data)

        feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data_latitude)
        (training_data_latitude, test_data_latitude) = data_latitude.randomSplit([0.7, 0.3])

        rf = RandomForestRegressor(featuresCol="indexedFeatures", labelCol="end_latitude")
        pipeline = Pipeline(stages=[feature_indexer, rf])
        model_latitude = pipeline.fit(training_data_latitude)

        predictions_latitude = model_latitude.transform(test_data_latitude)

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

        # Combine predicted lat/long and actual lat/long
        pred_long = predictions_longitude.select("predicted_longitude", "id")
        pred_lat = predictions_latitude.select("predicted_latitude", "id")

        predictions = pred_long.join(pred_lat, on=['id'])
        results_compiled = predictions.join(actual_lat_long, on=['id']).rdd

        # Print results on a map of Montreal
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