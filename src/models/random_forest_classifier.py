from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RFC
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class RandomForestClassifier:
    FEATURE_COLUMNS = ['start_name', 'month', 'day_of_week', 'hour_sin', 'hour_cos']

    def train_model(self, data):        
        # Create features vector from multiple columns
        assembler = VectorAssembler(inputCols=self.FEATURE_COLUMNS, outputCol='features')
        data_with_features_column = assembler.transform(data)

        feature_indexer = VectorIndexer(inputCol='features', outputCol='indexed_features').fit(data_with_features_column)
        random_forest = RFC(labelCol='end_name', featuresCol='indexed_features')

        pipeline = Pipeline(stages=[feature_indexer, random_forest])

        train_set, test_set = data_with_features_column.randomSplit([0.8, 0.2])

        # Train the model
        model = pipeline.fit(train_set)

        # Make predictions
        predictions = model.transform(test_set)

        # Output metrics
        evaluator = MulticlassClassificationEvaluator(labelCol='end_name', predictionCol='prediction')
        evaluator.setMetricName('accuracy')
        accuracy = evaluator.evaluate(predictions)

        evaluator.setMetricName('weightedPrecision')
        precision = evaluator.evaluate(predictions)

        evaluator.setMetricName('weightedRecall')
        recall = evaluator.evaluate(predictions)

        evaluator.setMetricName('f1')
        f1 = evaluator.evaluate(predictions)
    
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')

