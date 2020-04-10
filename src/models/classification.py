from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier as DTC, RandomForestClassifier as RFC
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


class Classifier:
    FEATURE_COLUMNS = ['start_cluster', 'month', 'day_of_week', 'hour_sin', 'hour_cos']

    def __init__(self):
        self.model = None

    def train_model(self, data):
        # Create features vector from multiple columns
        assembler = VectorAssembler(inputCols=self.FEATURE_COLUMNS, outputCol='features', handleInvalid='skip')
        data_with_features_column = assembler.transform(data)

        feature_indexer = VectorIndexer(inputCol='features', outputCol='indexed_features').fit(data_with_features_column)
        pipeline = Pipeline(stages=[feature_indexer, self.model])

        train_set, test_set = data_with_features_column.randomSplit([0.8, 0.2])

        # Train the model
        trained_model = pipeline.fit(train_set)

        # Make predictions
        predictions = trained_model.transform(test_set)

        # Output metrics
        evaluator = MulticlassClassificationEvaluator(labelCol='end_cluster', predictionCol='prediction')
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


class DecisionTreeClassifier(Classifier):
    def __init__(self):
        self.model = DTC(labelCol='end_cluster', featuresCol='indexed_features')


class RandomForestClassifier(Classifier):
    def __init__(self):
        self.model = RFC(labelCol='end_cluster', featuresCol='indexed_features')
