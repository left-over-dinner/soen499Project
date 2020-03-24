from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier as DTC
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class DecisionTreeClassifier:
    FEATURE_COLUMNS = ['indexed_start_name', 'hour','day_of_week']
    
    def train_model(self, data):
        # Convert start station names to numerical value
        data = StringIndexer(inputCol='start_name', outputCol='indexed_start_name').fit(data).transform(data)
        
        # Create features vector from multiple columns
        assembler = VectorAssembler(inputCols=self.FEATURE_COLUMNS, outputCol='features')
        data_with_features_column = assembler.transform(data)

        label_indexer = StringIndexer(inputCol='end_name', outputCol='indexed_end_name').fit(data_with_features_column)
        feature_indexer = VectorIndexer(inputCol='features', outputCol='indexed_features').fit(data_with_features_column)
        decision_tree = DTC(labelCol='indexed_end_name', featuresCol='indexed_features', maxBins=754)
        label_converter = IndexToString(inputCol='prediction', outputCol='predicted_end_name', labels=label_indexer.labels)

        pipeline = Pipeline(stages=[label_indexer, feature_indexer, decision_tree, label_converter])

        train_set, test_set = data_with_features_column.randomSplit([0.8, 0.2])

        # Train the model
        model = pipeline.fit(train_set)
        # Make predictions
        predictions = model.transform(test_set)

        # Determine accuracy of the model
        evaluator = MulticlassClassificationEvaluator(labelCol='indexed_end_name', predictionCol='prediction', metricName='accuracy')
        accuracy = evaluator.evaluate(predictions)
        print(f'Accuracy: {accuracy}')

        # Display summary
        summary = model.stages[2]
        print(summary)