from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

# initialize spark session
spark = SparkSession.builder \
    .appName("Part1_Sequential_Execution") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("Spark session started.")
spark.sparkContext.setLogLevel("ERROR")

# load split datasets from mounted data path inside the container
df_master = spark.read.csv('/opt/spark/data/bank_master.csv', header=True, inferSchema=True, sep=";")
df_worker1 = spark.read.csv('/opt/spark/data/bank_worker1.csv', header=True, inferSchema=True, sep=";")
df_worker2 = spark.read.csv('/opt/spark/data/bank_worker2.csv', header=True, inferSchema=True, sep=";")

df = df_master.union(df_worker1).union(df_worker2)
print("Total rows:", df.count())

# convert categorical columns to numeric using string indexer
df_indexed = df

df_indexed = StringIndexer(inputCol="job", outputCol="job_idx", handleInvalid="keep").fit(df_indexed).transform(df_indexed)
df_indexed = StringIndexer(inputCol="marital", outputCol="marital_idx", handleInvalid="keep").fit(df_indexed).transform(df_indexed)
df_indexed = StringIndexer(inputCol="education", outputCol="education_idx", handleInvalid="keep").fit(df_indexed).transform(df_indexed)
df_indexed = StringIndexer(inputCol="default", outputCol="default_idx", handleInvalid="keep").fit(df_indexed).transform(df_indexed)
df_indexed = StringIndexer(inputCol="housing", outputCol="housing_idx", handleInvalid="keep").fit(df_indexed).transform(df_indexed)
df_indexed = StringIndexer(inputCol="loan", outputCol="loan_idx", handleInvalid="keep").fit(df_indexed).transform(df_indexed)
df_indexed = StringIndexer(inputCol="contact", outputCol="contact_idx", handleInvalid="keep").fit(df_indexed).transform(df_indexed)
df_indexed = StringIndexer(inputCol="month", outputCol="month_idx", handleInvalid="keep").fit(df_indexed).transform(df_indexed)
df_indexed = StringIndexer(inputCol="poutcome", outputCol="poutcome_idx", handleInvalid="keep").fit(df_indexed).transform(df_indexed)

label_indexer = StringIndexer(inputCol="y", outputCol="label", handleInvalid="keep")
df_indexed = label_indexer.fit(df_indexed).transform(df_indexed)

# select features and assemble into vector
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job_idx', 'marital_idx', 'education_idx', 'default_idx', 'housing_idx', 'loan_idx', 'contact_idx', 'month_idx', 'poutcome_idx']

all_features = numeric_cols + cat_cols

assembler = VectorAssembler(inputCols=all_features, outputCol="features")
final_data = assembler.transform(df_indexed).select("features", "label")

# split train and test data
train_data, test_data = final_data.randomSplit([0.7, 0.3], seed=42)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)

# train decision tree model (algorithm 1)
print("\nTraining Decision Tree model...")
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5)

start_time_dt = time.time()
dt_model = dt.fit(train_data)
training_time_dt = time.time() - start_time_dt

print("Decision Tree Training Time (seconds):", training_time_dt)

predictions_dt = dt_model.transform(test_data)
accuracy_dt = evaluator.evaluate(predictions_dt)
print("Decision Tree Accuracy:", accuracy_dt * 100)

# train random forest model (algorithm 2)
print("\nTraining Random Forest model...")
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20, maxDepth=5)

start_time_rf = time.time()
rf_model = rf.fit(train_data)
training_time_rf = time.time() - start_time_rf

print("Random Forest Training Time (seconds):", training_time_rf)

predictions_rf = rf_model.transform(test_data)
accuracy_rf = evaluator.evaluate(predictions_rf)
print("Random Forest Accuracy:", accuracy_rf * 100)

spark.stop()
print("Done!")