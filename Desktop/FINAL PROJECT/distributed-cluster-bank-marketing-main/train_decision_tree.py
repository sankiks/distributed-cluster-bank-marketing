from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

# 1. Initialize Spark session
spark = SparkSession.builder \
    .appName("DecisionTreeLab") \
    .getOrCreate()

print("Spark session started.")

# 2. Load dataset
file_path = 'bank-full.csv'
df = spark.read.csv(file_path, header=True, inferSchema=True, sep=";")

print("Total rows in dataset:", df.count())

# 3. Convert categorical columns to numeric using StringIndexer
# Doing it step by step like a student lab assignment
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

# Convert target column 'y' to label
label_indexer = StringIndexer(inputCol="y", outputCol="label", handleInvalid="keep")
df_indexed = label_indexer.fit(df_indexed).transform(df_indexed)

# 4. Select features for training
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job_idx', 'marital_idx', 'education_idx', 'default_idx', 'housing_idx', 'loan_idx', 'contact_idx', 'month_idx', 'poutcome_idx']

all_features = numeric_cols + cat_cols

# Combine features into a vector
assembler = VectorAssembler(inputCols=all_features, outputCol="features")
final_data = assembler.transform(df_indexed).select("features", "label")

# 5. Split dataset into train (70%) and test (30%)
train_data, test_data = final_data.randomSplit([0.7, 0.3], seed=42)

# 6. Train Decision Tree model and measure time
print("Training Decision Tree model...")
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5)

start_time = time.time()
model = dt.fit(train_data)
end_time = time.time() - start_time

print("Training time (seconds):", end_time)

# 7. Evaluate model accuracy
predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

print("Test Accuracy:", accuracy * 100)

# Stop spark session
spark.stop()