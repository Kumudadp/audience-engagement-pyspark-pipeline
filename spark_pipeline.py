"""
PySpark Audience Engagement Pipeline
-------------------------------------
A small end-to-end Spark pipeline over a synthetic media-viewership dataset
(~1.2M events), in the spirit of audience-measurement work:

  1. Load & clean raw event data
  2. Join events with show metadata
  3. Aggregate viewership by show / genre / region
  4. Engineer features and train a classifier to predict "high engagement"
     viewing sessions (watch_duration_sec >= 30 min) from device/region/genre

Run:
    spark-submit spark_pipeline.py
or:
    python spark_pipeline.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("AudienceEngagementPipeline")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
events = spark.read.csv("viewership_events.csv", header=True, inferSchema=True)
shows = spark.read.csv("shows_metadata.csv", header=True, inferSchema=True)

print(f"Raw event rows: {events.count():,}")

# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
events_clean = (
    events
    .dropna(subset=["watch_duration_sec"])          # drop rows with missing duration
    .dropDuplicates(["device_id", "show_id", "timestamp"])
    .filter(F.col("watch_duration_sec") > 0)
    .withColumn("timestamp", F.to_timestamp("timestamp"))
)

print(f"Rows after cleaning: {events_clean.count():,}")

# ---------------------------------------------------------------------------
# 3. Join with show metadata
# ---------------------------------------------------------------------------
enriched = events_clean.join(F.broadcast(shows), on="show_id", how="inner")

# ---------------------------------------------------------------------------
# 4. Aggregations — the kind of rollups an audience-measurement team needs
# ---------------------------------------------------------------------------
by_genre_region = (
    enriched.groupBy("genre", "region")
    .agg(
        F.count("*").alias("total_views"),
        F.sum("watch_duration_sec").alias("total_watch_seconds"),
        F.round(F.avg("watch_duration_sec"), 1).alias("avg_watch_seconds"),
    )
    .orderBy(F.desc("total_watch_seconds"))
)

print("\nTop 10 genre/region combinations by total watch time:")
by_genre_region.show(10, truncate=False)

top_shows = (
    enriched.groupBy("show_id", "genre", "network")
    .agg(F.sum("watch_duration_sec").alias("total_watch_seconds"),
         F.count("*").alias("total_views"))
    .orderBy(F.desc("total_watch_seconds"))
)

print("Top 10 shows by total watch time:")
top_shows.show(10, truncate=False)

by_genre_region.coalesce(1).write.mode("overwrite").option("header", True).csv("output/by_genre_region")
top_shows.coalesce(1).write.mode("overwrite").option("header", True).csv("output/top_shows")

# ---------------------------------------------------------------------------
# 5. Feature engineering + MLlib classification
#    Label: "high engagement" session = watch_duration_sec >= 1800 (30 min)
# ---------------------------------------------------------------------------
ml_df = enriched.withColumn(
    "high_engagement", (F.col("watch_duration_sec") >= 1800).cast("int")
)

categorical_cols = ["region", "age_group", "device_type", "genre", "network"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in categorical_cols]
assembler = VectorAssembler(
    inputCols=[f"{c}_vec" for c in categorical_cols] + ["is_original"],
    outputCol="features",
)
lr = LogisticRegression(featuresCol="features", labelCol="high_engagement", maxIter=20)

pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

evaluator = BinaryClassificationEvaluator(labelCol="high_engagement", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)

accuracy = predictions.filter(F.col("high_engagement") == F.col("prediction")).count() / predictions.count()

print(f"\nModel: Logistic Regression predicting 'high engagement' viewing sessions")
print(f"Test set size: {test_df.count():,}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")

spark.stop()
