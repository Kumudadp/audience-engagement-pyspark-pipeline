# Audience Engagement Pipeline (PySpark)

An end-to-end Apache Spark pipeline built on a synthetic media-viewership
dataset (~1.2M events), modeled after the kind of audience-measurement data
used in TV/streaming analytics.

## What it does

1. **Load** ~1.2M raw viewership events + a 500-row show metadata table
2. **Clean** — drop nulls/duplicates, filter invalid rows, cast timestamps
3. **Join** events with show metadata (broadcast join on the small side)
4. **Aggregate** total watch time and views by genre/region and by show,
   written out as CSV
5. **Feature engineer + classify** — encode categorical features
   (region, age group, device type, genre, network) and train a Logistic
   Regression model in Spark MLlib to predict "high engagement" sessions
   (watch time ≥ 30 minutes), evaluated with AUC-ROC and accuracy

## Why

Built to get hands-on experience with distributed data processing (joins,
aggregations, and ML at scale) beyond coursework, in an audience-measurement
domain relevant to media analytics roles.

## Stack

Python, PySpark (DataFrame API, Spark MLlib), CSV I/O

## Run it

```bash
pip install pyspark
python generate_data.py     # creates the synthetic dataset (~70MB)
python spark_pipeline.py    # runs the full pipeline
```

Aggregated output lands in `output/by_genre_region/` and `output/top_shows/`.

## Notes

The dataset is synthetically generated (see `generate_data.py`) so the
data has no real-world engagement signal — this project is a demonstration
of the Spark pipeline mechanics (cleaning, joining, aggregating, and
training a model at scale), not a claim about actual viewership patterns.
