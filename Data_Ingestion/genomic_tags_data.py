# Databricks notebook source
# MAGIC %md
# MAGIC - Name: genomic_tags_data
# MAGIC - Purpose: Read the Genomic tags from the ml-20m data sets & take all the tags for each movie which has more than 0.2 relevance score
# MAGIC - Author:`Bhanu Prakash`

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

dbutils.widgets.text(name="genome_scores_path", label="genomic_scores_path", defaultValue= "/Volumes/gen_ai_hackathon/bronze/imbd_ratings/ml-20m/genome-scores.csv")
dbutils.widgets.text(name="genome_tags_path", label="genomic_tags_path", defaultValue="/Volumes/gen_ai_hackathon/bronze/imbd_ratings/ml-20m/genome-tags.csv")
dbutils.widgets.text(name="genome_score_threshold", label="genome_relevance_threshold", defaultValue="0.2")
dbutils.widgets.text(name="genome_table_name", label="genomic_tags_table_name", defaultValue="gen_ai_hackathon.bronze.genome_tags")

# COMMAND ----------

genome_scores_path = dbutils.widgets.get("genome_scores_path")
genome_tags_path = dbutils.widgets.get("genome_tags_path")
genome_score_threshold = float(dbutils.widgets.get("genome_score_threshold"))
genome_table_name = dbutils.widgets.get("genome_table_name")

# COMMAND ----------

overwrite_schema = False

# COMMAND ----------

genome_scores_df = spark.read.csv(genome_scores_path, header = True)
genome_tags_df = spark.read.csv(genome_tags_path, header = True)

# COMMAND ----------

display(genome_scores_df.count())

# COMMAND ----------

display(genome_scores_df.filter(genome_scores_df.relevance > genome_score_threshold).count())

# COMMAND ----------

threshold_genome_df = genome_scores_df.filter(genome_scores_df.relevance > genome_score_threshold)
genome_table_df = threshold_genome_df.alias("g").join(genome_tags_df.alias("t"),"tagId", "inner").select(col("movieId").alias("movie_id"), col("tagId").alias("tag_id"), "tag", "relevance")

# COMMAND ----------

display(genome_table_df.groupBy("movie_id").agg(count("tag_id")).orderBy(count("tag_id"), ascending=False))

# COMMAND ----------

# Transform the dataframe
df_transformed = genome_table_df.groupBy("movie_id").agg(
    to_json(collect_list(struct("tag", "relevance"))).alias("genome_tags")
)
df_transformed.display()

# COMMAND ----------

df_transformed.write.mode("overwrite").option("overwriteSchema", overwrite_schema).saveAsTable(genome_table_name)
