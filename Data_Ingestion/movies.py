# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

dbutils.widgets.text(name="movies_path", label="movies_path", defaultValue= "/Volumes/gen_ai_hackathon/bronze/imbd_ratings/ml-20m/movies.csv")
dbutils.widgets.text(name="links_path", label="links_path", defaultValue="/Volumes/gen_ai_hackathon/bronze/imbd_ratings/ml-20m/links.csv")
dbutils.widgets.text(name="movies_table_name", label="movies_table_name", defaultValue="gen_ai_hackathon.bronze.movies")

# COMMAND ----------

movies_path = dbutils.widgets.get("movies_path")
links_path = dbutils.widgets.get("links_path")
movies_table_name = dbutils.widgets.get("movies_table_name")

# COMMAND ----------

movies_df = spark.read.csv(movies_path, header = True)
links_df = spark.read.csv(links_path, header = True)

# COMMAND ----------

overwrite_schema = True

# COMMAND ----------

display(movies_df.count())

# COMMAND ----------

movie_links_df = movies_df.join(links_df, "movieId", "left").select(col("movieId").alias("movie_id"), "title", col("genres").alias("ml20_genres"), col("imdbId").alias("imdb_id"))
movie_links_df = movie_links_df.withColumn("ml20_genres", split(col("ml20_genres"), "\\|"))

# COMMAND ----------

movie_links_df.write.mode("overwrite").option("overwriteSchema", overwrite_schema).saveAsTable(movies_table_name)
