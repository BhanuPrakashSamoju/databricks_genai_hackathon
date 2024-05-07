# Databricks notebook source
# MAGIC %md
# MAGIC - Name: movies
# MAGIC - Purpose: Read the moies dataset from the ml-20m data sets & find the imdb ids of all these movies
# MAGIC - Author:`Bhanu Prakash`
# MAGIC - Dataset Citation: 
# MAGIC   > F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=<http://dx.doi.org/10.1145/2827872>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing libraries & setting variables

# COMMAND ----------

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

overwrite_schema = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the Input Data & Output Table name

# COMMAND ----------

movies_df = spark.read.csv(movies_path, header = True)
links_df = spark.read.csv(links_path, header = True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Exploration

# COMMAND ----------

display(movies_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Transformations & storing it into intermediate table

# COMMAND ----------

movie_links_df = movies_df.join(links_df, "movieId", "left").select(col("movieId").alias("movie_id"), "title", col("genres").alias("ml20_genres"), col("imdbId").alias("imdb_id"))
movie_links_df = movie_links_df.withColumn("ml20_genres", split(col("ml20_genres"), "\\|"))

# COMMAND ----------

movie_links_df.write.mode("overwrite").option("overwriteSchema", overwrite_schema).saveAsTable(movies_table_name)
