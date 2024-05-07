# Databricks notebook source
# MAGIC %md
# MAGIC - **Name: movies_info (Cleaned)**
# MAGIC - Purpose: Create a clean movies DB that is a combination of ml-20m dataset & IMDB dataset. The join is achieved through IMDB Ids
# MAGIC - Author:`Bhanu Prakash`
# MAGIC - Dataset Citation: 
# MAGIC   > F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=<http://dx.doi.org/10.1145/2827872>
# MAGIC   
# MAGIC   > Free IMDB data set from Bright Data=<https://brightdata.com/products/datasets/marketplace#all> 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing libraries & setting variables

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

overwrite_schema = True
movies_table_name = "gen_ai_hackathon.silver.movies_info"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Input data read

# COMMAND ----------

imdb_df = spark.read.table("gen_ai_hackathon.bronze.imdb_ratings_json")
movies_df = spark.read.table("gen_ai_hackathon.bronze.movies")
genome_tag_df = spark.read.table("gen_ai_hackathon.bronze.genome_tags")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Exploration

# COMMAND ----------

display(movies_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Transformations & storing it into final table

# COMMAND ----------

imdb_release_date = imdb_df.withColumn("date_of_release", split(col("details_release_date"), " ", 2)[0])
#.withColumn("place_of_release", split(col("details_release_date"), " ", 2)[1])
imdb_year_release = imdb_release_date.withColumn('release_year', element_at(split(col("date_of_release"), "/"), -1))
display(imdb_year_release)

# COMMAND ----------

imdb_ids_df = imdb_year_release.withColumn("id", regexp_replace(element_at(split(col("url"), "/"), -2), "^tt*", ""))
imdb_ids_df = imdb_ids_df.withColumnRenamed("title", "imdb_title")
#df_ids = df_ids.withColumn("id", col("id").cast(IntegerType()))
#display(df_ids.select("url", "id"))

# COMMAND ----------

movies_merged_df = movies_df.alias("ml20").join(imdb_ids_df.alias("imdb"), on=(col("ml20.imdb_id")==col("imdb.id")), how="left")
movies_genre_df = movies_merged_df.withColumn("genres", array_union(coalesce(col("ml20_genres"), array()), coalesce(col("genres"), array()))).drop("id", "ml20_genres")
display(movies_genre_df)

# COMMAND ----------

movies_genome_df = movies_genre_df.alias("m").join(genome_tag_df.alias("g"), on=(col("m.movie_id")==col("g.movie_id")), how = "left")

# COMMAND ----------

columns_list = ['m.movie_id', 'title', 'imdb_id', 'imdb_title', 'awards', "genome_tags", 'boxoffice_budget', 'credit', 'critics_review_count', 'details_also_known_as', 'details_countries_of_origin', 'details_filming_locations', 'details_language', 'details_official_site', 'details_production_companies', 'details_release_date', 'episode_count', 'featured_review', 'genres', 'imdb_rating', 'imdb_rating_count', 'media_type', 'popularity', 'presentation', 'review_count', 'review_rating', 'specs_aspect_ratio', 'specs_color', 'specs_sound_mix', 'storyline', 'top_cast', 'url']

final_movies_df = movies_genome_df.select(columns_list)

# COMMAND ----------

final_movies_df.write.mode("overwrite").option("overwriteSchema", overwrite_schema).saveAsTable(movies_table_name)
