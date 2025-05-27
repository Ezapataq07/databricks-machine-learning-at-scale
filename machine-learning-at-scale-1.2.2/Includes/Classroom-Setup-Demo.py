# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(precision=2)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# COMMAND ----------

import pandas as pd
@DBAcademyHelper.add_init
def create_large_wine_quality_table(self):
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE SCHEMA {DA.schema_name}")
    
    # Load the original wine dataset
    data_path = f"{DA.paths.datasets.wine_quality}/data"
    df = spark.read.format("delta").load(data_path)

    # Function to create a larger dataset for demonstration purposes
    def generate_large_wine_dataset(df, num_copies=100):
        pandas_df = df.toPandas()  # Convert to Pandas for duplication
        large_df = pd.concat([pandas_df.sample(frac=1).reset_index(drop=True)] * num_copies, ignore_index=True)  # Duplicate and shuffle
        return large_df

    # Convert back to Spark DataFrame and save as Delta table
    large_wine_df = spark.createDataFrame(generate_large_wine_dataset(df, num_copies=100))
    output_delta_table = f"{DA.paths.working_dir}/v01/large_wine_quality_delta"
    large_wine_df.write.format("delta").mode("overwrite").save(output_delta_table)

    print(f"Created large Delta table at {output_delta_table}")

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()
