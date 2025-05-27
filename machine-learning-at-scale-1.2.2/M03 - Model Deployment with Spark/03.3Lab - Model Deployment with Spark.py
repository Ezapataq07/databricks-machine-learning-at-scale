# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab - Model Deployment with Spark 
# MAGIC
# MAGIC In this lab, you will gain hands-on experience in deploying machine learning models using Apache Spark and optimizing query performance with Delta Lake. You will explore both single-node and distributed model deployment strategies, log models using MLflow for tracking, and apply advanced Delta Lake optimization techniques like **OPTIMIZE**, **VACUUM**, and **Liquid Clustering** to enhance performance and resource efficiency.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC
# MAGIC - **Task 1: Single-Node Model Deployment**  
# MAGIC   - Train a single-node **Gradient Boosted Tree (GBT)** model using **Scikit-learn**.
# MAGIC   - Log the trained model in **MLflow**.
# MAGIC   - Use **Spark UDFs** to perform parallelized inference on distributed data.
# MAGIC
# MAGIC - **Task 2: Distributed Model Deployment with Spark MLlib**  
# MAGIC   - Train a distributed **Gradient Boosted Tree (GBT)** model using Spark MLlib.
# MAGIC   - Log the model using **MLflow** and set a model alias for easy reference.
# MAGIC   - Perform distributed inference and save predictions to a Delta table.
# MAGIC
# MAGIC - **Task 3: Delta Lake Optimizations**  
# MAGIC   - Apply Delta Lake optimization strategies like **OPTIMIZE** to compact small files.
# MAGIC   - Perform **VACUUM** to clean up old, unused data files.
# MAGIC   - Apply **Z-ORDER Clustering** for faster reads on frequently queried columns.
# MAGIC   - Enable **Liquid Clustering** to incrementally optimize data layout.
# MAGIC
# MAGIC - **Task 4: Performance Comparison Before and After Optimization**  
# MAGIC   - Measure query performance before and after applying Delta Lake optimizations.
# MAGIC   - Compare the improvements in query speed and resource efficiency.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need a classic cluster running one of the following Databricks runtime(s): **16.3.x-cpu-ml-scala2.12**. **Do NOT use serverless compute to run this notebook**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -U optuna mlflow==2.9.2 delta-spark joblibspark pyspark==3.5.3
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup-lab"

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets.california_housing}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Pre-Steps: Data Preparation and Feature Engineering
# MAGIC Before you dive into model deployment, you need to prepare the California Housing dataset for model training and testing. This includes loading the dataset from Delta format, performing feature engineering, and splitting the data into training and testing sets.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Loading the California Housing Dataset
# MAGIC In this step, you'll load the California Housing dataset from Delta Lake and prepare it for both Spark and Scikit-learn model training. The dataset will be split into an 80/20 ratio for training and testing.

# COMMAND ----------

from sklearn.model_selection import train_test_split
from pyspark.ml.feature import VectorAssembler

# Load the California Housing dataset from Delta table
data_path = f"{DA.paths.datasets.california_housing}/data"
df = spark.read.format("delta").load(data_path)

# Define feature columns
feature_columns = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
]

# Assemble features into a vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_with_features = assembler.transform(df)

# Split the data into training and test sets (for SparkML Model Training)
train_df, test_df = df_with_features.randomSplit([0.8, 0.2], seed=42)

# Convert Spark DataFrame to Pandas DataFrame for single-node model training
train_pandas_df = train_df.select(feature_columns + ["label"]).toPandas()
test_pandas_df = test_df.select(feature_columns + ["label"]).toPandas()

# Split features and target variable for Scikit-learn training
X_train = train_pandas_df.drop(columns=["label"])
y_train = train_pandas_df["label"]
X_test = test_pandas_df.drop(columns=["label"])
y_test = test_pandas_df["label"]

# Display the first few rows of the dataset to verify data loading
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Single-Node Model Deployment with XGBoost and Spark UDFs
# MAGIC
# MAGIC In this task, you will train an **XGBRegressor** model using a **single-node training** approach. After training the model, you will log it using **MLflow** to enable tracking and version control. Additionally, you will apply **Spark UDFs** to perform parallelized inference on distributed data using the trained model. This approach demonstrates how to deploy a single-node model and leverage Spark's distributed computing capabilities for inference.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC 1. **Train an XGBRegressor model** on the provided training data.
# MAGIC 2. **Log the trained model** in MLflow for tracking and reproducibility.
# MAGIC 3. **Perform inference** on the test data to evaluate model predictions.
# MAGIC 4. **Apply Spark UDFs** to deploy the model and perform distributed, parallelized inference on the cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.1: Training and Logging the XGBoost Model  
# MAGIC
# MAGIC In this step, you will train an **XGBRegressor** model on the training dataset. After training, you will log the model in MLflow, allowing you to track the model's parameters and performance metrics for future reference.
# MAGIC
# MAGIC **Steps:**
# MAGIC - **Step 1:** Train an **XGBRegressor** model with specified parameters on the training data.
# MAGIC - **Step 2:** Log the trained model using MLflow to track the training run, including metrics like training and inference times.

# COMMAND ----------

# TODO
import xgboost as xgb
from xgboost import XGBRegressor
import mlflow
import mlflow.xgboost
import time
# Set the active experiment
mlflow.set_experiment(f"/Users/{DA.username}/lab_experiment")
# Train the XGBRegressor model
xgb_model = XGBRegressor(
    <FILL_IN>
)

# Measure training time
start_time = time.time()
xgb_model.fit(<FILL_IN>)
training_time_single_node = time.time() - start_time
print(f"Training time (single-node): {training_time_single_node:.2f} seconds")

# Log the XGBRegressor model with MLflow
with mlflow.start_run() as run:
    mlflow.xgboost.log_model(<FILL_IN>)
    run_id_single_node = run.info.run_id  # Capture the MLflow run ID for later use
    print(f"Model logged in run: {<FILL_IN>}")

    # Measure inference time
    start_time = time.time()
    y_pred = xgb_model.predict(X_test)
    inference_time_single_node = time.time() - start_time
    print(f"Inference time (single-node): {inference_time_single_node:.2f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.2: Performing Parallelized Inference Using Spark UDFs  
# MAGIC
# MAGIC In this step, you will use Spark UDFs to perform distributed inference by applying the trained **XGBRegressor** model across a Spark DataFrame. This approach allows you to scale the inference process across the cluster, taking advantage of Spark's parallel processing capabilities.
# MAGIC
# MAGIC **Steps:**
# MAGIC - **Step 1:** Load the trained XGBRegressor model from MLflow using its URI.
# MAGIC - **Step 2:** Define a Spark UDF that applies the loaded model to perform distributed inference on each data partition.
# MAGIC - **Step 3:** Display the results, including the original features, true labels, and predicted values, to evaluate the model's predictions on distributed data.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, struct
import mlflow.pyfunc

# Perform inference on the test set
y_pred = xgb_model.predict(<FILL_IN>)
# Load the logged model from MLflow
model_uri = f"runs:/{run_id_single_node}/xgboost_model"
predict_udf = mlflow.pyfunc.spark_udf(<FILL_IN>)

# Apply the UDF for inference on the distributed dataset
predictions = df_with_features.withColumn(<FILL_IN>)))

# Display the predictions along with original features and labels
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Distributed Model Deployment with Spark MLlib  
# MAGIC
# MAGIC In this task, you will train and deploy a **XGBRegressor** in a distributed environment using **Spark MLlib**. You will log the trained model with **MLflow** and perform distributed inference on the test dataset. Additionally, the predictions will be saved to a Delta table for further analysis. 
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC 1. **Train a distributed XGBRegressor model** using Spark MLlib.
# MAGIC 2. **Log the trained model** in MLflow for future use.
# MAGIC 3. **Perform distributed inference** on the test dataset using the trained model.
# MAGIC 4. **Save predictions** to a Delta table in Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.1: Training and Logging the XGBRegressor Model
# MAGIC
# MAGIC In this task, you will train an **XGBRegressor** model using the distributed Spark DataFrame. By preparing the features within Spark and converting to Pandas, you will leverage the power of both Spark for data processing and XGBoost for model training. After training, you will log the model in MLflow for version control and future use.
# MAGIC
# MAGIC **Steps:**
# MAGIC - **Step 1:** Prepare the features using **VectorAssembler** to transform the data for model training.
# MAGIC - **Step 2:** Convert the training data into a Pandas DataFrame and train the **XGBRegressor** model.
# MAGIC - **Step 3:** Log the trained model in **MLflow**, including its signature, to enable model tracking and reproducibility.
# MAGIC - **Step 4:** Measure and print both the training and inference times to assess model performance.
# MAGIC - **Step 5:** Save the trained model's URI in MLflow for easy reference in subsequent tasks.

# COMMAND ----------

# Distributed Model Deployment with XGBRegressor using Spark MLlib
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import struct
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import xgboost as xgb
from xgboost import XGBRegressor
import time
import pandas as pd

# Set the MLflow registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Create a VectorAssembler to prepare the features
assembler = VectorAssembler(<FILL_IN>)

# Assemble the features in Spark DataFrame
df_with_features = assembler.transform(df)

# Split the data into training and test sets
train_df, test_df = df_with_features.randomSplit([0.8, 0.2], seed=42)

# Prepare the training data as Pandas DataFrame for XGBoost
X_train = train_df.select(feature_columns).toPandas()
y_train = train_df.select("label").toPandas()
X_test = test_df.select(feature_columns).toPandas()
y_test = test_df.select("label").toPandas()

# Create the XGBRegressor model
xgb_regressor = XGBRegressor(
    <FILL_IN>
)

# Measure training time
start_time = time.time()
xgb_regressor.fit(X_train, y_train)
training_time_distributed = time.time() - start_time
print(f"Training time (distributed): {training_time_distributed} seconds")

# Log the XGBRegressor model with MLflow
with mlflow.start_run() as run:
    # Log the XGBRegressor model and infer its signature
    signature = infer_signature(<FILL_IN>)
    mlflow.xgboost.log_model(<FILL_IN>)
    run_id_distributed = run.info.run_id

# Measure inference time
start_time = time.time()
y_pred = xgb_regressor.predict(<FILL_IN>)
inference_time_distributed = time.time() - start_time
print(f"Inference time (distributed): {inference_time_distributed} seconds")

# Save the trained model's URI for inference
model_uri = f"runs:/{run_id_distributed}/xgboost_model"
print(f"Model saved at {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.2: Performing Distributed Inference and Saving Predictions to Delta Table
# MAGIC
# MAGIC In this step, you will apply the trained **Gradient Boosted Tree (GBT)** model to the test dataset and perform distributed inference across the Spark cluster. The predictions will be saved to a Delta table in Unity Catalog for further analysis.
# MAGIC
# MAGIC **Steps:**
# MAGIC - **Step 1:** Convert the test dataset to a Pandas DataFrame and use the trained **XGBRegressor** model to perform inference.
# MAGIC - **Step 2:** Combine the original test dataset with the predicted values and convert the results back to a Spark DataFrame.
# MAGIC - **Step 3:** Save the predictions to a Delta table for future analysis.
# MAGIC - **Step 4:** Explore the results by displaying the predictions alongside the actual labels, allowing for a comparison of the model's performance.

# COMMAND ----------

# Perform inference on the test dataset using the distributed XGBRegressor model
import pandas as pd
from pyspark.sql import functions as F

# Convert the test dataset to a Pandas DataFrame for prediction
X_test = test_df.select(feature_columns).toPandas()

# Perform inference using the trained XGBRegressor model
y_pred = xgb_regressor.predict(X_test)

# Convert predictions to a Pandas DataFrame
predictions_pd = <FILL_IN>

# Combine the original test DataFrame with the predictions
test_df_with_preds = test_df.toPandas()  # Convert test_df to Pandas DataFrame
test_df_with_preds["predicted_label"] = <FILL_IN>

# Convert the combined DataFrame back to a Spark DataFrame
predictions_spark_df =<FILL_IN>

# Display predictions
display(predictions_spark_df.select(*feature_columns, "label", "predicted_label"))

# Save predictions to a Delta table in Unity Catalog
predictions_spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.distributed_xgb_predictions_table")

# COMMAND ----------

# MAGIC %md
# MAGIC **Further Exploration:**  
# MAGIC
# MAGIC For more information on model tuning and deployment using Spark MLlib, refer to the [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Delta Lake Optimization Strategies
# MAGIC
# MAGIC In this section, you will explore how to optimize query performance and storage efficiency using Delta Lake features like **OPTIMIZE**, **VACUUM**, and **Liquid Clustering**. These optimization techniques will help compact data files, clean up unused files, and enhance query performance, especially for frequently accessed data.
# MAGIC
# MAGIC **Key Delta Lake Optimization Features**:
# MAGIC - **OPTIMIZE**: Compacts small files into larger ones, improving query read performance.
# MAGIC - **VACUUM**: Removes old, unused data files, optimizing storage and ensuring a clean data state.
# MAGIC - **Z-ORDER Clustering**: Clusters data based on columns frequently used in queries, speeding up read times by organizing the data more efficiently.
# MAGIC - **Liquid Clustering**: Automatically adjusts clustering over time based on query patterns for optimal performance.
# MAGIC
# MAGIC **Task Outline:**
# MAGIC
# MAGIC - **Step 1:** Apply **OPTIMIZE** to compact small files in the Delta table.
# MAGIC - **Step 2:** Apply **VACUUM** to clean up old files and reclaim storage.
# MAGIC - **Step 3:** Enable **Liquid Clustering** for automatic optimization.
# MAGIC - **Step 4:** Measure query performance before and after optimization.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Enable Predictive Optimization for Your Account
# MAGIC You must enable predictive optimization at the account level. You can then enable or disable predictive optimization at the catalog and schema levels.
# MAGIC
# MAGIC An account admin must complete the following steps to enable predictive optimization for all metastores in an account:
# MAGIC
# MAGIC * **Step 1:** Access the [Accounts Console](https://accounts.cloud.databricks.com/login).
# MAGIC
# MAGIC * **Step 2:** Navigate to **Settings**, then **Feature enablement**.
# MAGIC
# MAGIC * **Step 3:** Select **Enabled** next to **Predictive optimization**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.1: Applying OPTIMIZE to the Delta Table  
# MAGIC
# MAGIC In this task, you will apply **OPTIMIZE** on the Delta table to compact small files into larger ones. This operation reduces the overhead of reading many small files during queries, improving the overall performance of data reads.
# MAGIC
# MAGIC **Steps:**
# MAGIC - **Step 1:** Run the `OPTIMIZE` command on your Delta table.

# COMMAND ----------

# OPTIMIZE the Delta table to compact small files
spark.sql(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.2: Applying VACUUM to Clean Up Old Files
# MAGIC
# MAGIC The **VACUUM** operation removes old, unused data files from the Delta table that are no longer referenced by the latest version of the table. This helps reclaim storage space and ensures that only the necessary data files are retained.
# MAGIC
# MAGIC **Step:**  
# MAGIC - **Step 1:** Run the VACUUM operation to remove old, unused data files from your Delta table.

# COMMAND ----------

# Disable the retention duration check and apply VACUUM
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")

# Apply VACUUM to clean up old files and reclaim storage space
display(spark.sql(f"""
<FILL_IN>
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC For more information on distributed machine learning with Spark, visit [Spark MLlib's Guide](https://spark.apache.org/docs/latest/ml-guide.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.3: Enabling Liquid Clustering
# MAGIC
# MAGIC **Liquid Clustering** is an advanced Delta Lake feature that dynamically clusters your data based on the query patterns it observes over time. This ensures that frequently queried columns are automatically clustered, leading to faster reads.
# MAGIC
# MAGIC **Steps:**  
# MAGIC - **Step 1:** Enable **Liquid Clustering** on your Delta table by clustering it based on frequently queried columns.
# MAGIC - **Step 2:** Query the Delta table's history to check the clustering progress after enabling Liquid Clustering.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Enable Liquid Clustering on the Delta table, clustering by frequently queried columns
# MAGIC ALTER TABLE <FILL_IN>
# MAGIC CLUSTER BY (<FILL_IN>);
# MAGIC
# MAGIC -- Display the improvement after applying liquid clustering
# MAGIC DESCRIBE <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.4: Measuring Query Performance Before and After Optimization  
# MAGIC
# MAGIC To truly understand the impact of these optimizations, we will measure the query performance before and after applying the Delta Lake optimizations. This will help you observe the improvements in query speed and resource efficiency.
# MAGIC
# MAGIC **Steps:**  
# MAGIC - **Step 1:** Run the same query on your Delta table before and after applying the optimizations.
# MAGIC - **Step 2:** Measure the time taken for the query before optimization.
# MAGIC - **Step 3:** Apply **OPTIMIZE** and **VACUUM**.
# MAGIC - **Step 4:** Measure the query performance again and compare the improvement in speed.
# MAGIC - **Step 5:** Display the performance improvement percentage and discuss how the Delta Lake optimizations affected the query time.

# COMMAND ----------

from time import time

# Define the table and query
table_name = f"{DA.catalog_name}.{DA.schema_name}.distributed_dt_predictions_table"

# Define the query you will run before and after optimizations
query = <FILL_IN>

# Measure query performance before optimization
start_time_before = time()
df_before = spark.sql(query)
time_before = time() - start_time_before
print(f"Query Execution Time Before Optimization: {time_before:.2f} seconds")

# Apply Delta Lake optimizations (OPTIMIZE and VACUUM were already applied earlier)

# Measure query performance after optimization
start_time_after = time()
df_after = spark.sql(query)
time_after = <FILL_IN>
print(f"Query Execution Time After Optimization: {time_after:.2f} seconds")

# Calculate the performance improvement
performance_improvement = <FILL_IN> / time_before * 100
print(f"Performance improvement after optimizations: {performance_improvement:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you successfully explored various methods for deploying machine learning models using Spark, including single-node and distributed deployment strategies. You gained hands-on experience with:
# MAGIC
# MAGIC - Preparing data for model training and deployment using the Wine Quality dataset.
# MAGIC - Training and deploying a single-node machine learning model using Scikit-learn, followed by leveraging **Spark UDFs** for parallelized inference on distributed data.
# MAGIC - Training and deploying a distributed machine learning model using **Spark MLlib**, followed by performing distributed inference and logging the model with MLflow.
# MAGIC - Applying **Delta Lake optimizations** such as **OPTIMIZE**, **VACUUM**, and **Liquid Clustering** to enhance query performance, reduce storage costs, and improve overall resource efficiency.
# MAGIC
# MAGIC Finally, you compared the query performance before and after these optimizations, observing how Delta Lake's features significantly improved the speed and efficiency of your model serving pipeline.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
