# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Optimization Strategies with Spark and Delta Lake
# MAGIC
# MAGIC In this demo, you will learn how to use Delta Lake’s optimization techniques to enhance the performance of machine learning model deployments on Apache Spark. This demonstration will cover key Delta Lake features like `OPTIMIZE`, `VACUUM`, and advanced clustering techniques to improve query performance, reduce storage costs, and maintain data integrity.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC _By the end of this demo, you will be able to:_
# MAGIC
# MAGIC 1. **Understand the impact of Delta Lake optimizations** on large-scale datasets used in Spark-based model deployments.
# MAGIC 2. **Perform core optimizations** like compaction and cleanup using Delta Lake’s `OPTIMIZE` and `VACUUM` commands.
# MAGIC 3. **Implement Z-Ordering and Liquid Clustering** to optimize the data layout for faster query performance.
# MAGIC 4. **Analyze and compare query performance** before and after applying optimizations.
# MAGIC 5. **Understand the impact of optimizations on real-time data access and inference performance.**

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
# MAGIC %pip install databricks-sdk --upgrade
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.2

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
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimization Strategies with Spark and Delta Lake
# MAGIC
# MAGIC In this section, we will explore optimization techniques provided by Delta Lake to enhance performance, focusing on improving query speed, reducing storage costs, and maintaining data integrity for model serving.
# MAGIC
# MAGIC Delta Lake provides several automatic and manual optimization techniques that ensure fast and reliable querying over large datasets in distributed systems like Spark.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable Predictive Optimization for Your Account
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
# MAGIC ### Key Optimization Features in Delta Lake
# MAGIC
# MAGIC - **OPTIMIZE**: Coalesce small files into larger ones, improving read performance by reducing the overhead of managing small files.
# MAGIC - **VACUUM**: Clean up old data files that are no longer referenced, ensuring that the storage system is efficient and up-to-date.
# MAGIC - **Z-ORDER Clustering**: Optimize the data layout based on the clustering of specific columns, significantly speeding up queries.
# MAGIC - **Liquid Clustering**: For more advanced automatic clustering, which incrementally clusters new data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Steps for Running Delta Lake Optimizations:
# MAGIC
# MAGIC - **Run OPTIMIZE on Delta Tables**
# MAGIC
# MAGIC This step will compact smaller files into larger ones for faster query performance. Delta Lake helps with incremental clustering on optimized tables.

# COMMAND ----------

# OPTIMIZE the Delta table to compact small files
spark.sql(f"""
OPTIMIZE {DA.catalog_name}.{DA.schema_name}.distributed_predictions_table
""")

# COMMAND ----------

# MAGIC %md
# MAGIC - **Run VACUUM on Delta Tables**
# MAGIC
# MAGIC VACUUM is used to remove files no longer referenced by Delta tables, such as old versions of data that were overwritten or deleted.

# COMMAND ----------

# Disable the retention duration check
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")

# VACUUM to remove old files no longer referenced by the table
spark.sql(f"""
VACUUM {DA.catalog_name}.{DA.schema_name}.distributed_predictions_table RETAIN 0 HOURS
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Advanced Optimization Strategies:
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####Liquid Clustering
# MAGIC Liquid Clustering incrementally clusters new data for optimal query performance, automatically selecting and updating clustering keys based on historical query workloads.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE distributed_predictions_table
# MAGIC CLUSTER BY (quality, alcohol);

# COMMAND ----------

# MAGIC %md
# MAGIC Once enabled, predictive optimization intelligently selects the best clustering keys for performance improvement, monitoring and adjusting clustering columns over time based on query patterns.
# MAGIC
# MAGIC For more information, refer to the [Delta Lake Liquid Clustering Documentation](https://docs.databricks.com/en/delta/clustering.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ####Z-Ordering:
# MAGIC
# MAGIC Z-Ordering clusters the data for faster query performance by optimizing the data layout on disk, reducing I/O operations during query execution. This is particularly useful if queries frequently filter on specific columns.
# MAGIC
# MAGIC **🚨Note: You will receive an error in the next cell because the table was created using liquid clustering. The next cell is only for code demonstration.**

# COMMAND ----------

# Z-Ordering the Delta table based on columns used in queries (e.g., `quality`)
spark.sql(f"""
OPTIMIZE {DA.catalog_name}.{DA.schema_name}.distributed_predictions_table
ZORDER BY (quality)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Comparison of Query Performance Before and After Optimization
# MAGIC
# MAGIC Before applying the above optimizations, queries may take longer to execute due to the need to scan multiple small files or non-optimized layouts. After performing OPTIMIZE and VACUUM, expect:
# MAGIC
# MAGIC - 2x faster query performance.
# MAGIC - Up to 50% savings in storage due to file compaction and cleanup of old data files.
# MAGIC - No manual effort required for ongoing optimizations when predictive optimization is enabled at the account level.
# MAGIC
# MAGIC **Steps to Compare Query Performance:**
# MAGIC
# MAGIC - **Step 1:** Measure Query Performance Before Optimization
# MAGIC - **Step 2:** Apply Delta Lake Optimizations
# MAGIC - **Step 3:** Measure Query Performance After Optimization
# MAGIC   - Run the same query again after optimizations and measure the time taken.
# MAGIC - **Step 4:** Measure the time taken and compare the results.
# MAGIC   - Calculate the improvement in query performance.

# COMMAND ----------

from pyspark.sql import functions as F
from time import time

# Disable IO cache before running any query to ensure accurate performance measurement
spark.conf.set('spark.databricks.io.cache.enabled', False)
# Define the table and columns you will query
table_name = f"{DA.catalog_name}.{DA.schema_name}.predictions_before_optimize"

# Step 1: Run a query on the Delta table before optimization
query = f"""
SELECT quality, COUNT(*)
FROM {table_name}
GROUP BY quality
"""

# Start the timer for the query before optimization
start_time_before_optimize = time()
df_before_optimize = spark.sql(query)
display(df_before_optimize)  # Running the query and displaying the results

# Calculate the time taken for the query before optimization
time_before_optimize = time() - start_time_before_optimize
print(f"Query Execution Time Before Optimization: {time_before_optimize:.6f} seconds")

# Step 2: Perform Delta Lake Optimizations
print("Applying Delta Lake Optimizations...")

# Optimize to compact small files
spark.sql(f"OPTIMIZE {table_name}")

# Vacuum to remove old files
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
spark.sql(f"VACUUM {table_name} RETAIN 0 HOURS")

# Disable IO cache
spark.conf.set('spark.databricks.io.cache.enabled', False)

# Step 3: Run the same query again after optimization
start_time_after_optimize = time()
df_after_optimize = spark.sql(query)
display(df_after_optimize)  # Running the query and displaying the results

# Calculate the time taken for the query after optimization
time_after_optimize = time() - start_time_after_optimize
print(f"Query Execution Time After Optimization: {time_after_optimize:.6f} seconds")

# Step 4: Compare the results

# Calculate the improvement in query performance
performance_improvement = (time_before_optimize - time_after_optimize) / time_before_optimize * 100
print(f"Performance Improvement: {performance_improvement:.2f}%")

# Compare storage size before and after optimization
history_df = spark.sql(f"DESCRIBE HISTORY {table_name}")
display(history_df)

# Display the performance comparison results
performance_comparison = {
    "Metric": ["Execution Time (Before)", "Execution Time (After)", "Performance Improvement"],
    "Value": [f"{time_before_optimize:.2f} seconds", f"{time_after_optimize:.2f} seconds", f"{performance_improvement:.2f}%"]
}

import pandas as pd
performance_df = pd.DataFrame(performance_comparison)
display(performance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC For more information on Delta Lake optimizations, refer to the [Delta Lake Optimizations Documentation](https://docs.databricks.com/delta/optimizations.html).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Impact of Delta Lake Optimizations on Model Serving
# MAGIC By applying these Delta Lake optimizations, the model serving pipeline becomes faster and more resource-efficient, which leads to:
# MAGIC
# MAGIC - **Reduced Latency:** Quicker access to the relevant data, ensuring that model predictions are delivered with minimal delay.
# MAGIC
# MAGIC - **Improved Query Performance:** Optimized data layout and clustering lead to faster data access during inference, especially for real-time or high-frequency serving scenarios.
# MAGIC
# MAGIC - **Lower Resource Consumption:** With fewer small files and cleaner storage, the system uses fewer resources to retrieve data, resulting in better overall efficiency.
# MAGIC
# MAGIC Incorporating these optimizations ensures that your model serving pipeline is not only fast but also scalable, capable of handling real-time demands without compromising performance.

# COMMAND ----------

# Import necessary libraries
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag
from mlflow.models.signature import infer_signature

# Initialize Databricks Workspace client
w = WorkspaceClient()

# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client
client = mlflow.MlflowClient()

# Define the model name
model_name = f"{DA.catalog_name}.{DA.schema_name}.SparkML" 

# Set the experiment
experiment_name = f"/Users/{DA.username}/experiments_SparkML"
mlflow.set_experiment(experiment_name)

# Check if the model exists
try:
    model_version_champion = client.get_model_version_by_alias(
        name=model_name, 
        alias="champion"
    ).version
    print(f"Champion model version: {model_version_champion}")
except mlflow.exceptions.RestException:
    print(f"Model {model_name} does not exist. Registering a new model.")
    
    # Create a sample model (replace with your actual model training code)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load iris dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Train a RandomForest model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model with MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model-artifacts", 
            signature=signature
        )
        run_id = run.info.run_id

    # Register the model
    model_uri = f"runs:/{run_id}/model-artifacts"
    model_details = mlflow.register_model(model_uri, model_name)
    model_version_champion = model_details.version
    print(f"Registered new model version: {model_version_champion}")

# Define the endpoint configuration
endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": model_version_champion,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        }
    ]
}

# Create the endpoint configuration input from the dictionary
endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# Construct the endpoint name
endpoint_name = f"{DA.username}_SparkML_Demo_03"
endpoint_name = endpoint_name.replace(".", "-")
endpoint_name = endpoint_name.replace("@", "-")
endpoint_name = endpoint_name.replace("+", "_")
# Create the endpoint_name key/value pair to be passed on in the job configuration
dbutils.jobs.taskValues.set(key = "endpoint_name", value = endpoint_name)
print(f"Endpoint name: {endpoint_name}")

# Attempt to create or update the serving endpoint
try:
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=endpoint_config,
        tags=[EndpointTag.from_dict({"key": "db_academy", "value": "lab1_jobs_model"})]
    )
    print(f"Creating endpoint {endpoint_name} with models {model_name} versions {model_version_champion}")
except Exception as e:
    if "already exists" in e.args[0]:
        print(f"Endpoint with name {endpoint_name} already exists")
    else:
        raise(e)
    
# Display the endpoint URL
displayHTML(f'Our Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{endpoint_name}">Model Serving Endpoint')

# COMMAND ----------

# MAGIC %md
# MAGIC 🚨 **Deleting the Model Serving Endpoint**
# MAGIC
# MAGIC After completing the demo, it's important to clean up any resources that were created, including the model serving endpoint. Deleting the endpoint ensures that you're not consuming unnecessary resources, and helps maintain a clean workspace.

# COMMAND ----------

def delete_model_serving_endpoint(endpoint_name):
    w.serving_endpoints.delete(name=endpoint_name)
    print(endpoint_name, "endpoint is deleted!")
        
delete_model_serving_endpoint(endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Conclusion
# MAGIC
# MAGIC In this demo, we explored key Delta Lake optimization strategies to enhance the performance of machine learning workloads deployed on Apache Spark. By applying optimizations like **OPTIMIZE**, **VACUUM**, and **Z-ORDER clustering**, we demonstrated how to improve query execution times and reduce storage costs. Additionally, we introduced **Liquid Clustering** as an advanced technique for dynamically managing data layout based on query patterns.
# MAGIC
# MAGIC We also performed a performance comparison to quantify the improvements achieved by these optimizations, emphasizing the importance of efficient data layout and management in large-scale data systems. With these Delta Lake techniques, you can significantly reduce latency and optimize resource consumption, making your model serving pipeline more scalable and responsive to real-time demands.
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
