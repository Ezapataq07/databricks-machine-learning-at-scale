# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Model Deployment with Spark
# MAGIC
# MAGIC In this demo, you will learn how to deploy machine learning models using Apache Spark. We will explore different deployment strategies, comparing single-node and distributed Spark ML models, and performing inference using Spark UDFs to scale predictions efficiently across large datasets.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC _By the end of this demo, you will be able to:_
# MAGIC
# MAGIC 1. **Understand deployment methods** for machine learning models using Spark, including single-node and distributed model deployments.
# MAGIC 2. **Compare single-node vs. distributed model deployments** using Spark DataFrames and Spark MLlib.
# MAGIC 3. **Perform parallelized inference** using `spark_udf` on distributed data to efficiently handle large datasets.

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
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup-Demo"

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
print(f"Dataset Location:  {DA.paths.datasets.wine_quality}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-Steps: Data Preparation
# MAGIC
# MAGIC Before diving into model deployment, we first need to prepare our dataset for both training and inference. In this demo, we will use the **Wine Quality** dataset. The data will be loaded from a Delta table, and we will perform necessary manipulations such as assembling features and splitting the dataset into training and test sets.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC 1. **Load the Wine Quality dataset** from Delta Lake.
# MAGIC 2. **Assemble features** into a vector using `VectorAssembler` for Spark ML training.
# MAGIC 3. **Split the dataset** into training (80%) and testing (20%) sets for both Spark ML and single-node Scikit-learn training.
# MAGIC 4. **Convert the Spark DataFrame** into Pandas DataFrame for single-node model training.

# COMMAND ----------

from sklearn.model_selection import train_test_split
from pyspark.ml.feature import VectorAssembler
# Load the large Wine Quality dataset from the new Delta table
data_path = f"{DA.paths.working_dir}/v01/large_wine_quality_delta"
df = spark.read.format("delta").load(data_path)
# Define feature columns
feature_columns = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", 
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", 
    "pH", "sulphates", "alcohol"
]
# Assemble features into a vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_with_features = assembler.transform(df)

# Split the data into training and test sets (for SparkML Model Training)
train_df, test_df = df_with_features.randomSplit([0.8, 0.2], seed=42)
# Convert Spark DataFrame to Pandas DataFrame for single-node model training
train_pandas_df = train_df.select(feature_columns + ["quality"]).toPandas()
test_pandas_df = test_df.select(feature_columns + ["quality"]).toPandas()

# Split features and target variable
X_train = train_pandas_df.drop(columns=["quality"])
y_train = train_pandas_df["quality"]
X_test = test_pandas_df.drop(columns=["quality"])
y_test = test_pandas_df["quality"]

# Display the first few rows of the DataFrame to ensure correctness
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Methods to Deploy Machine Learning Models in Spark
# MAGIC
# MAGIC Spark provides multiple approaches for deploying machine learning models, making it a powerful and flexible framework for both data processing and model serving. Depending on the size of your dataset and the computational requirements, you can choose from two main approaches for model deployment:
# MAGIC
# MAGIC - **Single-node model deployment:** The model is trained and executed on a single machine. Even though the model is single-node, Spark can still distribute the data for processing. This approach works well for smaller datasets and simpler applications.
# MAGIC   
# MAGIC - **Distributed model deployment:** This approach fully leverages Spark's distributed architecture to handle large datasets and complex computations. Models are deployed across multiple nodes in a cluster, making it suitable for large-scale machine learning tasks.
# MAGIC
# MAGIC For more information on Spark MLlib and its capabilities for distributed machine learning, refer to the [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why Use Spark for Model Deployment?
# MAGIC Apache Spark provides a robust platform for deploying machine learning models at scale. Spark’s ability to process distributed data and its seamless integration with Spark MLlib, Scikit-learn, and other ML libraries makes it a versatile tool for both small and large datasets. By the end of this demo, you’ll have a solid understanding of how to choose between single-node and distributed model deployment and how Delta Lake optimizations can enhance your model's performance.
# MAGIC
# MAGIC **Key Benefits:**
# MAGIC - **Scalability:** Spark is designed to handle massive datasets, scaling seamlessly across clusters. This makes it ideal for training and deploying models on large data.
# MAGIC - **Versatility:** Spark integrates with various machine learning libraries, including Scikit-learn, Spark MLlib, XGBoost, and more. This allows you to choose the best tools for training your models while benefiting from Spark’s distributed architecture.
# MAGIC - **Efficiency:** With its in-memory processing and parallelized operations, Spark ensures efficient data processing and model serving. This is useful for both batch and real-time inference scenarios.
# MAGIC
# MAGIC By deploying machine learning models on Spark, you can achieve high-performance, scalable model serving that meets both real-time and batch processing needs.
# MAGIC
# MAGIC **For further exploration:**  
# MAGIC - [Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html): Learn more about Spark’s native machine learning library.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Comparing Single-node and Distributed Model Deployments in Spark
# MAGIC In this section, you will explore and compare two different methods for deploying a machine learning model in Spark using the same model configurations in **Part 1**. We will specifically focus on comparing the training and inference time for deploying the **Decision Tree model** in both single-node and distributed configurations.
# MAGIC
# MAGIC - **Single-node model deployment**: We will use the **DecisionTreeRegressor** model from `Scikit-learn`, which runs on a single machine to perform training and inference.
# MAGIC - **Distributed model deployment**: We will use the **DecisionTreeRegressor** model from Spark MLlib, which performs distributed training and inference across multiple nodes in a Spark cluster.
# MAGIC
# MAGIC Both methods will be applied to the same dataset. However, the deployment approaches will differ significantly in scalability and computational performance. The single-node model executes training and inference on a single machine, while the distributed model leverages Spark’s parallel processing capabilities across a cluster.
# MAGIC
# MAGIC The objective of this comparison is to evaluate the execution efficiency of both configurations in terms of training and inference time.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single-node Model Deployment with Spark DataFrames
# MAGIC In this section, we will perform single-node model deployment using the **DecisionTreeRegressor** from Scikit-learn. While the model training and inference are executed on a single machine, Spark’s distributed data capabilities will still be leveraged to handle and distribute the data efficiently.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC 1. **Train a DecisionTreeRegressor model** using Scikit-learn. Although the data is distributed across the cluster, the training will occur on a single machine.
# MAGIC 2. **Perform inference on the test data** using the trained model.
# MAGIC 3. **Log the model with MLflow** for future reference and tracking.
# MAGIC 4. **Evaluate the model’s performance** by comparing the time taken for training and inference.

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor
import mlflow
import time

# Set the active experiment
mlflow.set_experiment(f"/Users/{DA.username}/Model_Deployment_with_Spark")

# Train a Decision Tree model using Scikit-learn
model = DecisionTreeRegressor(max_depth=5, random_state=42)

# Measure training time
start_time = time.time()
model.fit(X_train, y_train)
training_time_single_node = time.time() - start_time
print(f"Training time (single-node): {training_time_single_node} seconds")

# Log model with MLflow
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "decision_tree_single_node")
    run_id_single_node = run.info.run_id  # Capture the run ID
    print(f"Single-node model logged in run: {run_id_single_node}")
    
    # Measure inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time_single_node = time.time() - start_time
    print(f"Inference time (single-node): {inference_time_single_node} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed Model Deployment with Spark MLlib
# MAGIC
# MAGIC In this section, we will demonstrate the deployment of a **DecisionTreeRegressor** model using Spark MLlib. Spark MLlib fully leverages the distributed nature of Spark, making it scalable and ideal for training and inference on large datasets.
# MAGIC
# MAGIC We will train a **DecisionTreeRegressor** model, perform distributed inference, and log the trained model using MLflow for future tracking and deployments.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. **Train a DecisionTreeRegressor model** using Spark MLlib on a distributed Spark DataFrame.
# MAGIC 2. **Log the trained model** using MLflow for version control and future inference.
# MAGIC 3. **Set an alias** for the registered model version to easily identify and retrieve the best-performing model.
# MAGIC 4. **Perform distributed inference** on the test data.
# MAGIC
# MAGIC **Walkthrough:**
# MAGIC - **Initialize the DecisionTreeRegressor model**: We will use Spark's `DecisionTreeRegressor` to initialize and train the model on distributed data, allowing for scalable training.
# MAGIC - **Create a pipeline**: By using Spark's `Pipeline`, we can chain multiple steps, including feature vector assembly and model training, into a single unified workflow.
# MAGIC - **Train the model**: The training will be executed across a distributed Spark DataFrame, which allows for efficient and scalable processing of large datasets.
# MAGIC - **Log the model with MLflow**: The trained model will be logged into MLflow for tracking, reproducibility, and potential future deployments.
# MAGIC - **Set model alias**: We will assign a "champion" alias to the latest model version in MLflow, making it easier to reference the best-performing version for future operations.

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline
import mlflow
import time

# Remove or rename the existing 'features' column if it exists
if 'features' in train_df.columns:
    train_df = train_df.drop('features')
if 'features' in test_df.columns:
    test_df = test_df.drop('features')

# Define and train the DecisionTreeRegressor model in distributed mode
decision_tree = DecisionTreeRegressor(featuresCol="features", labelCol="quality", maxDepth=5)

# Create a pipeline for training
pipeline = Pipeline(stages=[assembler, decision_tree])

# Measure training time for the distributed model
start_time = time.time()
dt_model = pipeline.fit(train_df)
training_time_distributed = time.time() - start_time
print(f"Training time (distributed): {training_time_distributed} seconds")

# Log the model using MLflow
with mlflow.start_run() as run:
    mlflow.spark.log_model(dt_model, artifact_path="model-artifacts")
    run_id_distributed = run.info.run_id  # Capture the run ID
    print(f"Distributed model logged in run: {run_id_distributed}")

    # Measure inference time
    start_time = time.time()
    predictions = dt_model.transform(test_df)
    inference_time_distributed = time.time() - start_time
    print(f"Inference time (distributed): {inference_time_distributed} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparison: Single-Node vs. Distributed Model
# MAGIC
# MAGIC In this section, we will compare the performance of the **Decision Tree Regressor** model deployed in both single-node and distributed configurations. Rather than focusing solely on error metrics, we will assess the performance based on the following key factors:
# MAGIC
# MAGIC 1. **Training Time**: Measure the time taken to train the model in both single-node and distributed configurations. This will highlight the efficiency of each approach in handling data processing and training.
# MAGIC 2. **Inference Time**: Measure the time taken to perform inference on the test data using each model. This will help compare the scalability of the deployment methods when applying the trained model to make predictions.
# MAGIC 3. **Resource Utilization**: Optionally, monitor and compare resource utilization (like memory and CPU usage) for each approach to understand the resource efficiency of each deployment.
# MAGIC
# MAGIC By focusing on training and inference times, this comparison emphasizes the practical differences in scalability and efficiency between single-node and distributed deployments using Spark. This comparison will help you choose the best approach based on your dataset size and computational needs.

# COMMAND ----------

# Create a comparison table for training and inference times
comparison_data = {
    "Model": ["Single-Node DecisionTreeRegressor", "Distributed DecisionTreeRegressor"],
    "Training Time (seconds)": [training_time_single_node, training_time_distributed],
    "Inference Time (seconds)": [inference_time_single_node, inference_time_distributed]
}

import pandas as pd
comparison_df = pd.DataFrame(comparison_data)

# Display the comparison table
print("Time Comparison between Single-Node and Distributed Gradient Boosting Models:")
display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Performing Parallelized Inference Using `spark_udf`
# MAGIC
# MAGIC **Why Use UDFs for Parallelized Inference?**
# MAGIC
# MAGIC User Defined Functions (UDFs) in Spark allow you to apply custom functions to data in parallel. By defining a UDF, you can distribute the inference workload across a Spark cluster. This method ensures that even if your model is single-node, the inference can scale across your distributed data.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC 1. **Define a Spark UDF** that applies your trained Scikit-learn model to distributed data.
# MAGIC 2. **Use the UDF** to perform inference on the distributed Spark DataFrame.
# MAGIC 3. **Log the results** and view predictions using Spark.
# MAGIC
# MAGIC
# MAGIC **For further learning:**  
# MAGIC You can explore more about UDFs and parallelized operations in the [Spark UDF Documentation](https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html#pandas-udfs-a-k-a-vectorized-udfs).

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType, struct
import mlflow.pyfunc
from pyspark.ml.feature import VectorAssembler

#Define a Spark UDF that uses your trained model

# Load the logged model from MLflow using the run_id captured earlier
model_uri = f"runs:/{run_id_single_node}/decision_tree_single_node"  # Replace 'run_id' with your logged model's run ID
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")

# Prepare the Spark DataFrame by creating feature vectors
# (Ensure that 'features' column doesn't already exist from earlier)
if "features" not in df_with_features.columns:
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_with_features = assembler.transform(df_with_features)

#Apply the UDF to distributed data for parallelized inference
# The UDF is applied on the 'features' column to generate predictions
predictions = df_with_features.withColumn(
    "predicted_quality",
    predict_udf(struct(*feature_columns))
)

#Display predictions
display(predictions.select(*feature_columns, "quality", "predicted_quality"))

# Saving predictions to a Delta table:
predictions.write.format("delta").mode("overwrite").saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.single_node_udf_predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed Inference Using Spark ML
# MAGIC
# MAGIC **Why Use Distributed Inference with Spark ML?**
# MAGIC
# MAGIC For larger datasets and machine learning models, Spark MLlib provides an efficient way to perform inference in a distributed manner. Spark MLlib handles both data distribution and computation, making it highly scalable and ideal for large-scale workloads. 
# MAGIC
# MAGIC Distributed inference allows us to apply the trained model to data across multiple nodes, improving performance and enabling the model to handle large datasets in real-time.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. **Prepare Features**: Use Spark’s `VectorAssembler` to combine multiple feature columns into a single feature vector.
# MAGIC 2. **Train a Distributed Model**: Train a distributed **Decision Tree Regressor** model with Spark ML on your dataset.
# MAGIC 3. **Perform Distributed Inference**: Apply the trained model to new data using Spark ML's `transform()` method.
# MAGIC 4. **Save the Predictions**: Store the inference results (predictions) in a Delta table for further analysis or reporting.
# MAGIC
# MAGIC **For further exploration** [Spark MLlib's Guide](https://spark.apache.org/docs/latest/ml-guide.html).

# COMMAND ----------

if "features" in test_df.columns:
    test_df = test_df.drop("features")

# Perform inference on the test data using the distributed GBT model
predictions_dt = dt_model.transform(test_df)

# Show predictions
display(predictions_dt.select("features", "quality", "prediction"))

# Save predictions to a Delta table in Unity Catalog
predictions_dt.write.format("delta").mode("overwrite").saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.distributed_predictions_table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we explored two key deployment strategies using Spark: single-node deployment with Scikit-learn and distributed deployment with Spark MLlib. We highlighted the benefits and challenges of each approach in terms of scalability and processing power. Additionally, we demonstrated parallelized inference using Spark UDFs, showcasing how to leverage Spark’s distributed architecture for efficient large-scale predictions. This comprehensive approach to model deployment ensures that machine learning models can efficiently scale and serve predictions across large datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
