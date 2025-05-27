# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Demo: Model Development with Spark
# MAGIC Welcome to this Demo on **model development** using **Apache Spark** and **Delta Lake**. In this Demo, we will explore the process of developing a machine learning model from data preparation to model deployment. By leveraging Spark ML and Delta Lake, we will perform critical tasks such as reading data from Delta tables, transforming it, and building a regression model that can be evaluated and registered in Unity Catalog using **MLflow**.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC _By the end of this demo, you will be able to:_
# MAGIC
# MAGIC 1. **Data Preparation:**
# MAGIC    - **Read** a Delta table into a Spark DataFrame.
# MAGIC    - **Perform** data manipulation using the Spark DataFrame API.
# MAGIC    - **Write** transformed data back to a Delta table.
# MAGIC
# MAGIC 2. **Model Development:**
# MAGIC    - Perform a reproducible **train-test split** using Spark ML.
# MAGIC
# MAGIC    - **Model Preparation:**
# MAGIC      - Assemble a feature vector using `VectorAssembler` in Spark ML.
# MAGIC    
# MAGIC    - **Model Training:**
# MAGIC      - Fit a regression model using Spark ML.
# MAGIC      - Create and fit a `Pipeline` to automate the training and evaluation process.
# MAGIC    
# MAGIC    - **Model Evaluation:**
# MAGIC      - Use the trained model to compute predictions on test data.
# MAGIC      - Measure model performance using evaluation metrics like **Root Mean Squared Error** (RMSE) and **R²** (Coefficient of Determination).
# MAGIC    
# MAGIC    - **Model Registration:**
# MAGIC      - Log and register the trained model in **Unity Catalog** using **MLflow** for versioning and deployment.

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
# MAGIC Before starting the demo, run the provided classroom setup script. In particular, you will be creating a database called `new_craw` within unity Catalog.

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
# MAGIC ##Part 1: Data Preparation
# MAGIC In this section, We will Show how to prepare the dataset for machine learning by reading data from a Delta table, performing data manipulations using the Spark DataFrame API, and writing the cleaned data back to a Delta table for further use.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Read a Delta Table into a Spark DataFrame
# MAGIC Delta Lake, built on top of Apache Spark, provides ACID transactions, scalable metadata handling, and the unification of batch and streaming data. This makes it ideal for handling large datasets while ensuring data integrity and performance.
# MAGIC
# MAGIC **Instructions:**
# MAGIC - Define the path to the Delta table that contains the data.
# MAGIC - Use the **`spark.read.format("delta")`** function to load data from the Delta table into a Spark DataFrame.
# MAGIC - Verify the schema and the loaded data.

# COMMAND ----------

# MAGIC %md
# MAGIC **[Delta Lake Documentation](https://docs.delta.io/latest/delta-intro.html)**: Learn more about Delta Lake’s core features, including ACID transactions and schema enforcement.

# COMMAND ----------

# Path to the Delta table
data_path = f"{DA.paths.working_dir}/v01/large_wine_quality_delta"

# Load data into a Spark DataFrame
df = spark.read.format("delta").load(data_path)

# Display the schema of the DataFrame
df.printSchema()

# Display the DataFrame
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perform Basic Data Manipulations Using the Spark DataFrame API
# MAGIC
# MAGIC Next, we will filter and select relevant columns from the dataset for model training. The Spark DataFrame API allows us to easily perform these operations.
# MAGIC
# MAGIC **Instructions:**
# MAGIC - **Select relevant columns** for the regression task (such as features and target labels).
# MAGIC - **Filter** the rows based on the condition where `quality` is greater than 3.
# MAGIC - **Display summary statistics** to understand the dataset distribution.

# COMMAND ----------

# MAGIC %md
# MAGIC **[PySpark DataFrame API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html)**: Dive deeper into Spark’s DataFrame API for data selection, filtering, and transformation operations.

# COMMAND ----------

from pyspark.sql.functions import col

# Select specific columns for the regression task
df_selected = df.select("fixed_acidity", 
                        "volatile_acidity", 
                        "citric_acid", 
                        "residual_sugar", 
                        "chlorides", 
                        "free_sulfur_dioxide", 
                        "total_sulfur_dioxide", 
                        "density", 
                        "pH", 
                        "sulphates", 
                        "alcohol", 
                        "quality"
                        )

# Filter rows where the quality is greater than 3 (basic filtering)
df_filtered = df_selected.filter(col("quality") > 3)

# Display the summary statistics of the dataset
display(df_filtered.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Spark DataFrame to a Delta Table
# MAGIC Once the data has been transformed and filtered, we can write it back to a Delta table. This allows us to maintain a versioned, scalable dataset that can be accessed and updated in subsequent steps of the pipeline.
# MAGIC
# MAGIC **Instructions:**
# MAGIC - **Define the output** path for the Delta table.
# MAGIC - **Write the transformed data** back to the Delta table in "append" or "overwrite" mode.
# MAGIC - **Verify the written data** by reading the Delta table again.

# COMMAND ----------

# Define the output Delta table path
output_delta_table = f"{DA.catalog_name}.{DA.schema_name}.delta_table"

# Write the filtered DataFrame to the Delta table (Append Mode)
df_filtered.write.format("delta").mode("append").saveAsTable(output_delta_table)

# Overwrite the Delta table with new data
df_filtered.write.format("delta").mode("overwrite").saveAsTable(output_delta_table)

# Read the data back from the Delta table to verify
df_output = spark.read.format("delta").table(output_delta_table)

# Display the the newly saved Delta table
display(df_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Model Development
# MAGIC In this Part, we will focus on building a machine learning model using **Spark ML**. We will explore the steps required to prepare data for model training, train a regression model, and evaluate its performance.
# MAGIC
# MAGIC We will also cover how to create and fit a **Pipeline** in Spark to automate data transformations and model training, making it easier to manage and reproduce these steps.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perform a Reproducible Train-Test Split Using Spark ML
# MAGIC A crucial step in developing a machine learning model is to split the dataset into a **training set** and a **test set**. This ensures that the model is trained on one portion of the data and evaluated on another, helping assess its performance on unseen data. 
# MAGIC
# MAGIC In this step, we will use **Spark ML** to split the data into 80% for training and 20% for testing. By setting a **seed**, we can ensure the *random* split is reproducible, meaning the data will be split the same way every time we run this step.
# MAGIC
# MAGIC **Instructions**:
# MAGIC
# MAGIC 1. Use the `randomSplit` function from Spark to divide the dataset into training and test sets.
# MAGIC 2. Specify the proportions for the split: 80% of the data for training and 20% for testing.
# MAGIC 3. Set a random seed (e.g., `seed=42`) to ensure the split is reproducible across different runs.

# COMMAND ----------

# MAGIC %md
# MAGIC **[Spark ML DataFrame API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.randomSplit.html)**: Learn more about splitting data and handling DataFrames in Spark ML.

# COMMAND ----------

# Split the data into 80% training and 20% testing sets
train_df, test_df = df_filtered.randomSplit([0.8, 0.2], seed=42)

# Display the number of records in each set
print(f"Training Data Count: {train_df.count()}")
print(f"Test Data Count: {test_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Preparation
# MAGIC
# MAGIC Once the data is split into training and test sets, the next step is to prepare the features for model training. This involves **assembling the selected features** into a single vector that can be fed into the machine learning model. Spark ML’s `VectorAssembler` is a key tool for this process, as it consolidates multiple feature columns into one vector.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Assemble a Feature Vector Using Spark ML
# MAGIC
# MAGIC In this step, we will use VectorAssembler to combine the relevant feature columns into a **single feature vector**. This vector is necessary for feeding the data into machine learning models that expect the input features in a vectorized format. Additionally, we will apply **feature scaling** using StandardScaler to normalize the feature values, which is an important step for models like **Linear Regression, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN)**. However, **tree-based models (like Gradient Boosted Trees and Random Forests) do not require feature scaling** as they are not sensitive to feature magnitude.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Select the feature columns** from the dataset that will be used to train the model.
# MAGIC 2. **Use VectorAssembler** to assemble these feature columns into a single vector named `features`.
# MAGIC 3. **Normalize the feature vector** using StandardScaler:
# MAGIC    - `withMean=True` → Centers the data by subtracting the mean (zero-centered features).
# MAGIC    - `withStd=True` → Scales features by dividing by the standard deviation (column-wise scaling).
# MAGIC 4. Apply the transformations to both the training and test datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC **Further Exploration:**
# MAGIC - **[VectorAssembler API](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html)**: Learn more about how to assemble features in Spark ML.
# MAGIC - **[StandardScaler API](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StandardScaler.html)**: Explore how to scale features for improved model performance.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler

# Define the feature columns
feature_columns = ["fixed_acidity", 
                   "volatile_acidity", 
                   "citric_acid", 
                   "residual_sugar", 
                   "chlorides", 
                   "free_sulfur_dioxide", 
                   "total_sulfur_dioxide", 
                   "density", 
                   "pH", 
                   "sulphates", 
                   "alcohol"]

# Assemble the feature vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Apply the assembler to the training and test datasets to create the 'features' column
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

# Initialize the StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)

# Fit the scaler on the training data
scaler_model = scaler.fit(train_df)

# Transform both the training and test data using the same scaler model
train_df = scaler_model.transform(train_df)
test_df = scaler_model.transform(test_df)

# Display the scaled features
display(train_df.select("scaled_features", "quality"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Training
# MAGIC
# MAGIC After preparing the feature vectors, the next step is to train a machine learning model. In this section, we will use a **Linear Regression Model** to fit a regression model. Additionally, we will streamline the model training process by creating and fitting a **Pipeline** in Spark ML, which automates the data transformations and model training steps.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fit a Model Using Spark ML
# MAGIC
# MAGIC In this step, we will train a machine learning model using the Linear Regression algorithm. Linear Regression is a common method for regression tasks, as it estimates relationships between the dependent variable and one or more independent variables.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Initialize the LinearRegression model** and specify the necessary parameters, such as the input feature column (scaled_features) and the target column (quality).
# MAGIC 2. **Train the model** on the training dataset using the fit() method.
# MAGIC 3. **Make predictions** on the test dataset using the transform() method.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# Initialize Linear Regression model
lr = LinearRegression(featuresCol="scaled_features", labelCol="quality")

# Train the model using the training data
lr_model = lr.fit(train_df)

# Make predictions on the test data      
lr_predictions = lr_model.transform(test_df)

# Display the predictions
display(lr_predictions.select("scaled_features", "quality", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create and Fit a Pipeline Using Spark ML
# MAGIC
# MAGIC To automate and streamline the machine learning workflow, we can use **Pipelines** in Spark ML. A pipeline chains multiple stages of data transformation and model training, making the process reusable and easy to manage. In this case, we will chain the feature assembler, the StandardScaler, and the Linear Regression model into a single pipeline.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create and Fit a Pipeline Using Spark ML
# MAGIC
# MAGIC To automate and streamline the machine learning workflow, we can use **Pipelines** in Spark ML. A pipeline chains multiple stages of data transformation and model training, making the process reusable and easy to manage. In this case, we will chain the feature assembler and the `GBTRegressor` model into a single pipeline in the following sense: we take the original DataFrame, `train_df`, transform it into a new DataFrame that contains the feature vectors, and we use these vectors as inputs for the `GBTRegressor` to make final predictions with the trained model. 
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Create pipeline stages** by combining the feature assembler and the trained `GBTRegressor` model.
# MAGIC 2. **Build the pipeline** using the `Pipeline` class in Spark ML.
# MAGIC 3. **Train the pipeline** model on the training dataset, and it will automatically handle the data transformation and model training steps.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Further Exploration:
# MAGIC - **[Linear Regression API](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html)**: Learn more about Linear Regression for regression tasks.
# MAGIC - **[Spark ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html)**: Explore how to build and use pipelines to streamline machine learning workflows.
# MAGIC

# COMMAND ----------

from pyspark.ml import Pipeline

# If the 'features' column already exists, drop it to avoid conflict
if "features" in train_df.columns or "scaled_features" in train_df.columns:
    train_df = train_df.drop("features", "scaled_features")

# Define the stages of the pipeline
stages = [assembler, scaler, lr]

# Create the pipeline
pipeline = Pipeline(stages=stages)

# Train the pipeline model on the training data
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Evaluation
# MAGIC
# MAGIC Once the model has been trained, the next step is to evaluate its performance on unseen data. In this section, we will:
# MAGIC - **Generate predictions** using the test dataset.
# MAGIC - **Evaluate the model’s performance** using key regression metrics like **Root Mean Squared Error (RMSE)** and **R²** (Coefficient of Determination).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Compute Basic Predictions Using a Spark ML Model
# MAGIC
# MAGIC After the training phase, you can use the model to make predictions on the test data. The predictions are then compared with the actual values to measure the model’s accuracy.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC 1. **Ensure the `features` column** is ready for predictions by dropping any pre-existing version in the test DataFrame.
# MAGIC 2. **Use the trained pipeline** to make predictions on the test data.
# MAGIC 3. **Display the predictions** alongside the actual values for comparison.
# MAGIC

# COMMAND ----------

if "features" in test_df.columns or "scaled_features" in test_df.columns:
    test_df = test_df.drop("features", "scaled_features")
# Make predictions on the test data using the pipeline
pipeline_predictions = pipeline_model.transform(test_df)

# Display the predictions alongside actual values
display(pipeline_predictions.select("scaled_features", "quality", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluate a Regression Model Using a Spark ML API
# MAGIC
# MAGIC To measure how well the model is performing, we will use two key metrics: **Root Mean Squared Error (RMSE)** and **R²** (Coefficient of Determination). RMSE gives us the average prediction error, while R² indicates how much variance in the target variable is explained by the model.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC 1. **Initialize evaluators** for both RMSE and R² metrics using the `RegressionEvaluator` API.
# MAGIC 2. **Evaluate the model** using both metrics.
# MAGIC 3. **Print the results** to assess the model's performance.

# COMMAND ----------

# MAGIC %md
# MAGIC **Further Exploration:**
# MAGIC - **[Spark ML RegressionEvaluator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RegressionEvaluator.html)**: Learn more about the metrics and evaluation methods available in Spark ML.
# MAGIC - **[Root Mean Squared Error (RMSE)](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.evaluation.RegressionMetrics.html#pyspark.mllib.evaluation.RegressionMetrics.rootMeanSquaredError)**: Understand how RMSE works and its significance in regression analysis.
# MAGIC - **[Coefficient of Determination (R²)](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.evaluation.RegressionMetrics.html#pyspark.mllib.evaluation.RegressionMetrics.r2)**: Explore how R² measures the goodness of fit for a regression model.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

# Initialize the regression evaluator for RMSE and R²
evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="quality", metricName="rmse")
evaluator_r2 = RegressionEvaluator(predictionCol="prediction", labelCol="quality", metricName="r2")

# Evaluate RMSE and R²
rmse = evaluator_rmse.evaluate(pipeline_predictions)
r2 = evaluator_r2.evaluate(pipeline_predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² (Coefficient of Determination): {r2}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Description:**
# MAGIC - **RMSE**: Measures the average magnitude of the prediction errors, with lower values indicating better performance.
# MAGIC - **R²**: Provides insight into how well the model explains the variance in the target variable, with a value closer to 1 indicating a better fit.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Registration (Optional)
# MAGIC
# MAGIC Once the model is trained and evaluated, we can register it in **Unity Catalog** using **MLflow**. Model registration is essential for maintaining version control, enabling model sharing across teams, and facilitating model deployment in production environments. 
# MAGIC
# MAGIC By logging the model with MLflow, you can track metrics like **RMSE** and **R²**, and make the model accessible for further usage, versioning, or deployment.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Register the Model in Unity Catalog with MLflow
# MAGIC
# MAGIC In this step, we will:
# MAGIC 1. Log the trained pipeline model to MLflow.
# MAGIC 2. Record evaluation metrics (RMSE and R²).
# MAGIC 3. Register the model in **Unity Catalog**, making it easy to share, test, manage, and deploy the model.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Set the registry URI** to Unity Catalog.
# MAGIC 2. **Infer the model signature** using the training data, which captures the input/output schema for reproducibility.
# MAGIC 3. **Log the model and evaluation metrics** using MLflow within a new MLflow run.
# MAGIC 4. **Register the model** in Unity Catalog for version control.
# MAGIC 5. **Set an alias** for the model version, such as "champion," to identify the best-performing model.

# COMMAND ----------

# MAGIC %md
# MAGIC **Further Exploration:**
# MAGIC - **[MLflow Documentation](https://mlflow.org/docs/latest/index.html)**: Learn more about MLflow for model tracking, logging, and deployment.
# MAGIC - **[Databricks Unity Catalog Documentation](https://docs.databricks.com/data-governance/unity-catalog/index.html)**: Explore how Unity Catalog facilitates model versioning, governance, and sharing.
# MAGIC - **[MLflow Model Registry](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/workspace-model-registry.html)**: Understand how to manage and deploy models using MLflow’s model registry.

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Set the registry URI to Unity Catalog
mlflow.set_registry_uri('databricks-uc')
client = mlflow.tracking.MlflowClient()

# Define model name with 3-level namespace
model_name = f"{DA.catalog_name}.{DA.schema_name}.wine-quality-model"

# Infer the signature using the original feature columns
signature = infer_signature(train_df.select(*feature_columns), train_df.select("quality"))

# Start an MLflow run to log metrics and model
with mlflow.start_run(run_name="Wine Quality Model Development") as run:
    
    # Log the metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log the trained pipeline model to MLflow with the model signature
    mlflow.spark.log_model(
        pipeline_model, 
        "wine_quality_pipeline_model", 
        registered_model_name=model_name,
        signature=signature
    )
    
    print("Model and metrics logged successfully in MLflow!")
    
    # Print MLflow run link
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    mlflow_run = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
    print(f"MLflow Run ID: {run_id}")
    print(f"MLflow Run: {mlflow_run}")

# Register the model in Unity Catalog
def get_latest_model_version(model_name):
    model_version_infos = client.search_model_versions(f"name = '{model_name}'")
    return max([model_version_info.version for model_version_info in model_version_infos])

latest_model_version = get_latest_model_version(model_name)

# Set an alias for the latest model version
client.set_registered_model_alias(model_name, "champion", latest_model_version)

print(f"Model registered with version: {latest_model_version} and alias: 'champion'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we developed a machine learning model using **Apache Spark** and **Delta Lake**. We prepared data from a Delta table, built and evaluated a regression model using **Spark ML**, and assessed its performance with metrics like **RMSE** and **R²**. Finally, we registered the model in **Unity Catalog** with **MLflow**, demonstrating the seamless process of tracking and deploying models in production environments. This workflow showcases the power of Spark for scalable machine learning.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
