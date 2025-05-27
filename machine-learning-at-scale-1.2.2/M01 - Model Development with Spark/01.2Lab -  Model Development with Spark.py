# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab - Model Development with Spark
# MAGIC In this lab, you will engage in an end-to-end workflow to develop a machine learning model using **Apache Spark** and **Delta Lake**. You'll start by loading data from a Delta table, performing essential data manipulations, and then build a regression model using **Spark ML**. Additionally, you will log and track your model using **MLflow** to demonstrate how to manage machine learning models in production environments.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _By the end of this lab, you will:_
# MAGIC
# MAGIC - **Task 1: Prepare Data for Model Training Using Spark DataFrames**
# MAGIC   - **Task 1.1:** Read a Delta Table into a Spark DataFrame.
# MAGIC   - **Task 1.2:** Perform Basic Data Manipulations.
# MAGIC   - **Task 1.3:** Write Data to a Delta Table.
# MAGIC
# MAGIC - **Task 2: Build and Evaluate a Machine Learning Model Using Spark ML**
# MAGIC   - **Task 2.1:** Perform a Train-Test Split Using Spark ML.
# MAGIC   - **Task 2.2:** Assemble a Feature Vector Using Spark ML.
# MAGIC   - **Task 2.3:** Additional Feature Engineering (Independent Exploration) 
# MAGIC     - You will have the freedom to experiment with different transformations and see how they impact the model’s performance.
# MAGIC   - **Task 2.4:** Fit a Linear Regression Model Using Spark ML.
# MAGIC   - **Task 2.5:** Create and Fit a Pipeline.
# MAGIC
# MAGIC - **Task 3: Evaluate and Log the Model with MLflow**
# MAGIC
# MAGIC   - **Task 3.1:** Compute Predictions
# MAGIC     - Use the linear regression model (LRM) you built previously to make predictions.
# MAGIC   - **Task 3.2:** Evaluate Model Performance
# MAGIC     - Use metrics such as RMSE and R² to evaluate your LRM.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Instructions**
# MAGIC
# MAGIC Follow the lab steps closely to ensure you build a complete and well-performing model. Each task will guide you through important concepts, and you will have opportunities to explore Spark ML and Delta Lake features through independent exploration.
# MAGIC
# MAGIC If you feel confident after completing the main tasks, try some **bonus challenges** at the end of each section for a deeper understanding of the material!

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
# MAGIC Before starting the Lab, run the provided classroom setup script.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup-lab"

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this Lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets.california_housing}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 1: Data Preparation
# MAGIC In this task, you will load a dataset from a Delta table, apply basic transformations, and save the cleaned data back to a Delta table. This task is fundamental for ensuring the data is properly prepared for building a machine learning model.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Task 1.1: Read a Delta Table into a Spark DataFrame
# MAGIC
# MAGIC In this step, you will read data from a Delta table into a **Spark DataFrame**. Delta tables provide ACID transactions, scalable metadata handling, and unification of streaming and batch data processing.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. Use the `spark.read.format("delta")` method to read the Delta table.
# MAGIC 2. Display the schema of the DataFrame to understand the structure of the dataset.
# MAGIC 3. Preview the data using `display()` to ensure it’s loaded correctly.

# COMMAND ----------

# Load data into a Spark DataFrame
housing_df = spark.read.format("delta").load(f"{DA.paths.working_dir}/v01/large_california_housing_delta")

# Display schema
housing_df.printSchema()

# Display data
display(housing_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.2: Perform Basic Data Manipulations
# MAGIC
# MAGIC Now that the data is loaded, the next step is to clean and filter the data for model training. You will select relevant columns and filter the rows based on certain conditions.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Select** the columns that are most relevant for your machine learning task (e.g., features like `MedInc`, `HouseAge`, etc.).
# MAGIC 2. **Filter** the rows to remove any invalid or irrelevant data.
# MAGIC 3. Optionally, explore additional transformations using **PySpark** functions like `withColumn()` to modify or add new columns.

# COMMAND ----------

from pyspark.sql.functions import col

# Select and filter relevant columns
df_filtered = housing_df.select('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'label')


# Example: Apply basic filtering (modify as you explore)
df_filtered = df_filtered.filter(col('AveRooms') > 1)

# Display filtered data
display(df_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploration Prompt:**
# MAGIC - **Try This**: Experiment with additional transformations using `withColumn()` to create new columns or modify existing ones. For example, you could add a new column that normalizes the `MedInc` column.
# MAGIC - **Documentation**: [Use this PySpark DataFrames Guide](https://spark.apache.org/docs/latest/api/python/getting_started/index.html) to explore various DataFrame operations.
# MAGIC - **Bonus Task**: Try adding filters based on different thresholds or applying statistical transformations.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Task 1.3: Write Data to a Delta Table
# MAGIC After filtering and manipulating the data, you will now save the transformed DataFrame back to a Delta table. This ensures that the data is versioned and can be used in later steps of the machine learning pipeline.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. Define the path to the output Delta table.
# MAGIC 2. Save the DataFrame using the **Delta Lake write modes**, such as `overwrite` or `append`.
# MAGIC 3. Verify the saved data by reading it back from the Delta table and displaying it.

# COMMAND ----------

# Define the output Delta table path without using dbfs
output_delta_table = f"{DA.catalog_name}.{DA.schema_name}.lab_delta_table"

# Save the filtered DataFrame to a Delta table (overwrite mode)
df_filtered.write.format('delta').mode('overwrite').saveAsTable(output_delta_table)

# Verify by reading the data back from the Delta table
df_output =  spark.read.format('delta').table(output_delta_table)
# Display the data read from the Delta table to verify
display(df_output)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploration Prompt:**
# MAGIC - **Explore Write Modes**: Experiment with different write modes such as `append`, `ignore`, and `error`. Learn more from the [Delta Lake Quickstart Guide](https://docs.delta.io/latest/quick-start.html).
# MAGIC - **Bonus Task**: Try saving the data using a different write mode (e.g., `append`) and observe the behavior when you attempt to overwrite or append the same data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Model Development
# MAGIC
# MAGIC In this task, you will split the dataset, create a feature vector, and train a machine learning model using **Spark ML**. You will also have the opportunity to experiment with feature engineering techniques to enhance the model's performance.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.1: Perform a Train-Test Split Using Spark ML
# MAGIC
# MAGIC Splitting the dataset into training and test sets is essential for evaluating model performance. In this task, you will perform a reproducible train-test split.
# MAGIC
# MAGIC **Instructions**:
# MAGIC 1. **Split the data** into training and test sets using an 80/20 split.
# MAGIC 2. Use the `randomSplit()` method with a random seed to ensure reproducibility.
# MAGIC

# COMMAND ----------

# Perform a reproducible train-test split
train_df, test_df = df_output.randomSplit([0.8,0.2], seed=48)

# Display the number of records in each set
display(train_df.count())
display(test_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploration Prompt:**
# MAGIC - Try experimenting with different train-test split ratios (e.g., 70/30, 90/10) and observe how the size of each set changes.
# MAGIC - Learn more about `randomSplit()` by visiting the [Spark Documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.randomSplit.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ###Task 2.2: Assemble a Feature Vector Using Spark ML
# MAGIC
# MAGIC Now, you will use **VectorAssembler** to combine multiple features into a single vector. You will also experiment with **StandardScaler** to scale the feature vectors, which is important for algorithms like linear regression.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Define feature columns** and combine them into a single vector using `VectorAssembler`.
# MAGIC 2. **Scale the features** using `StandardScaler` to normalize them.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler

# Define feature columns (input columns)
feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# Assemble the feature vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

# Initialize StandardScaler to scale the feature vectors
scaler = StandardScaler(inputCol=assembler.getOutputCol(), outputCol='features_scaled')

# Scale the feature vectors
scaler_model =  scaler.fit(train_df)
train_df = scaler_model.transform(train_df)
test_df = scaler_model.transform(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploration Prompt:**
# MAGIC - **Experiment with Scaling**: Try running the model without scaling the features and compare the results. How does scaling affect the performance?
# MAGIC - **Documentation**: Refer to the [VectorAssembler Documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html) for more configuration options.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.3: Additional Feature Engineering (Independent Exploration)
# MAGIC In this section, you can experiment with additional feature transformations to enhance your model. Refer to the [Spark ML Feature Transformers documentation](https://spark.apache.org/docs/latest/ml-features.html) for more options.
# MAGIC
# MAGIC **Explore the following transformations:**
# MAGIC
# MAGIC - **Add another transformer:** For instance, you can use [**`PolynomialExpansion`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.PolynomialExpansion.html) to generate polynomial features from your existing data.
# MAGIC - **Feature extraction:** Try using [**`VarianceThresholdSelector`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VarianceThresholdSelector.html) to select the most relevant features based on statistical tests.
# MAGIC
# MAGIC Feel free to experiment with these transformations or explore other options from the documentation.

# COMMAND ----------

# TODO
# Perform Polynomial Expansion and Feature Selection
# Task: Add polynomial features to the dataset using PolynomialExpansion (degree=2),
# and then perform feature selection using VarianceThresholdSelector to filter features with low variance.
# Implement the code below to transform both train_df and test_df.
from pyspark.ml.feature import PolynomialExpansion

poly_expansion = PolynomialExpansion(degree=2, inputCol='features_scaled', outputCol='poly_features')
train_df = poly_expansion.transform(train_df)
test_df = poly_expansion.transform(test_df)

from pyspark.ml.feature import VarianceThresholdSelector
vt_selector = VarianceThresholdSelector(varianceThreshold=8, featuresCol='poly_features', outputCol='selected_features')
vt_selector_model = vt_selector.fit(train_df)

train_df = vt_selector_model.transform(train_df)
test_df = vt_selector_model.transform(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploration Prompt:**
# MAGIC - **Try Different Transformers**: Explore other feature transformers from the [Spark ML Feature Transformers documentation](https://spark.apache.org/docs/latest/ml-features.html). For example, experiment with `PCA` for dimensionality reduction.
# MAGIC - **Bonus Task**: Investigate the effect of polynomial expansion with higher degrees (e.g., 3 or 4).

# COMMAND ----------

# MAGIC %md
# MAGIC ###Task 2.4: Fit a Linear Regression Model Using Spark ML
# MAGIC
# MAGIC In this task, you will train a **Linear Regression** model using the features you’ve engineered. After training, you will generate predictions on the test set.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Initialize the `LinearRegression` model** using the feature and label columns.
# MAGIC 2. **Train the model** on the training dataset and generate predictions on the test set.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# Initialize the Linear Regression model
lr = LinearRegression(featuresCol='selected_features')

# Train the model on the training dataset
lr_model = lr.fit(train_df)

# Make predictions on the test data
predictions = lr_model.transform(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploration Prompt:**
# MAGIC - **Experiment with Other Models**: Try replacing the `LinearRegression` model with another regression model such as `DecisionTreeRegressor` or `GBTRegressor`. How do the results differ?
# MAGIC - **Documentation**: Learn more about different regression models in the [Spark ML Documentation](https://spark.apache.org/docs/latest/ml-classification-regression.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ###Task 2.5: Create and Fit a Pipeline
# MAGIC
# MAGIC Now, you will streamline the feature engineering and model training steps by creating a **Pipeline**. Pipelines allow you to combine multiple stages of transformation and modeling into a single, reusable workflow.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Define the pipeline stages** by combining the feature transformers and the regression model.
# MAGIC 2. **Fit the pipeline** on the training data and log the model using MLflow.
# MAGIC - Combine feature engineering and model training into a **Pipeline** for a streamlined process.

# COMMAND ----------

from pyspark.ml import Pipeline
import mlflow
import mlflow.spark

# Define the stages for the pipeline
stages = [assembler, scaler, poly_expansion, vt_selector, lr]
pipeline = Pipeline(stages=stages)

# Remove any existing columns in the training data to avoid conflicts.
# You can drop columns that were created earlier to ensure the pipeline generates these columns again without errors.
columns_to_drop = ['features','features_scaled','poly_features','selected_features']
train_df = train_df.drop(*columns_to_drop)
# Train the pipeline model on the training dataset.
# This will apply all transformations and train the regression model in a single step.
# After training the model, log it to MLflow for tracking and version control.
with mlflow.start_run() as run:
    # Fit the pipeline on the training data
    pipeline_model = pipeline.fit(train_df)
    
    # Log the trained pipeline model to MLflow for tracking and versioning
    mlflow.spark.log_model(pipeline_model, "pipeline_model")
    
    # Log the run details such as run ID and experiment ID
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    mlflow_run = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
    
    # Print the MLflow run information for easy reference
    print(f"MLflow Run ID: {run_id}")
    print(f"MLflow Run: {mlflow_run}")


# COMMAND ----------

# MAGIC %md
# MAGIC **Exploration Prompt:**
# MAGIC - **Experiment with Different Stages**: Try adding or removing stages from the pipeline (e.g., without polynomial expansion) and observe the impact on the model's performance.
# MAGIC - **Bonus Task**: Add a cross-validation stage using `CrossValidator` to tune model hyperparameters within the pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Model Evaluation
# MAGIC
# MAGIC In this task, you will evaluate the performance of your trained model using common regression metrics such as **RMSE** (Root Mean Squared Error) and **R²** (Coefficient of Determination). Additionally, you will log the evaluation metrics and predictions to **MLflow**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.1: Compute Predictions
# MAGIC
# MAGIC In this step, you'll use the trained pipeline model to compute predictions on the test data, which includes all engineered features and transformations.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Remove conflicting columns** from the test dataset to avoid errors.
# MAGIC 2. **Apply the pipeline model** to the test dataset to generate predictions.
# MAGIC 3. **Log the predictions** to MLflow for tracking and version control.
# MAGIC

# COMMAND ----------

# Remove existing columns in the test data to avoid conflicts
test_df = test_df.drop(*columns_to_drop)

# Apply the trained pipeline model to the test dataset to generate predictions
pipeline_predictions = pipeline_model.transform(test_df)

# Start a new MLflow run to log the predictions
with mlflow.start_run(run_name="Compute Predictions") as run:
    
    # Log the predictions as an artifact in MLflow (optional: log predictions file)
    # You can store predictions as a DataFrame or save them to a file for further analysis
    # Optionally, the predictions can be logged as artifacts if saved to a file
    
    # Log the MLflow run ID and link to the run for easy reference
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    mlflow_run = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
    
    # Print the MLflow run details for reference
    print(f"MLflow Run ID: {run_id}")
    print(f"MLflow Run: {mlflow_run}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.2: Evaluate Model Performance
# MAGIC
# MAGIC Now, you will evaluate your model using two common regression metrics:
# MAGIC - **Root Mean Squared Error (RMSE)**: Measures the average magnitude of prediction errors (lower is better).
# MAGIC - **R² (Coefficient of Determination)**: Measures how much variance in the target variable is explained by the model (closer to 1 is better).
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Initialize the evaluators** for both RMSE and R².
# MAGIC 2. **Evaluate the model** on the test data using the `RegressionEvaluator`.
# MAGIC 3. **Log the metrics** to MLflow for tracking.
# MAGIC

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
import mlflow

# Initialize the evaluators for RMSE and R²
evaluator_rmse = RegressionEvaluator(predictionCol='prediction', labelCol="label", metricName="rmse")
evaluator_r2 = RegressionEvaluator(predictionCol='prediction', labelCol='label', metricName="r2")

# Evaluate RMSE and R² on the pipeline predictions
# Use the trained pipeline model's predictions to evaluate its performance using both RMSE and R²
rmse = evaluator_rmse.evaluate(pipeline_predictions)
r2 = evaluator_r2.evaluate(pipeline_predictions)

# Start a new MLflow run to log the evaluation metrics
# This run will log the evaluation metrics to track the performance of the model in MLflow
with mlflow.start_run(run_name="Evaluate Housing Model") as run:
    
    # Log RMSE and R² metrics to MLflow for tracking and future reference
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Retrieve the run ID and experiment ID, and generate a link to the MLflow run for easy access
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    mlflow_run = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
    
    # Print the MLflow run ID and the link to the logged run for easy reference
    print(f"MLflow Run ID: {run_id}")
    print(f"MLflow Run: {mlflow_run}")

# Print the evaluation metrics (RMSE and R²) to see how well the model performed
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² (Coefficient of Determination): {r2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you successfully explored the end-to-end machine learning workflow using Apache Spark, Delta Lake, and MLflow. You started by preparing the dataset, followed by performing feature engineering to enhance model performance. After that, you trained a machine learning model using Spark ML and evaluated it with metrics like RMSE and R². Lastly, you logged the model and its evaluation metrics to MLflow for tracking and future reference, showcasing how models can be managed effectively in a production environment.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>
