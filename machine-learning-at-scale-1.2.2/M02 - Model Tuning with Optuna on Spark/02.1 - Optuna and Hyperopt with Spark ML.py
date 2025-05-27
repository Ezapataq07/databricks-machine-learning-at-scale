# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo: Optuna and Hyperopt with Spark ML
# MAGIC
# MAGIC In this demo, you will learn how to use **Optuna** and **Hyperopt**—powerful hyperparameter optimization (HPO) frameworks—to tune machine learning models in **Databricks** utilizing **Spark ML**.
# MAGIC
# MAGIC We will demonstrate how to implement these frameworks using a **Random Forest Regressor** from SparkML, covering:
# MAGIC - **Defining search spaces** for HPO.
# MAGIC - **Creating objective functions** tailored to different frameworks.
# MAGIC - **Optimizing hyperparameters** using two execution strategies:
# MAGIC   - **Single-node multithreading** for local tuning.
# MAGIC   - **Distributed Spark execution** for large-scale tuning.
# MAGIC
# MAGIC Additionally, we will track and log the results using **MLflow**, enabling efficient management and monitoring of the tuning process.
# MAGIC
# MAGIC ### **Distributed Machine Learning in Databricks**
# MAGIC Distributing the workload for Hyperparameter tuning with Spark can be broken down into two key components:
# MAGIC
# MAGIC 1. **Model Training Level:**  
# MAGIC    - Utilize **PySpark DataFrames** for distributed data processing.
# MAGIC    - Leverage **Spark ML algorithms**, which are inherently scalable.
# MAGIC
# MAGIC 2. **Optimization Level:**  
# MAGIC    - Use **distributed computing frameworks** (i.e., HyperOpt with **Spark Trials**) to parallelize multiple training runs.
# MAGIC    - Scale hyperparameter searches efficiently using Spark’s distributed environment.
# MAGIC
# MAGIC ### **A Warning Concerning HyperOpt on Databricks**
# MAGIC The open-source version of Hyperpot is no longer being maintained and will be removed in the DBR ML versions 16.0+. *This notebook is currently running on a version that supports Hyperopt.* Databricks recommends using Optuna for single-node optimization or RayTune for a similar experience.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## **Learning Objectives**
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC ### **Compare Different Hyperparameter Tuning Frameworks**
# MAGIC - **Optuna** for tuning models on a **single machine** with parallel execution.
# MAGIC - **SparkML + HyperOpt** for fully **distributed tuning across a Spark cluster**.
# MAGIC
# MAGIC ### **Perform Hyperparameter Tuning using Optuna**
# MAGIC - Define an **objective function** tailored to your model.
# MAGIC - Configure **a search space** for hyperparameter optimization.
# MAGIC - Optimize hyperparameters using **single-node execution**.
# MAGIC
# MAGIC ### **Perform Hyperparameter Tuning using HyperOpt with Spark ML**
# MAGIC - Implement **CrossValidator** within **Spark ML** for scalable hyperparameter tuning.
# MAGIC - Utilize **HyperOpt with Spark Trials** for parallel optimization.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need a classic cluster running one of the following Databricks runtime(s): **16.3.x-cpu-ml-scala2.12**. **Do NOT use serverless compute to run this notebook**.

# COMMAND ----------

# MAGIC %pip install -U optuna optuna-integration mlflow
# MAGIC %pip install --upgrade ray[tune]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02.1a

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
# MAGIC ## Load Data and Perform Train-Test Split
# MAGIC
# MAGIC In this step, we will load the dataset from the **Delta table** `wine_quality_features`, which is stored in **Unity Catalog** under:  
# MAGIC `{DA.catalog_name}.{DA.schema_name}.wine_quality_features`
# MAGIC
# MAGIC ### **Instructions:**
# MAGIC 1. **Load the dataset** from the Delta table using `spark.read.table()`.
# MAGIC 2. **Split the dataset** into **training (80%) and testing (20%)** sets to evaluate the model's performance.
# MAGIC    - Since we are using **PySpark DataFrames**, we will use `.randomSplit()` for the split.

# COMMAND ----------

df = spark.read.format("delta").table(f"{DA.catalog_name}.{DA.schema_name}.wine_quality_features")
# Split the dataset into training and test sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: HPO with Optuna and Distributed Training of a Spark Model
# MAGIC In this part, we will use **Optuna for hyperparameter optimization** while training a Spark ML model in **parallel**.
# MAGIC
# MAGIC ### How This Works:
# MAGIC - **Optuna runs on a single machine** to manage hyperparameter tuning, where it suggests configurations for each trial and records their performance.
# MAGIC - **Model training can be distributed**, ensuring the ability to scale out and speed up for large datasets and complex models.
# MAGIC - **Each Optuna trial runs a new model training job** on the Spark cluster, allowing it to evaluate different hyperparameter configurations efficiently.
# MAGIC
# MAGIC This approach allows us to leverage **distributed computing for training** while keeping **hyperparameter optimization lightweight and efficient** on a single node across multiple threads.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Objective Function for Optuna
# MAGIC The first step will be to define the **objective function** for Optuna. This is the function that Optuna will minimize by optimizing hyperparameters like the number of trees (`numTrees`) and the depth of the tree (`maxDepth`). In our case, the objective function is the `Root Mean Squared Error` (RMSE) since our model is a random forest regressor. However, we will use a distributed training approach within this function by running the training on Spark workers.
# MAGIC
# MAGIC **Instructions:**
# MAGIC - **Initialize TPESampler Configuration**. In this example we will use Bayesian optimization along with a Gaussian prior to help stabilize the Parzen estimator (known as the Tree-structured Parzen Estimator algorithm).  
# MAGIC - **Initialize hyperparameters** using Optuna's `trial.suggest_int()` function. This function samples integers between `low` and `high` for the hyperparameter `<hyperparameter_name>` when calling `trial.suggest_int('<hyperparameter_name>', low, high)`. 
# MAGIC - **Train the model** using **Spark's distributed cluster** by running the `RandomForestRegressor` model on Spark workers.
# MAGIC - **Evaluate the model** using the **RMSE** metric (`rmse`), and return it as the value to *minimize* during the optimization. Note, we will tell Optuna to minimize the returned **RMSE** value when we create an Optuna study later. This happens outside the definition of the objective function. 
# MAGIC
# MAGIC Refer to the documentation for:
# MAGIC * [optuna.samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) for the choice of samplers
# MAGIC * [optuna.trial.Trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html) for a full list of functions supported to define a hyperparameter search space.
# MAGIC

# COMMAND ----------

import optuna

optuna_sampler = optuna.samplers.TPESampler(
  consider_prior=True, #Enhance the stability of Parzen estimator by imposing a Gaussian prior when True
  n_startup_trials=3, #The random sampling is used instead of the TPE algorithm until the given number of trials finish in the same study.
  seed=123 # Seed for random number generator.
)

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

class ObjectiveOptuna:
    """
    Objective function class for Optuna hyperparameter tuning with SparkML models.
    
    Instead of loading the dataset in each trial execution, this class receives 
    the training and test datasets during initialization, improving efficiency.
    """

    def __init__(self, train_df, test_df, label_column="label"):
        """
        Initializes the objective function with training and test datasets.

        Args:
            train_df (DataFrame): Spark DataFrame containing features and label for training.
            test_df (DataFrame): Spark DataFrame containing features and label for evaluation.
            label_column (str): Name of the label column in the dataset. Default is "label".
        """
        self.train_df = train_df
        self.test_df = test_df
        self.label_column = label_column
    
    def objective_sparkmodel_distributed_Optuna(self, trial):
        """
        Optuna objective function for tuning regression models using SparkML. Possible models are: Linear Regression, Random Forest, and Gradient-Boosted Trees.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

        Returns:
            float: Root Mean Squared Error (RMSE) to minimize.
        """

        # Select model type
        model_name = trial.suggest_categorical("model", ["LinearRegression", "RandomForest", "GBTRegressor"])

        if model_name == "LinearRegression":
            # Hyperparameter tuning for Linear Regression
            model = LinearRegression(
                featuresCol="features",
                labelCol=self.label_column,
                regParam=trial.suggest_float("reg_param", 0.0, 1.0),
                elasticNetParam=trial.suggest_float("elastic_net_param", 0.0, 1.0)
            )

        elif model_name == "RandomForest":
            # Hyperparameter tuning for Random Forest
            model = RandomForestRegressor(
                featuresCol="features",
                labelCol=self.label_column,
                numTrees=trial.suggest_int("num_trees", 2, 5, log=True),
                maxDepth=trial.suggest_int("max_depth", 3, 10),
                minInstancesPerNode=trial.suggest_int("min_instances_per_node", 1, 10)
            )

        elif model_name == "GBTRegressor":
            # Hyperparameter tuning for Gradient-Boosted Trees
            model = GBTRegressor(
                featuresCol="features",
                labelCol=self.label_column,
                maxDepth=trial.suggest_int("max_depth", 3, 10),
                maxIter=trial.suggest_int("n_estimators", 2, 5, log=True),
                stepSize=trial.suggest_float("learning_rate", 0.01, 0.5)
            )

        # Train the model
        trained_model = model.fit(self.train_df)

        # Generate predictions
        predictions = trained_model.transform(self.test_df)

        # Evaluate performance using RMSE
        rmse = RegressionEvaluator(
            labelCol=self.label_column,
            predictionCol="prediction",
            metricName="rmse"
        ).evaluate(predictions)

        return rmse

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimize The Spark ML model on Single-Machine Optuna and Log Results with MLflow
# MAGIC In this step, we will utilize `MLflow` to track the optimization process by adding out-of-the box logging provided by Optuna trials using `MLflowCallBack()`. Once we have our logging parameters configured, there are two additional steps to take care of before moving onto the run. 
# MAGIC
# MAGIC 1. Initialize Optuna's `optuna.create_study()`. A *study* is corresponds to the optimization task, which is a set of trials and a trial is a process of evaluating an *objective function*.
# MAGIC 1. Tell Optuna how we want to optimize with `optimize()`. 
# MAGIC
# MAGIC Each trial will be logged to MLflow, including the hyperparameters tested and the corresponding `RMSE` values. Optuna will handle the optimization, while training continues to be distributed across Spark workers.
# MAGIC
# MAGIC **Instructions:**
# MAGIC - **Set up MLflow** to track the experiments using `MLflowCallBack()`.
# MAGIC - **Define the storage location** with the variable `storage_url`. In this demonstration, we will be using the driver node to persist our study information, allowing for distributed optimization. 
# MAGIC - **Setup an Optuna study** with `optuna.study()`. 
# MAGIC - **Optimize hyperparameters** using Optuna's `study.optimize()` method.
# MAGIC - **Log results to MLflow**, including the best hyperparameters and RMSE.
# MAGIC - **End the MLflow run** to ensure that all information is saved.
# MAGIC
# MAGIC *Note on parallelization: The value of `n_jobs` within the `optimization()` function is the number of parallel jobs. If this argument is set to -1 (as we have done below), then the number of parallel jobs is set to the number of CPU cores (the default value for this demonstration is 4 cores).*

# COMMAND ----------

import os
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

# Set up MLflow experiment tracking
experiment_name_spark = os.path.join(
    os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()),
    "02a - Model Tuning with Optuna_spark"
)
mlflow.set_experiment(experiment_name_spark)
experiment_id_spark = mlflow.get_experiment_by_name(experiment_name_spark).experiment_id

def optuna_hpo_fn(n_trials: int, experiment_id: str, optuna_sampler) -> optuna.study.Study:
    """
    Runs hyperparameter optimization using Optuna with MLflow logging.

    Args:
        n_trials (int): Number of trials for optimization.
        experiment_id (str): MLflow experiment ID for logging.
        optuna_sampler (optuna.samplers.BaseSampler): Optuna sampler for search strategy.

    Returns:
        optuna.study.Study: The Optuna study object with optimization results.
    """

    # MLflow callback to log results
    mlflow_callback_spark = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="RMSE",
        create_experiment=False,
        mlflow_kwargs={"experiment_id": experiment_id}
    )

    # Define the objective function
    objective_function = ObjectiveOptuna(train_df, test_df, label_column="quality").objective_sparkmodel_distributed_Optuna

    # Create or load an Optuna study
    study = optuna.create_study(
        study_name="sparkmodel_optuna_distributed_hpo",
        sampler=optuna_sampler,
        load_if_exists=True,
        direction="minimize"
    )

    # Run optimization
    study.optimize(
        objective_function,
        n_trials=n_trials,
        n_jobs=-1,  # Parallel execution
        callbacks=[mlflow_callback_spark]
    )

    # Extract best trial results
    best_trial = study.best_trial
    best_rmse = best_trial.value  # RMSE metric

    # Display results
    print(f"Best Trial Number: {best_trial.number}")
    print(f"Best Hyperparameters: {best_trial.params}")
    print(f"Best RMSE: {best_rmse:.4f}")

    # Log the best results manually in MLflow
    with mlflow.start_run(run_name="best_trial_results"):
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("Best RMSE", best_rmse)

    return study  # Return study for further analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #### Execute the Single Node Study

# COMMAND ----------

# Disable MLflow autologging to prevent unwanted logging of model artifacts
mlflow.autolog(log_models=False, disable=True)

# Invoke Optuna training function on the driver node
single_node_study = optuna_hpo_fn(
    n_trials=10,
    experiment_id=experiment_id_spark,
    optuna_sampler=optuna_sampler
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explanation: Distributing Hyperparameter Tuning and Model Training
# MAGIC
# MAGIC The previous cells implemented **distributed hyperparameter tuning and training** using **Optuna**, **MLflow**, and **Spark MLlib**. 
# MAGIC
# MAGIC #### **Key Characteristics of This Setup**
# MAGIC | Aspect | Current Implementation |
# MAGIC |--------|------------------------|
# MAGIC | **Hyperparameter Tuning** | Runs on a **single machine** (Optuna executes locally, even if multiple trials run in parallel). |
# MAGIC | **Parallel Execution** | Trials are parallelized **within a single machine** and training happens in a **distributed fashion** across multiple threads. |
# MAGIC | **Database Storage** | Uses **default in memory storage** for Optuna trials, limiting multi-machine and multi-process execution. |
# MAGIC | **Experiment Logging** | MLflow logs hyperparameters and RMSE for each trial. |
# MAGIC
# MAGIC #### **How to Fully Distribute Hyperparameter Tuning**
# MAGIC While this implementation already distributes model training, Optuna's default execution with `n_jobs` utilizes multithreading on a single node, which, due to Python's Global Interpreter Lock, allows for concurrency but not true parallelism in CPU-bound tasks. To achieve true parallelization, Optuna can be configured to use multiprocessing, either on a single node or across multiple nodes, by setting up an appropriate backend such as a relational database. To fully distribute the hyperparameter search across multiple machines:
# MAGIC
# MAGIC **Use a centralized database**:
# MAGIC    - Within create_study include storage such as `storage="sqlite:////local_disk0/optuna_distributed_model.db"` for Multi-processing parallelization with single node or client/server Relational Databases like PostgreSQL or MySQL **ex:** for Multi-processing parallelization with multiple nodes
# MAGIC      ```
# MAGIC      storage="mysql://root@localhost/example"
# MAGIC      ```
# MAGIC    - This allows multiple workers to share and execute trials.
# MAGIC    - **Requirement:** Launch a MySQL instance (can be on AWS RDS, Azure Database for MySQL, GCP Cloud SQL, or an on-prem server).  [See Optuna Documentation](https://optuna.readthedocs.io/en/latest/faq.html#how-can-i-parallelize-optimization)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Approaches for HPO with SparkML on Databricks
# MAGIC
# MAGIC Hyperparameter tuning in a **Databricks environment** can be challenging due to **SparkContext limitations** and **process forking issues** in managed clusters. Below are **three recommended approaches** to effectively perform hyperparameter tuning while avoiding common pitfalls.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Challenges with Hyperparameter Tuning in Spark and Python**
# MAGIC 1. **Serialization Issues**:  
# MAGIC    - Passing **Spark objects** (e.g., Spark DataFrame, SparkSession, SparkContext) into a **distributed function** (Hyperopt or a Spark UDF) can cause failures due to **pickling restrictions**.
# MAGIC    
# MAGIC 2. **Single SparkContext Per Notebook**:  
# MAGIC    - Databricks runs a **single Spark driver** (the notebook environment) with one **Spark session**.  
# MAGIC    - Workers **cannot** create new Spark sessions (`SparkSession.getOrCreate()`) without **proper master settings**.
# MAGIC
# MAGIC 3. **Ray’s Process Forking Issue**:  
# MAGIC    - Even in **local mode**, Ray spawns separate processes **per trial**.  
# MAGIC    - These processes **do not inherit** the Spark master URL or Spark session.  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## **Recommended Approaches**
# MAGIC
# MAGIC ### **Option 1: Use Spark’s Built-in Hyperparameter Tuning Tools**
# MAGIC #### Best for: **Native Spark ML hyperparameter tuning**
# MAGIC - **How it Works**:  
# MAGIC   - Leverage **Spark ML’s** `CrossValidator` or `TrainValidationSplit` to perform distributed hyperparameter tuning.  
# MAGIC   - Spark handles **parallelism natively**.
# MAGIC
# MAGIC - **Pros**:
# MAGIC   - Fully **compatible** with Databricks.  
# MAGIC   - Runs in **distributed mode**, leveraging Spark Executors.  
# MAGIC   - Avoids **SparkContext serialization issues**.  
# MAGIC
# MAGIC - **Cons**:
# MAGIC   - Limited to **grid search or random search** (without custom logic).  
# MAGIC   - No advanced Bayesian Optimization (unless implemented manually).  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Option 2: Use Hyperopt with SparkTrials for Distributed Tuning**
# MAGIC #### Best for: **Bayesian Optimization on a Spark Cluster**
# MAGIC - **How it Works**:  
# MAGIC   - Use **Hyperopt** with `SparkTrials`, which **distributes hyperparameter search** across Spark Executors.  
# MAGIC   - Unlike traditional grid/random search, **Bayesian Optimization** **intelligently** selects the best hyperparameters.
# MAGIC
# MAGIC - **Pros**:
# MAGIC   - **Bayesian Optimization** (more efficient than exhaustive search).  
# MAGIC   - **Parallel execution across Spark Executors** (avoids SparkContext issues).  
# MAGIC   - Supports **custom ML models** beyond Spark ML.  
# MAGIC
# MAGIC - **Cons**:
# MAGIC   - Requires using Hyperopt’s `fmin()` API instead of `CrossValidator`.  
# MAGIC   - Does **not** directly integrate with Spark ML Pipelines (models are trained manually).
# MAGIC   - *Note: The open-source version of Hyperopt is no longer being maintained.  Hyperopt will be removed in the next major DBR ML version. Databricks recommends using either Optuna for single-node optimization or RayTune for a similar experience to the deprecated Hyperopt distributed hyperparameter tuning functionality.* [Read more here.](https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/index.html)   
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Option 1: Use Spark ML’s Built-in Hyperparameter Tuning Tools

# COMMAND ----------

import os
import time
import mlflow
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

label_column = "quality"

# MLflow Experiment Setup
experiment_name_spark_cv = os.path.join(
    os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()),
    "02c - Model Tuning with spark cv"
)
mlflow.set_experiment(experiment_name_spark_cv)
experiment_id_spark_cv = mlflow.get_experiment_by_name(experiment_name_spark_cv).experiment_id

# Ensure feature vectorization
if "features" not in train_df.columns:
    feature_cols = [col for col in train_df.columns if col != label_column]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_df_transformed = assembler.transform(train_df).select("features", label_column).na.drop()
else:
    feature_cols = [col for col in train_df.columns if col != label_column]
    train_df_transformed = train_df

# Define RandomForestRegressor and hyperparameter grid
rf = RandomForestRegressor(featuresCol="features", labelCol=label_column, seed=42)

param_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [5, 10, 20])    # Number of trees
    .addGrid(rf.maxDepth, [2, 5, 10])    # Max tree depth
    .build()
)

# Set up CrossValidator
evaluator = RegressionEvaluator(labelCol=label_column, predictionCol="prediction", metricName="rmse")

cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,  # 3-fold cross-validation
    parallelism=4  # Parallel execution
)

with mlflow.start_run(run_name="spark_cv_rf", experiment_id=experiment_id_spark_cv):
    try:
        # Start training
        start_time = time.time()
        cv_model = cv.fit(train_df_transformed)
        training_duration = time.time() - start_time
        mlflow.log_metric("training_duration_s", training_duration)

        # Retrieve best model and evaluate
        best_model = cv_model.bestModel
        train_predictions = best_model.transform(train_df_transformed)
        train_rmse = evaluator.evaluate(train_predictions)
        mlflow.log_metric("train_rmse", train_rmse)
        print(f"Best Model RMSE on training folds: {train_rmse:.4f}")

        # Evaluate on test set
        test_predictions = best_model.transform(test_df)
        test_rmse = evaluator.evaluate(test_predictions)
        mlflow.log_metric("test_rmse", test_rmse)
        print(f"Test RMSE: {test_rmse:.4f}")

        # Log best hyperparameters
        best_num_trees = best_model.getNumTrees
        best_max_depth = best_model.getOrDefault("maxDepth")
        mlflow.log_param("best_numTrees", best_num_trees)
        mlflow.log_param("best_maxDepth", best_max_depth)

        print(f"Best hyperparameters → numTrees={best_num_trees}, maxDepth={best_max_depth}")

        # Log feature importances
        if hasattr(best_model, "featureImportances"):
            importances = best_model.featureImportances
            feat_imp_map = {col: val for col, val in zip(feature_cols, importances.toArray())}
            mlflow.log_text(str(feat_imp_map), "feature_importances.txt")
            print("Feature Importances:", feat_imp_map)

        # Log all hyperparameter results
        avg_metrics = cv_model.avgMetrics
        print("\nHyperparameter Combinations and Avg RMSE:")
        print("-------------------------------------------------------")
        print(f"{'numTrees':<12}{'maxDepth':<12}{'avg_rmse':<10}")
        print("-------------------------------------------------------")

        for i, param_map in enumerate(param_grid):
            avg_rmse = avg_metrics[i] if i < len(avg_metrics) else "N/A"  # Handle index errors safely
            num_trees_val = param_map.get(rf.numTrees, "N/A")
            max_depth_val = param_map.get(rf.maxDepth, "N/A")
            print(f"{num_trees_val:<12}{max_depth_val:<12}{avg_rmse:<10.4f}")

    except Exception as e:
        print(f"Error during cross-validation: {e}")

# End MLflow Run
mlflow.end_run()
print("Cross-validation complete. Check MLflow UI for details.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Use Hyperopt with SparkTrials for Distributed Tuning

# COMMAND ----------

# MAGIC %md
# MAGIC Create a function to define a model to train - here a decision tree regressor - and initialize the first training and RMSE value.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark

# Set up MLflow experiment tracking
experiment_name_Hyperopt = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
) + "/02d - Model Tuning with Hyperopt_spark"
mlflow.set_experiment(experiment_name_Hyperopt)
experiment_id_Hyperopt = mlflow.get_experiment_by_name(experiment_name_Hyperopt).experiment_id

def train_tree(minInstancesPerNode, maxBins):
    with mlflow.start_run(nested=True, experiment_id=experiment_id_Hyperopt):
        
        # (A) Assemble features into a single "features" column
        assembler = VectorAssembler(
            inputCols=feature_columns, 
            outputCol="features"
        )
        
        # (B) Decision Tree Regressor
        dtr = DecisionTreeRegressor(
            labelCol=label_column,
            featuresCol="features",
            minInstancesPerNode=minInstancesPerNode,
            maxBins=maxBins
        )

        if "features" not in train_df.columns:
            # Combine the stages into a pipeline
            pipeline = Pipeline(stages=[assembler, dtr])
        else:
            train_df_transformed = train_df      
            pipeline = Pipeline(stages=[dtr])
        
        # Train (fit) the pipeline
        model = pipeline.fit(train_df)
        
        # Evaluate on the test set
        evaluator = RegressionEvaluator(
            labelCol=label_column,
            predictionCol="prediction",
            metricName="rmse"  # You can choose "mse", "mae", or "r2" as well
        )
        predictions = model.transform(test_df)
        test_metric = evaluator.evaluate(predictions)
        
        # Log the RMSE to MLflow
        mlflow.log_metric("test_rmse", test_metric)
    
    return model, test_metric
mlflow.end_run()

# COMMAND ----------

initial_model, test_rmse = train_tree(minInstancesPerNode=200, maxBins=2)
print(f"The trained decision tree regressor achieved an RMSE of {test_rmse} on the test data.")

# COMMAND ----------

# MAGIC %md
# MAGIC Define the train method for use with Hyperopt.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def train_with_hyperopt(params):
  """
  An example train method that calls into Spark MLlib.
  This method is passed to hyperopt.fmin().
  
  :param params: hyperparameters as a dict (consistent with how search space is defined).
  :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure).
  """
  # Convert these to int, since hyperopt may sample floats
  minInstancesPerNode = int(params['minInstancesPerNode'])
  maxBins = int(params['maxBins'])

  model, rmse = train_tree(minInstancesPerNode, maxBins)
  
  # For hyperopt, "loss" should be minimized, so we can directly use the RMSE
  loss = rmse
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC Define the range of hyperparameters to explore with the dictionary `space`, choose the optimization algorithm (`algo`), and run the optimization with MLflow logging enabled.

# COMMAND ----------

space = {
  'minInstancesPerNode': hp.uniform('minInstancesPerNode', 10, 200),
  'maxBins': hp.uniform('maxBins', 2, 32),
}

algo = tpe.suggest

with mlflow.start_run(run_name="parallel_spark_training_hyperopt", experiment_id=experiment_id_Hyperopt) as parent_run:
  best_params = fmin(
    fn=train_with_hyperopt,
    space=space,
    algo=algo,
    max_evals=8
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Get the best parameters and get a final version of the model.

# COMMAND ----------

# Print out the parameters that produced the best model
best_params

best_minInstancesPerNode = int(best_params['minInstancesPerNode'])
best_maxBins = int(best_params['maxBins'])

final_model, best_rmse = train_tree(best_minInstancesPerNode, best_maxBins)

# COMMAND ----------

# MAGIC %md
# MAGIC Display the RMSE values for the initialized model and the final model after optimization.

# COMMAND ----------

# Evaluate both models on the test set for a final comparison
evaluator = RegressionEvaluator(labelCol=label_column, metricName="rmse")

initial_model_test_rmse = evaluator.evaluate(initial_model.transform(test_df))
final_model_test_rmse = evaluator.evaluate(final_model.transform(test_df))

print(f"On the test data, the initial (untuned) model achieved RMSE = {initial_model_test_rmse}, "
      f"and the final (tuned) model achieved RMSE = {final_model_test_rmse}.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC
# MAGIC In this demo, we explored how to optimize machine learning models in **Databricks** using **Optuna** and **HyperOpt** with **Spark ML**. We demonstrated how these frameworks handle hyperparameter tuning at both the **model training level** and **optimization level**, leveraging distributed computing for scalability.
# MAGIC
# MAGIC We compared multiple strategies for hyperparameter tuning:
# MAGIC - **Single-machine tuning** using Optuna for efficient local execution.
# MAGIC - **Distributed hyperparameter tuning** using HyperOpt with Spark Trials to scale across a cluster.
# MAGIC - **End-to-end SparkML tuning** using `CrossValidator` for native Spark-based optimization.
# MAGIC
# MAGIC ### **Key Takeaways**
# MAGIC - **Parallelization strategies** significantly impact model training efficiency and resource utilization.
# MAGIC - **Databricks provides multiple options** for hyperparameter tuning, allowing flexibility in balancing **scalability vs. compute cost**.
# MAGIC - **MLflow enables seamless experiment tracking**, making it easier to compare results across different tuning frameworks.
# MAGIC
# MAGIC By leveraging these frameworks effectively, you can enhance model performance, streamline experimentation, and scale machine learning workflows efficiently within Databricks.
# MAGIC
# MAGIC ### Next Steps
# MAGIC In the next demonstration, we will see how to use Ray Tune for hyperparameter optimization leveraging a single node and our understanding of Optuna from this demonstration. 
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
