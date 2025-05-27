# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Machine Learning at Scale
# MAGIC
# MAGIC In this course, you will gain theoretical and practical knowledge of Apache Spark’s architecture and its application to machine learning workloads within Databricks. You will learn when to use Spark for data preparation, model training, and deployment, while also gaining hands-on experience with Spark ML and pandas APIs on Spark. This course will introduce you to advanced concepts like hyperparameter tuning and scaling Optuna with Spark. This course will use features and concepts introduced in the associate course such as MLflow and Unity Catalog for comprehensive model packaging and governance. 
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC The content was developed for participants with these skills/knowledge/abilities:  
# MAGIC - Intermediate-level knowledge of traditional ML concepts, development, and utilizing Python for ML projects.
# MAGIC - It is recommended that the user has intermediate-level experience with Spark concepts. 
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Course Agenda  
# MAGIC The following modules are part of the **Machine Learning at Scale** course by **Databricks Academy**.
# MAGIC
# MAGIC | # | Module Name | Lesson Name |
# MAGIC |---|-------------|-------------|
# MAGIC | 1 | **[Model Development with Spark]($./M01 - Model Development with Spark)** | • *Lecture:* Model Development with Spark <br> • [**Demo:** Model Development with Spark]($./M01 - Model Development with Spark/01.1 -  Model Development with Spark) <br> • [**Lab:** Model Development with Spark]($./M01 - Model Development with Spark/01.2Lab -  Model Development with Spark) |
# MAGIC | 2 | **[Model Tuning with Optuna on Spark]($./M02 - Model Tuning with Optuna on Spark)** | • *Lecture:* Model Tuning with Optuna <br> • [**Demo:** Optuna and Hyperopt with Spark ML]($./M02 - Model Tuning with Optuna on Spark/02.1 - Optuna and Hyperopt with Spark ML) <br> • [**Demo:** HPO with Ray Tune]($./M02 - Model Tuning with Optuna on Spark/02.2 - HPO with Ray Tune) |
# MAGIC | 3 | **[Model Deployment with Spark]($./M03 - Model Deployment with Spark)** | • *Lecture:* Deployment with Spark <br> • [**Demo:** Model Deployment with Spark]($./M03 - Model Deployment with Spark/03.1 - Model Deployment with Spark) <br> • [**Demo:** Optimization Strategies with Spark and Delta Lake]($./M03 - Model Deployment with Spark/03.2 - Optimization Strategies with Spark and Delta Lake) <br> • [**Lab:** Model Deployment with Spark]($./M03 - Model Deployment with Spark/03.3Lab - Model Deployment with Spark) |
# MAGIC | 4 | **[Pandas API]($./M04 - Pandas APIs)** | • *Lecture:* Pandas APIs <br> • *Lecture:* Pandas UDFs and Functions APIs <br> • [**Demo:** Pandas APIs]($./M04 - Pandas APIs/04.1 - Pandas APIs) <br> • [**Lab:** Pandas APIs]($./M04 - Pandas APIs/04.2Lab - Pandas APIs) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * Use Databricks Runtime version: **`16.3.x-cpu-ml-scala2.12`** for running all demo and lab notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
