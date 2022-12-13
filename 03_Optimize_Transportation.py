# Databricks notebook source
# MAGIC %md
# MAGIC # Transport Optimization

# COMMAND ----------

# MAGIC %md
# MAGIC *Prerequisite: Make sure to run 02_Fine_Grained_Demand_Forecasting before running this notebook.*
# MAGIC 
# MAGIC In this notebook we solve the LP to optimize transport from the plants to the distribution centers for each products. Furthermore, we show how to scale to hunderd thousands of products.
# MAGIC 
# MAGIC Key highlights for this notebook:
# MAGIC - Use Databricks' collaborative and interactive notebook environment to find optimization procedure
# MAGIC - Pandas UDFs (user-defined functions) can take your single-node data science code, and distribute it across different keys 

# COMMAND ----------

# MAGIC %pip install pulp

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false

# COMMAND ----------

print(cloud_storage_path)
print(dbName)

# COMMAND ----------

import os
import datetime as dt
import numpy as np
import pandas as pd

from pulp import *

import pyspark.sql.functions as f
from pyspark.sql.types import *

# COMMAND ----------

# Demand for each distribution center
distribution_center_demand = spark.read.table(f"{dbName}.distribution_center_demand")
display(distribution_center_demand)

# COMMAND ----------

# Transportation cost for each plant to each distribution center for each product
transport_cost_table = spark.read.table(f"{dbName}.transport_cost_table")
display(transport_cost_table)

# COMMAND ----------

# MAGIC %md
# MAGIC # Goal
# MAGIC Plant: W = (A,B,C)
# MAGIC DC: B = (1,2,3,4,5)
# MAGIC 
# MAGIC Quantities: x_(w,b) >= 0 for all w in W and b in B, and must be zero or positive integers
# MAGIC 
# MAGIC The goal is to minimize transportation costs for each product:
# MAGIC 
# MAGIC cost_(A,1)*x_(A,1) + … + cost_(C,5)*x_(C,5) → min! W.r.t. X’s
# MAGIC 
# MAGIC The constraints are given by the demands for each product:
# MAGIC 
# MAGIC x_(A,1) +  x_(B,1) +  x_(C,1) >= The demand of distribution center 1 from step 1
# MAGIC ….......
# MAGIC x_(A,5) +  x_(B,5) +  x_(C,5) >= The demand of distribution center 5 from step 1
# MAGIC 
# MAGIC We could also add supply constraints here.
# MAGIC 
# MAGIC The goal is to automatically formulate the LP, put it in a Python function and then use applyInPandas.

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


