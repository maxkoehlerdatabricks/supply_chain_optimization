# Databricks notebook source
# MAGIC %md
# MAGIC # Fine Grained Demand Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC *Prerequisite: Make sure to run 01_Introduction_And_Setup before running this notebook.*
# MAGIC 
# MAGIC In this notebook we to a one-week-ahead forecast to estimate next week's demand for each store and product. We then aggregate on a distribution center level for each product.
# MAGIC 
# MAGIC Key highlights for this notebook:
# MAGIC - Use Databricks' collaborative and interactive notebook environment to find an appropriate time series mdoel
# MAGIC - Pandas UDFs (user-defined functions) can take your single-node data science code, and distribute it across different keys (e.g. SKU)  

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false

# COMMAND ----------

print(cloud_storage_path)
print(dbName)

# COMMAND ----------

import os
import datetime as dt
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as md

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX

import mlflow
import hyperopt
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
mlflow.autolog(disable=True)

from statsmodels.tsa.api import Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

import pyspark.sql.functions as f
from pyspark.sql.types import *

# COMMAND ----------

demand_df = spark.read.table(f"{dbName}.part_level_demand")
demand_df = demand_df.cache() # just for this example notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examine an example: Extract a single time series and convert to pandas dataframe

# COMMAND ----------

display(demand_df)

# COMMAND ----------

example_product = demand_df.select("product").orderBy("product").limit(1).collect()[0].product
example_store = demand_df.select("store").orderBy("store").limit(1).collect()[0].store
example_store

# COMMAND ----------

example_product = demand_df.select("product").orderBy("product").limit(1).collect()[0].product
example_store = demand_df.select("store").orderBy("store").limit(1).collect()[0].store
pdf = demand_df.filter( (f.col("product") == example_product) & (f.col("store") == example_store)  ).toPandas()

# Create single series 
series_df = pd.Series(pdf['demand'].values, index=pdf['date'])
series_df = series_df.asfreq(freq='W-MON')

display(series_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find a Model via Holt’s Winters Seasonal Method

# COMMAND ----------

fit1 = ExponentialSmoothing(
    pdf,
    seasonal_periods=3,
    trend="add",
    seasonal="add",
    use_boxcox=True,
    initialization_method="estimated",
).fit(method="ls")
fcast1 = fit1.forecast(forecast_horizon).rename("Additive trend and additive seasonal")

# COMMAND ----------


