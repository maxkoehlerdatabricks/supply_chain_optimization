# Databricks notebook source
dbutils.widgets.dropdown('reset_all_data', 'false', ['true', 'false'], 'Reset all data')
dbutils.widgets.text('dbName',  'supply_chain_optimization_max_kohler' , 'Database Name')
dbutils.widgets.text('cloud_storage_path',  '/Users/max.kohler@databricks.com/field_demos/supply_chain_optimization', 'Storage Path')

# COMMAND ----------

print("Starting ./_resources/01-data-generator")

# COMMAND ----------

cloud_storage_path = dbutils.widgets.get('cloud_storage_path')
dbName = dbutils.widgets.get('dbName')
reset_all_data = dbutils.widgets.get('reset_all_data') == 'true'

# COMMAND ----------

print(cloud_storage_path)
print(dbName)
print(reset_all_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Packages

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime

from dateutil.relativedelta import relativedelta
from dateutil import rrule

import os
import string
import random

import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql.window import Window

import statsmodels.api as sm
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate demand series data

# COMMAND ----------

# MAGIC %md
# MAGIC Parameters

# COMMAND ----------

n=3 # Number of replicates per product category
ts_length_in_weeks = 104 # Length of a time series in years
number_of_stores = 30
n_distribution_centers = 5

# COMMAND ----------

# MAGIC %md
# MAGIC Create a Product Table

# COMMAND ----------

products_categories = spark.createDataFrame(
  ["drilling machine","cordless screwdriver","impact drill","current meter","hammer","screwdriver","nail","screw","spirit level","toolbox"], 
  StringType()).toDF("product_categories")

products_versions = spark.createDataFrame(
  list(range(1,(n+1))),
  StringType()).toDF("product_versions")

product_table = (
  products_categories.
  crossJoin(products_versions).
  select(f.concat_ws('_', f.col("product_categories"), f.col("product_versions")).alias("product"))
                )

display(product_table)

# COMMAND ----------

# MAGIC %md 
# MAGIC Introduce Stores

# COMMAND ----------

store_table = spark.createDataFrame(
  list(range(1,(number_of_stores+1))),
  StringType()).toDF("stores_number")

store_table = store_table.select(f.concat_ws('_',f.lit("Store"), f.col("stores_number")).alias("store"))

display(store_table)

# COMMAND ----------

products_in_stores_table = (
  product_table.
  crossJoin(store_table)
)
display(products_in_stores_table)

# COMMAND ----------

# MAGIC %md 
# MAGIC Generate Date Series

# COMMAND ----------

# End Date: Monday of the current week
end_date = datetime.datetime.now().replace(hour=0, minute=0, second= 0, microsecond=0) 
end_date = end_date + datetime.timedelta(-end_date.weekday()) #Make sure to get the monday before

# Start date: Is a monday, since we will go back integer number of weeks
start_date = end_date + relativedelta(weeks= (- ts_length_in_weeks))

# Make a sequence 
date_range = list(rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date))

#Create a pandas data frame
date_range = pd.DataFrame(date_range, columns =['date'])

display(date_range)

# COMMAND ----------

# MAGIC %md
# MAGIC Simulate parameters for ARMA series

# COMMAND ----------

# MAGIC %md
# MAGIC To Do: 
# MAGIC - Make more realistic to have similar magnitudes per product group

# COMMAND ----------

# Define schema for new columns
arma_schema = StructType(
  [
    StructField("Variance_RN", FloatType(), True),
    StructField("Offset_RN", FloatType(), True),
    StructField("AR_Pars_RN", ArrayType(FloatType()), True),
    StructField("MA_Pars_RN", ArrayType(FloatType()), True)
  ]
)

# Generate random numbers for the ARMA process
np.random.seed(123)
n_ = products_in_stores_table.count()


variance_random_number = list(abs(np.random.normal(100, 50, n_)))
offset_random_number = list(np.maximum(abs(np.random.normal(10000, 5000, n_)), 4000))
ar_length_random_number = np.random.choice(list(range(1,4)), n_)
ar_parameters_random_number = [np.random.uniform(low=0.1, high=0.9, size=x) for x in ar_length_random_number] 
ma_length_random_number = np.random.choice(list(range(1,4)), n_)
ma_parameters_random_number = [np.random.uniform(low=0.1, high=0.9, size=x) for x in ma_length_random_number] 


# Collect in a dataframe
pdf_helper = (pd.DataFrame(variance_random_number, columns =['Variance_RN']). 
              assign(Offset_RN = offset_random_number).
              assign(AR_Pars_RN = ar_parameters_random_number).
              assign(MA_Pars_RN = ma_parameters_random_number) 
             )

spark_df_helper = spark.createDataFrame(pdf_helper, schema=arma_schema)
spark_df_helper = (spark_df_helper.
  withColumn("row_id", f.monotonically_increasing_id()).
  withColumn('row_num', f.row_number().over(Window.orderBy('row_id'))).
  drop(f.col("row_id"))
                  )

products_in_stores_table = (products_in_stores_table.
                            withColumn("row_id", f.monotonically_increasing_id()).
                            withColumn('row_num', f.row_number().over(Window.orderBy('row_id'))).
                            drop(f.col("row_id"))
                           )


products_in_stores_table = products_in_stores_table.join(spark_df_helper, ("row_num")).drop(f.col("row_num"))
display(products_in_stores_table)

# COMMAND ----------

# MAGIC %md 
# MAGIC Generate individual demand series

# COMMAND ----------

# To maximize parallelism, we can allocate each ("product", store") group its own Spark task.
# We can achieve this by:
# - disabling Adaptive Query Execution (AQE) just for this step
# - partitioning our input Spark DataFrame as follows:
spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")
n_tasks = products_in_stores_table.select("product", "store").distinct().count()


# function to generate an ARMA process
def generate_arma(arparams, maparams, var, offset, number_of_points, plot):
  np.random.seed(123)
  ar = np.r_[1, arparams] 
  ma = np.r_[1, maparams] 
  y = sm.tsa.arma_generate_sample(ar, ma, number_of_points, scale=var, burnin= 3000) + offset
  y = np.round(y).astype(int)
  y = np.absolute(y)
  
  if plot:
    x = np.arange(1, len(y) +1)
    plt.plot(x, y, color ="red")
    plt.show()
    
  return(y)


#Schema for output dataframe
schema = StructType(  
                    [
                      StructField("product", StringType(), True),
                      StructField("store", StringType(), True),
                      StructField("date", DateType(), True),
                      StructField("demand", FloatType(), True),
                      StructField("row_number", FloatType(), True)
                    ])

# Generate an ARMA
def time_series_generator_pandas_udf(pdf):
  out_df = date_range.assign(
   demand = generate_arma(arparams = pdf.AR_Pars_RN.iloc[0], 
                        maparams= pdf.MA_Pars_RN.iloc[0], 
                        var = pdf.Variance_RN.iloc[0], 
                        offset = pdf.Offset_RN.iloc[0], 
                        number_of_points = date_range.shape[0], 
                        plot = False),
  product = pdf["product"].iloc[0],
  store = pdf["store"].iloc[0]
  )
  
  out_df["row_number"] = range(0,len(out_df))
  
  out_df = out_df[["product", "store", "date", "demand", "row_number"]]

  return(out_df)

#pdf = products_in_stores_table.toPandas().head(1)

# Apply the Pandas UDF and clean up
demand_df = ( 
  products_in_stores_table.
   #repartition(n_tasks, "product", "store").
   groupby("product", "store"). 
   applyInPandas(time_series_generator_pandas_udf, schema)
)

#assert date_range.shape[0] * products_in_stores_table.count() == demand_df.count()

display(demand_df)

# COMMAND ----------

# Select a sepecific time series
display(demand_df.join(demand_df.sample(False, 1 / demand_df.count(), seed=0).limit(1).select("product", "store"), on=["product", "store"], how="inner"))

# COMMAND ----------

# MAGIC %md
# MAGIC Save as a Delta table

# COMMAND ----------

demand_df_delta_path = os.path.join(cloud_storage_path, 'demand_df_delta')

# COMMAND ----------

# Write the data 
demand_df.write \
.mode("overwrite") \
.format("delta") \
.save(demand_df_delta_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.part_level_demand")
spark.sql(f"CREATE TABLE {dbName}.part_level_demand USING DELTA LOCATION '{demand_df_delta_path}'")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dbName}.part_level_demand"))

# COMMAND ----------

display(spark.sql(f"SELECT COUNT(*) as row_count FROM {dbName}.part_level_demand"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate hardware to distribution center mapping table

# COMMAND ----------

distribution_centers = (
  spark.createDataFrame(list(range(1, n_distribution_centers + 1)),StringType()).
  toDF("distribution_center_helper").
  withColumn("distribution_center", f.concat_ws('_', f.lit("Distribution_Center"), f.col("distribution_center_helper"))).
  select("distribution_center")
)


display(distribution_centers)

# COMMAND ----------

# We need more distribution centers than stores
assert (distribution_centers.count() <= store_table.count()) & (distribution_centers.count() > 0)

#Replicate distribution centers such that all distribution centers are used, but the table has the same number of rows than store_table
divmod_res = divmod(store_table.count(), distribution_centers.count())

rest_helper = distribution_centers.limit(divmod_res[1])
maximum_integer_divisor = (
  spark.createDataFrame(list(range(1, divmod_res[0] + 1)),StringType()).
  toDF("number_helper").
  crossJoin(distribution_centers).
  select("distribution_center")
)

distribution_centers_replicated = maximum_integer_divisor.unionAll(rest_helper)

assert distribution_centers_replicated.count() == store_table.count()

# Append distribution_centers_replicated and store_table column-wise
distribution_centers_replicated = (distribution_centers_replicated.
  withColumn("row_id", f.monotonically_increasing_id()).
  withColumn('row_num', f.row_number().over(Window.orderBy('row_id'))).
  drop(f.col("row_id"))
                  )

store_table = (store_table.
                            withColumn("row_id", f.monotonically_increasing_id()).
                            withColumn('row_num', f.row_number().over(Window.orderBy('row_id'))).
                            drop(f.col("row_id"))
                           )


distribution_center_to_store_mapping_table = store_table.join(distribution_centers_replicated, ("row_num")).drop(f.col("row_num"))
store_table = store_table.drop(f.col("row_num"))
distribution_centers_replicated = distribution_centers_replicated.drop(f.col("row_num"))

display(distribution_center_to_store_mapping_table)

# COMMAND ----------

# MAGIC %md
# MAGIC Save as a Delta table

# COMMAND ----------

demand_df_delta_path = os.path.join(cloud_storage_path, 'demand_df_delta')

# COMMAND ----------

# Write the data 
demand_df.write \
.mode("overwrite") \
.format("delta") \
.save(demand_df_delta_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.part_level_demand")
spark.sql(f"CREATE TABLE {dbName}.part_level_demand USING DELTA LOCATION '{demand_df_delta_path}'")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dbName}.part_level_demand"))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

print("Ending ./_resources/01-data-generator")
