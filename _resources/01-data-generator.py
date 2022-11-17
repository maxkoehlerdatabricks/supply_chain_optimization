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
# MAGIC # Hierarchical Time Series Generator
# MAGIC This notebook-section simulates hierarchical time series data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate demand series data

# COMMAND ----------

#################################################
# Python Packages
#################################################
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

# COMMAND ----------

# MAGIC %md
# MAGIC Parameters

# COMMAND ----------

n=3 # Number of replicates per product category
ts_length_in_weeks = 104 # Length of a time series in years
number_of_stores = 30

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

#Create a table
date_range_table = spark.createDataFrame(
  date_range,
  DateType()).toDF("date")


display(date_range_table)

# COMMAND ----------

# MAGIC %md
# MAGIC Simulate parameters for ARMA series

# COMMAND ----------

# MAGIC %md
# MAGIC To Do: Make more realistic

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

# Append column-wise
spark_df_helper = spark.createDataFrame(pdf_helper, schema=arma_schema)
spark_df_helper = spark_df_helper.withColumn("row_id", monotonically_increasing_id()).withColumn('row_num', f.row_number().over(Window.orderBy('row_id')))
products_in_stores_table = products_in_stores_table.withColumn("row_id", monotonically_increasing_id()).withColumn('row_num', f.row_number().over(Window.orderBy('row_id')))
products_in_stores_table = products_in_stores_table.join(spark_df_helper, ("row_num")).drop("row_id", "row_num")
display(products_in_stores_table)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC Generate Demand Series

# COMMAND ----------

import statsmodels.api as sm
import matplotlib.pyplot as plt


#################################################
# Generate an individual time series for each Product-SKU combination
#################################################

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
schema = StructType(  products_in_stores_table.schema.fields + 
                    [
                      StructField("date", DateType(), True),
                      StructField("demand", FloatType(), True),
                      StructField("row_number", FloatType(), True)
                    ])

# Generate an ARMA
#pdf = product_hierarchy_extended.toPandas().head(1)

# Generate a time series with random parameters
# @pandas_udf(schema, PandasUDFType.GROUPED_MAP)

def time_series_generator_pandas_udf(pdf):
  out_df = date_range.assign(
   demand = generate_arma(arparams = pdf.AR_Pars_RN.iloc[0], 
                          maparams= pdf.MA_Pars_RN.iloc[0], 
                          var = pdf.Variance_RN.iloc[0], 
                          offset = pdf.Offset_RN.iloc[0], 
                          number_of_points = date_range.shape[0], 
                          plot = False),
    product = pdf.product.iloc[0]
  )

  out_df = out_df[["product", "date", "demand"]]
  
  out_df["row_number"] = range(0,len(out_df))

  return(out_df)

# Apply the Pandas UDF and clean up
demand_df = ( 
  products_in_stores_table 
  .groupby("product") 
  .applyInPandas(time_series_generator_pandas_udf, sku_ts_schema) 
  .withColumn("Demand" , col("Demand") * col("Corona_Factor")) 
  .withColumn("Demand", when(col("Corona_Breakpoint_Helper") == 0,   
                             col("Demand") + trend_factor_before_corona * sqrt(col("Row_Number"))) 
                        .otherwise( col("Demand")))  
  .withColumn("Demand" , col("Demand") * col("Factor_XMas"))
  .withColumn("Demand" , round(col("Demand")))
  .select(col("Product"), col("SKU"), col("Date"), col("Demand") )
   )


display(demand_df)

# COMMAND ----------

# Plot individual series
res_table = demand_df.toPandas()
all_combis = res_table[[ "Product" , "SKU" ]].drop_duplicates()
random_series_to_plot = pd.merge(  res_table,   all_combis.iloc[[random.choice(list(range(len(all_combis))))]] ,  on =  [ "Product" , "SKU" ], how = "inner" )
selected_product = random_series_to_plot[ 'Product' ].iloc[0]
selected_sku = random_series_to_plot[ 'SKU' ].iloc[0]
random_series_to_plot = random_series_to_plot[["Date","Demand"]]

#Plot
plt.plot_date(random_series_to_plot.Date, random_series_to_plot.Demand, linestyle='solid')
plt.gcf().autofmt_xdate()
plt.title(f"Product: {selected_product}, SKU: {selected_sku}.")
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()

# COMMAND ----------

# Plot a sepecific time series
display(demand_df.join(demand_df.sample(False, 1 / demand_df.count(), seed=0).limit(1).select("SKU"), on=["SKU"], how="inner"))

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
# MAGIC # Simulate BoM Data
# MAGIC This notebook section simulates Bill-Of-Material data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate data

# COMMAND ----------

import string
import networkx as nx
import random
import numpy as np
import os

# COMMAND ----------

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generate_random_strings(n):
  random.seed(123)
  random_mat_numbers = set()
  while True:
    random_mat_numbers.add(id_generator(size=5))
    if len(random_mat_numbers) >= n:
      break
  return(random_mat_numbers)

# COMMAND ----------

def extend_one_step(node_from_):
  res_ = [  ]
  node_list_to_be_extended_ = [  ]
  # second level
  random_split_number = random.randint(2, 4)
  for i in range(random_split_number):
    node_to = random_mat_numbers.pop()
    node_list_to_be_extended_.append(node_to)
    res_.append((node_to, node_from_))
  return res_, node_list_to_be_extended_

# COMMAND ----------

def extend_one_level(node_list_to_be_extended, level, sku):
  
  
  print(f"""In  'extend_one_level': level={level} and sku = {sku}  """)
  
  if level == 1:
    head_node = random_mat_numbers.pop() 
    node_list_to_be_extended_one_level = [ ]
    node_list_to_be_extended_one_level.append(head_node)
    res_one_level = [ (head_node, sku) ]
  else:
    res_one_level = [ ]
    node_list_to_be_extended_one_level = [ ]
    
    if len(node_list_to_be_extended) > 2:
      node_list_to_be_extended_ = node_list_to_be_extended[ : 3 ]
    else:
      node_list_to_be_extended_ = node_list_to_be_extended

    for node in node_list_to_be_extended_:
      res_one_step = [ ]
      node_list_to_be_extended_one_step = [ ]
      
      res_one_step, node_list_to_be_extended_one_step = extend_one_step(node)    
      res_one_level.extend(res_one_step)
      node_list_to_be_extended_one_level.extend(node_list_to_be_extended_one_step)
  
  return res_one_level, node_list_to_be_extended_one_level

# COMMAND ----------

#Generate a set of material numbers
random_mat_numbers = generate_random_strings(1000000)

# COMMAND ----------

#Create a listof all SKU's
demand_df = spark.read.table(f"{dbName}.part_level_demand")
all_skus = demand_df.select('SKU').distinct().rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# Generaze edges
depth = 3
edge_list = [ ]

for sku in all_skus: 
  new_node_list = [ ]
  for level_ in range(1, (depth + 1)):
    new_edge_list, new_node_list = extend_one_level(new_node_list, level = level_, sku=sku)
    edge_list.extend(new_edge_list)

# COMMAND ----------

# Define the graph 
g=nx.DiGraph()
g.add_edges_from(edge_list)  

# COMMAND ----------

# Assign a quantity for the graph
edge_df = nx.to_pandas_edgelist(g)
edge_df = edge_df.assign(qty = np.where(edge_df.target.str.len() == 10, 1, np.random.randint(1,4, size=edge_df.shape[0])))

# COMMAND ----------

#Create the fnal mat number to sku mapper 
final_mat_number_to_sku_mapper = edge_df[edge_df.target.str.match('SRL|LRL|CAM|SRR|LRR_.*')][["source","target"]]
final_mat_number_to_sku_mapper = final_mat_number_to_sku_mapper.rename(columns={"source": "final_mat_number", "target": "sku"} )

# COMMAND ----------

#Create the fnal mat number to sku mapper
final_mat_number_to_sku_mapper = edge_df[edge_df.target.str.match('SRL|LRL|CAM|SRR|LRR_.*')][["source","target"]]
final_mat_number_to_sku_mapper = final_mat_number_to_sku_mapper.rename(columns={"source": "final_mat_number", "target": "sku"} )

# COMMAND ----------

# Create BoM
bom = edge_df[~edge_df.target.str.match('SRL|LRL|CAM|SRR|LRR_.*')]
bom = bom.rename(columns={"source": "material_in", "target": "material_out"} )

# COMMAND ----------

bom_df = spark.createDataFrame(bom) 
final_mat_number_to_sku_mapper_df = spark.createDataFrame(final_mat_number_to_sku_mapper)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register tables in database

# COMMAND ----------

bom_df_delta_path = os.path.join(cloud_storage_path, 'bom_df_delta')

# COMMAND ----------

# Write the data 
bom_df.write \
.mode("overwrite") \
.format("delta") \
.save(bom_df_delta_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.bom")
spark.sql(f"CREATE TABLE {dbName}.bom USING DELTA LOCATION '{bom_df_delta_path}'")

# COMMAND ----------

final_mat_number_to_sku_mapper_df_path = os.path.join(cloud_storage_path, 'sku_mapper_df_delta')

# COMMAND ----------

final_mat_number_to_sku_mapper_df.write \
.mode("overwrite") \
.format("delta") \
.save(final_mat_number_to_sku_mapper_df_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.sku_mapper")
spark.sql(f"CREATE TABLE {dbName}.sku_mapper USING DELTA LOCATION '{final_mat_number_to_sku_mapper_df_path}'")

# COMMAND ----------

display(spark.sql(f"select * from {dbName}.sku_mapper"))

# COMMAND ----------

display(spark.sql(f"select * from {dbName}.bom"))

# COMMAND ----------

print("Ending ./_resources/01-data-generator")

# COMMAND ----------


