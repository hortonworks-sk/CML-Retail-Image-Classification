# # Estimating $\pi$
#
# This is the simplest PySpark example. It shows how to estimate $\pi$ in parallel
# using Monte Carlo integration. If you're new to PySpark, start here!

from __future__ import print_function
import sys
from random import random
from operator import add
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonPi")\
    .getOrCreate()



spark.sql("SHOW databases").show()

spark.sql("use sbk_banking").show()

spark.sql("SHOW tables").show()

spark.sql("SELECT * FROM customers limit 10").show()


#spark.sql("create table test1(id int)")

   
spark.stop()