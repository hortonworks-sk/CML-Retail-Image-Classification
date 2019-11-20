

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