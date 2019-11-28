import pandas as pd
import seaborn as sns
import pyspark.sql.functions;
from pyspark.sql.functions import rand, when;
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonPi")\
    .getOrCreate()


    

#spark.sql("SHOW databases").show()

spark.sql("use sbk_banking").show()

#spark.sql("SHOW tables").show()

#spark.sql("SHOW tables").show()

#spark.sql("desc customers_combined").show()


df = spark.sql('SELECT customer_id, title, givenname, surname  FROM customers_combined where title !="title" and surname!="" limit 20 ')


#df.show()

pdf_pre = df.toPandas()

pdf_pre.style.format({"surname": lambda x:x.upper()})\
    .format({"title": lambda x:x.upper()})

df2 = df.withColumn("acc", rand()/3)

#df2.show()

pdf = df2.toPandas()

#pdf[['acc']]

#pdf.tail()



cm = sns.light_palette("lightblue", as_cmap=True)
  


pdf.style.format({"surname": lambda x:x.upper()})\
    .format({"title": lambda x:x.upper()})\
    .background_gradient(cmap=cm, subset=['acc'])
  


#spark.sql("create table test1(id int)")

   
#spark.stop()

