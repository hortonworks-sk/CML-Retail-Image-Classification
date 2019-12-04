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


sdf = spark.sql('SELECT customer_id, title, givenname, surname  FROM customers_combined where title !="title" and surname!="" limit 20 ')


#df.show()

pdf_pre = sdf.toPandas()

pdf_pre.style.format({"surname": lambda x:x.upper()})\
    .format({"title": lambda x:x.upper()})

sdf2 = sdf.withColumn("conversion probability", rand()/3)

#df2.show()

pdf = sdf2.toPandas()

#pdf[['acc']]

#pdf.tail()



cm = sns.light_palette("lightblue", as_cmap=True)
  


pdf.style.format({"surname": lambda x:x.upper()})\
    .format({"title": lambda x:x.upper()})\
    .background_gradient(cmap=cm, subset=['conversion probability'])
  


#spark.sql("create table test1(id int)")

   
#spark.stop()

