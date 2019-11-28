

CREATE DATABASE IF NOT EXISTS sbk_banking LOCATION 's3a://partners-buckets/partners-ab-05/sbk/banking';


CREATE DATABASE IF NOT EXISTS sbk_banking LOCATION 's3a://sprakash-us-east/sbk-banking/db/customers';

use sbk_banking;

CREATE EXTERNAL TABLE IF NOT EXISTS `customers` (
  `customer_id` int, 
  `gender` string, 
  `title` string, 
  `givenname` string, 
  `middleinitial` string, 
  `surname` string, 
  `streetaddress` string, 
  `city` string, 
  `state` string, 
  `statefull` string, 
  `zipcode` string, 
  `country` string, 
  `countryfull` string, 
  `emailaddress` string, 
  `username` string, 
  `password` string, 
  `telephonenumber` string, 
  `telephonecountrycode` int, 
  `mothersmaiden` string, 
  `dob` string, 
  `age` int, 
  `cctype` string, 
  `ccnumber` string, 
  `cvv2` string, 
  `ccexpires` string, 
  `nationalid` string, 
  `occupation` string, 
  `employer` string, 
  `vehicle` string, 
  `default` string, 
  `housing` string, 
  `loan` string,  
  `contact` string, 
  `month` string, 
  `day_of_week` string,  
  `duration` string,  
  `campaign` string,  
  `pdays` string, 
  `previous` string, 
  `poutcome` string,  
  `emp_var_rate` string,  
  `cons_price_idx` string,  
  `cons_conf_idx` string
)
ROW FORMAT DELIMITED
  FIELDS TERMINATED BY ','
  STORED AS TEXTFILE
  LOCATION 's3a://partners-buckets/partners-ab-05/sbk/banking/customers'
tblproperties("skip.header.line.count"="1");



CREATE EXTERNAL TABLE IF NOT EXISTS `customers` (
  `customer_id` int, 
  `gender` string, 
  `title` string, 
  `givenname` string, 
  `middleinitial` string, 
  `surname` string, 
  `streetaddress` string, 
  `city` string, 
  `state` string, 
  `statefull` string, 
  `zipcode` string, 
  `country` string, 
  `countryfull` string, 
  `emailaddress` string, 
  `username` string, 
  `password` string, 
  `telephonenumber` string, 
  `telephonecountrycode` int, 
  `mothersmaiden` string, 
  `dob` string, 
  `age` int, 
  `cctype` string, 
  `ccnumber` string, 
  `cvv2` string, 
  `ccexpires` string, 
  `nationalid` string, 
  `occupation` string, 
  `employer` string, 
  `vehicle` string, 
  `default` string, 
  `housing` string, 
  `loan` string,  
  `contact` string, 
  `month` string, 
  `day_of_week` string,  
  `duration` string,  
  `campaign` string,  
  `pdays` string, 
  `previous` string, 
  `poutcome` string,  
  `emp_var_rate` string,  
  `cons_price_idx` string,  
  `cons_conf_idx` string
)
ROW FORMAT DELIMITED
  FIELDS TERMINATED BY ','
  STORED AS TEXTFILE
  LOCATION 's3a://sprakash-us-east/sbk-banking/db/customers'
tblproperties("skip.header.line.count"="1");

# from combined with matching names

CREATE EXTERNAL TABLE IF NOT EXISTS `customers_combined` (
  `customer_id` int, 
  `title` string, 
  `givenname` string, 
  `surname` string, 
  `streetaddress` string, 
  `city` string, 
  `state` string, 
  `zipcode` string, 
  `country` string, 
  `emailaddress` string, 
  `telephonenumber` string, 
  `mothersmaiden` string, 
  `cctype` string, 
  `ccnumber` string, 
  `cvv2` string, 
  `ccexpires` string, 
  `nationalid` string,
  `age` int, 
  `job` string, 
  `marital` string, 
  `education` string, 
  `default` string, 
  `housing` string, 
  `loan` string,  
  `contact` string, 
  `month` string, 
  `day_of_week` string,  
  `duration` string,  
  `campaign` string,  
  `pdays` string, 
  `previous` string, 
  `poutcome` string,
  `emp_var_rate` string,
  `cons_price_idx` string,
  `cons_conf_idx` string
)
ROW FORMAT DELIMITED
  FIELDS TERMINATED BY ','
  STORED AS TEXTFILE
  LOCATION 's3a://sprakash-us-east/sbk-banking/db/customers2'
tblproperties("skip.header.line.count"="1");



CREATE EXTERNAL TABLE IF NOT EXISTS `customers_cut` 
AS select 
  `customer_id`, 
  `title`, 
  `givenname`, 
  `surname`, 
  `streetaddress`, 
  `city`, 
  `state`, 
  `zipcode`, 
  `country`, 
  `countryfull`, 
  `emailaddress`, 
  `telephonenumber`, 
  `mothersmaiden`, 
  `gender`, 
  `age`, 
  `cctype`, 
  `ccnumber`, 
  `cvv2`, 
  `ccexpires`, 
  `nationalid`, 
  `occupation`, 
  `employer`, 
  `vehicle`, 
  `default`, 
  `housing`, 
  `loan`,  
  `contact`, 
  `month`, 
  `day_of_week`,  
  `duration`,  
  `campaign`,  
  `pdays`, 
  `previous`, 
  `poutcome`,  
  `emp_var_rate`,  
  `cons_price_idx`,  
  `cons_conf_idx`
  from customers;


LOAD DATA INPATH 's3a://partners-buckets/partners-ab-05/sbk/banking/banking_combined.csv' OVERWRITE INTO TABLE customers; 




ROW FORMAT SERDE 
  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  's3a://all-se-env-1026/all-se-env-1026-dl/warehouse/tablespace/managed/hive/cdp_overview.db/customers'
TBLPROPERTIES (
  'bucketing_version'='2', 
  'transactional'='true', 
  'transactional_properties'='insert_only', 
  'transient_lastDdlTime'='1572631013')
