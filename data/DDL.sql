
CREATE DATABASE IF NOT EXISTS sbk_banking LOCATION 's3a://sbk-buckets/banking';


CREATE DATABASE IF NOT EXISTS sbk_banking LOCATION 's3a://partners-buckets/partners-ab-05/sbk/banking';


CREATE DATABASE IF NOT EXISTS sbk_banking LOCATION 's3a://sprakash-us-east/sbk-banking/db';

use sbk_banking;



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


create table if not exists customers_marketing as
select
  *
from
  customers_combined



