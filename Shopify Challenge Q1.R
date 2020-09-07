library(mlbench)
library(tidyverse)
library(caret)
library(Boruta)
library(dplyr)
library(tidyverse) # Load Libraries
library(dplyr)
library(knitr)
library(corrplot)
library(glmnet)
library(mice)
library(ggplot2)
library(Amelia)
library(readr)
library(readxl)


df <- read_excel(file.choose(),sheet=1)
summary(df)

missing_values <- sort(sapply(df, function(x) { sum(is.na(x)) }), decreasing=TRUE) ## I check to see if there are any missing values in the dataset
missing_values # No missing values

range(df$order_id) ## 5,000 orders were placed in the 30-day window period 
range(df$user_id)
range(df$order_amount)  ## This shows that  the orders go from $90 all the way through $704,000. A high value raises a red flag.
range(df$total_items)## This shows that customers get from a single pair (1) of sneakers all the way through 2,000 sneakers



ggplot(df,aes(y=order_amount,x=total_items))+ggtitle("Outliers")+geom_point() ## This line of code shows us that there are outliers in our dataset.

unique(df$order_amount) ## This line returns the $ order amount. Most amount seem reasonable except for a few ones such as: 704000,25725,51450,154350. These order amounts need to be further investigated
unique(df$total_items) ## This line of code shows that total items are: 1,2,3,4,5,6,8,2000. An order of 2000 must be a  mistake and should be further investigated


df_new <- subset(df, total_items!="2000" & order_amount<=2000) ## I create a different subset by excluding the outliers in the dataset
unique(df_new$order_amount)
unique(df_new$total_items)
ggplot(df_new,aes(y=order_amount,x=total_items))+ggtitle("No Outliers")+geom_point() ## This new plot shows no more outliers present



#### Calculate the new AOV would be
mean(df_new$total_items) # On average, 2 sneakers are purchased per order
AOV<-sum(df_new$order_amount)/4937
AOV ## An AOV of $302.58 ??? $300. This value makes sense if we take into consideration that the average number of sneakers per order is 2.

