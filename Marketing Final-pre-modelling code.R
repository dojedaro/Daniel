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


df <- read.csv(choose.files(),sep=",",header=TRUE)

df<- subset( df,select=-c(year))

head(df)
summary(df)
str(df)


table(df$age) ## observe what the different values each feature has
table(df$marital)
table(df$default)
table(df$housing)
table(df$loan)
table(df$balance)
table(df$contact)
table(df$job)
table(df$education)
table(df$month)
table(df$duration)
table(df$campaign)
table(df$pdays)
table(df$poutcome)
table(df$y)
table(df$previous)
table(df$day)


###check all str types of our df dataset:
unique(lapply(df,class))



#variables that are numeric: 
names(df[,which(sapply(df, class) %in% c("numeric","integer"))])  
#"ID"       "age"      "balance"  "day"      "duration" "campaign" "pdays"    "previous"

#variables that are categorical:
names(df[,which(sapply(df, class) %in% c("factor"))])  
# "job"       "marital"   "education" "default"   "housing"   "loan"      "contact"   "month"     "poutcome"   "y" 



####source and timeline:  Please refer to the following link. This link directly take you  <- https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# The instances are ordered by data from May 2008 to November 2010



##convert all defined categorical variables to character format
df  = df %>% 
  dplyr::rename(term_deposit=y) %>%     ##rename target variable y to term_deposit
  mutate_if(is.character, as.factor)

convert<-ifelse(df$term_deposit=="no",0,1)
df$term_deposit<-convert
df$term_deposit<- as.factor(df$term_deposit)




df[ df == "unknown" ] <- NA

missing_values <- sort(sapply(df, function(x) { sum(is.na(x)) }), decreasing=TRUE) #check to see if there are any missing values and display the frequency in descending order
missing_values 



#df$education <- ifelse(df$education == "primary", 10,
 #                            ifelse(df$secondary == "secondary", 50,
  #                                  ifelse(df$tertiary == "tertiary", 100, NA)))

#### Group the age feature and pdays feature into groups and we can then convert them to categorical

range(df$age)


agebreaks <- c(15,25,30,35,40,45,50,55,60,100)
agelabels <- c("15-25","26-30","31-35","36-40","41-45","46-50", "51-55", "56-60", "61+")
age_group <- as.numeric(df$age)
df<-cbind(df,age_group)
df$age_group <- cut(df$age_group, breaks = agebreaks, labels = agelabels, right = FALSE)
table(df$age_group) 

df$age_group<- as.factor(df$age_group)
#15-20  21-25  26-30  31-35  36-40  41-45  46-50  51-55  56-60  61-65  66-70  71-75 
#47    762   4464   9740   8349   6185   5470   4488   3922    974    256    254 
#76-80  81-85  86-90  91-95 96-100 
#170     98     23      7      2 


##then group pdays into different intervals
range(df$pdays)
#]  -1 871
pdaysbreaks <- c(-1,1,30,60,90,120,150,180,210,240,270,300,330,360,390,871)
pdayslabels <- c("no_contact","1-30","31-60","61-90","91-120","121-150","151-180","181-210","211-240",
                 "241-270","271-300","301-330","331-360","361-390","390+")

df$pdays <- as.numeric(df$pdays)
df$pdays <- cut(df$pdays, breaks = pdaysbreaks, labels = pdayslabels, right = FALSE)
df$pdays
df$pdays <- as.factor(df$pdays)
table(df$pdays)
#no_contact       1-30      31-60      61-90     91-120    121-150    151-180    181-210    211-240    241-270    271-300 
#36954        187        100        377       1215        424        850       1281        210        605        511 
#301-330    331-360    361-390       390+ 
#  483       1088        663        262 



library(tidyverse)
library(lubridate)

#df$date <- paste(df$year, df$mon, df$day, sep="-") %>% ymd() %>% as.Date()
#df$date <-as.factor(df$date)



###############data explotory and data visualization:
library(ggplot2)
library(DataExplorer)
#install.packages("gdata")
library(gdata)

plot_missing(df) ## Are there missing values, and what is the missing data profile?


plot_bar(df) ## How does the categorical frequency for each discrete variable look like?
plot_histogram(df) ## What is the distribution of each continuous variable?
plot_bar(df$term_deposit)
#Visualize correlation matrix between numeric variables:
num_var<-c("age","balance","day","duration","campaign","pdays","previous")
df_numvar<-df[,num_var]
str(df_numvar)
M<-cor(df_numvar)
head(round(M,2))

library(corrplot)
corrplot(M, method="pie")

###Let's plot the target variable y against all  numeric variables:
plot_boxplot(df,by="term_deposit")       
###we may be able to see if each variable has impact on the term_deposit decision: ie. "previous"shows previous cx contacts do make ppl to make term_deposit more likely







table(df$term_deposit) #y only has 2 levels-yes/no: 88.3% y=no=0 and 11.7% y=yes=1, we conclude the original dataset is imbalanced. 
##0     1 
#39922  5289 


df<- subset(df, select = -c(duration, poutcome))



# Create a custom function to fix missing values ("NAs") and preserve the NA info as surrogate variables
fixNAs<-function(data_frame){
  # Define reactions to NAs
  integer_reac<-0
  factor_reac<-"FIXED_NA"
  character_reac<-"FIXED_NA"
  date_reac<-as.Date("1900-01-01")
  # Loop through columns in the data frame and depending on which class the variable is, apply the defined reaction and create a surrogate
  
  for (i in 1 : ncol(data_frame)){
    if (class(data_frame[,i]) %in% c("numeric","integer")) {
      if (any(is.na(data_frame[,i]))){
        data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
          as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
        data_frame[is.na(data_frame[,i]),i]<-integer_reac
      }
    } else
      if (class(data_frame[,i]) %in% c("factor")) {
        if (any(is.na(data_frame[,i]))){
          data_frame[,i]<-as.character(data_frame[,i])
          data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
            as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
          data_frame[is.na(data_frame[,i]),i]<-factor_reac
          data_frame[,i]<-as.factor(data_frame[,i])
          
        } 
      } else {
        if (class(data_frame[,i]) %in% c("character")) {
          if (any(is.na(data_frame[,i]))){
            data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
              as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
            data_frame[is.na(data_frame[,i]),i]<-character_reac
          }  
        } else {
          if (class(data_frame[,i]) %in% c("Date")) {
            if (any(is.na(data_frame[,i]))){
              data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
                as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
              data_frame[is.na(data_frame[,i]),i]<-date_reac
            }
          }  
        }       
      }
  } 
  return(data_frame) 
}

df<-fixNAs(df) #Apply fixNAs function to the data to fix missing values


combinerarecategories<-function(data_frame,mincount){ 
  for (i in 1 : ncol(data_frame)){
    a<-data_frame[,i]
    replace <- names(which(table(a) < mincount))
    levels(a)[levels(a) %in% replace] <-paste("Other",colnames(data_frame)[i],sep=".")
    data_frame[,i]<-a }
  return(data_frame) }


df<-combinerarecategories(df,10) 


summary(df)
write.csv(df,"cleanedup_dataset.csv")

