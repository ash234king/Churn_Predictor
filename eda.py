import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("data/Churn_Modelling.csv")
print(df.size) ## 140000 records
print(df.columns) ## 14 columns
print(df.info())

print(df.head())

print(df.describe())

print(df.isnull().sum())

df_numerical=df.select_dtypes(include=['number'])
corr=df_numerical.corr()

plt.figure(figsize=(24,30))
sns.heatmap(corr,annot=True)
plt.show()
## so from the heatmap we can observe that Age is highly resulted to the column exited(the output coloumn)

## So in this i have kept eda for age as the major factor since its highly dependent on my output feature exited than other features
counter=df['Age'].value_counts()
labels=counter.index
values=counter.values
print(counter)

plt.figure(figsize=(24,20))
sns.barplot(x=labels,y=values)
plt.title('Bar Chart for Age')
plt.xlabel('Age')
plt.ylabel('Counts')
plt.show()
## Insights 
## people those who are with the bank there ages mostly between 23-52
## people with age 37 are maximum

counter=counter.head(20)
labels=counter.index
values=counter.values
plt.figure(figsize=(24,20))
plt.pie(values,labels=labels,autopct='%1.1f%%')
plt.title('Pie chart for top 20 Age categories')
plt.show()
## Insights 
## people with ages 37, 38,35 are maximum


##Box plots
sns.boxplot(x='Exited',y='Age',data=df)
plt.title('Box plot for Age with respect to exited')
plt.show()
## Insights
## When the person is not likely to churn then there are many outliers compared to when the person is likely to churn
## when the person is not likely to churn then the median of those person ages is 35 and when they are likely to churn their ages median is 45


## violin plot
sns.violinplot(y=df['Age'],color='green')
plt.title('Violin plot for Age')
plt.show()
## the spread of age is mostly between 30-40

counter=df['Tenure'].value_counts()
labels=counter.index
values=counter.values
print(counter)

plt.figure(figsize=(24,20))
sns.barplot(x=labels,y=values)
plt.title('Bar Chart for Tenure')
plt.xlabel('Tenure')
plt.ylabel('Counts')
plt.show()
## Insights 
## people those who are with the bank have a maximum tenure of 1-2 years 
## people with tenure of 2 years are maximum

counter=counter.head(20)
labels=counter.index
values=counter.values
plt.figure(figsize=(24,20))
plt.pie(values,labels=labels,autopct='%1.1f%%')
plt.title('Pie chart for Tenure categories')
plt.show()
## Insights 
## people with tenure of 1 to 2 years are maximum i.e. 10.4% and 10.5%

##Box plots
sns.boxplot(x='Exited',y='Tenure',data=df)
plt.title('Box plot for Tenure with respect to exited')
plt.show()
## Insights
## When the person is not likely to churn then there are many outliers compared to when the person is likely to churn
## when the person is not likely to churn then the median of those person ages is 35 and when they are likely to churn their ages median is 45


## violin plot
sns.violinplot(y=df['Tenure'],color='green')
plt.title('Violin plot for Tenure')
plt.show()
## in this the spread is majorly at 2 but we can ignore that as there is not much difference between other values so we can say that they mostly have an equal spread