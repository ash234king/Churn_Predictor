import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("Churn_Modelling.csv")
print(df.size)
print(df.columns)
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

## so we will check whether the data has balanced entries or not
