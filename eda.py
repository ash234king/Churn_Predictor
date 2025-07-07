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



