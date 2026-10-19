from sklearn.datasets import fetch_california_housing
import pandas as pd

data=fetch_california_housing()

df=pd.DataFrame(data.data,columns=data.feature_names)
print(df)
df['Price']=data.target 
print(df.shape)
print(df.describe())


print(df.head(10)) 