import pandas as pd

# read data
df=pd.read_excel("fraud_data.xlsx")

# convert gender to bool
df["gender"]=[1 if i=="F" else 0 for i in df.gender]
