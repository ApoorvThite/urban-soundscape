import pandas as pd

df = pd.read_csv("data/311_mapped_clusters.csv")
print(df.columns)
print(df[["cluster"]].head())