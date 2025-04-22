from pipeline.preprocessing import preprocess_data
import pandas as pd

df = pd.read_csv("data/akc-data-latest.csv")
df = preprocess_data(df)
print(df.head())