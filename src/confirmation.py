import pandas as pd
df = pd.read_csv("data/dataset.csv")
print(df['label'].value_counts())
