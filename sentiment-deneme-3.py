import pandas as pd
import numpy as np

# CSV dosyası okunur
df = pd.read_csv("sentiment-v2.csv")

# Formül: estimated_compound = (pos - neg) / sqrt((pos - neg)^2 + 15)
def calculate_estimated_compound(row):
    pos = row['Positive']
    neg = row['Negative']
    score = pos - neg
    return score / np.sqrt(score**2 + 15)

df['Compound'] = df.apply(calculate_estimated_compound, axis=1)

df.to_csv("sentiment-v3.csv", index=False)

print(df.head())
