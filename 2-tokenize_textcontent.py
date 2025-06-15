import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

df = pd.read_csv('annual_reports_dataset_cleaned.csv')

df['Tokens'] = df['TextContent'].apply(word_tokenize)



df.to_csv("annual_reports_dataset_cleaned_tokenized.csv",index=False)
