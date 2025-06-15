import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

df = pd.read_csv('annual_reports_dataset.csv')

def clean_text(text):
    if pd.isnull(text):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return ' '.join(cleaned_words)

df['TextContent'] = df['TextContent'].apply(clean_text)

df.to_csv('annual_reports_dataset_cleaned.csv', index=False)

print("✅ Temizleme işlemi tamamlandı. 'annual_reports_dataset_cleaned.csv' oluşturuldu.")
