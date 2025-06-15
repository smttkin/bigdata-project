import pandas as pd
from transformers import pipeline
import torch 
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Set device (0 = first GPU, -1 = CPU)
device = 0 if torch.cuda.is_available() else -1

# Load dataset
df = pd.read_csv("annual_reports_dataset.csv")

# Load FinBERT sentiment analysis pipeline
classifier = pipeline("text-classification", model="DataWizardd/finbert-sentiment-binary",tokenizer="DataWizardd/finbert-sentiment-binary",device=device, batch_size=512,truncation=True)

i = 1
def get_sentiment_chunked(text, chunk_size=512):
    global i
    print(i)
    i += 1
    text = str(text)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    pos_total = 0
    neu_total = 0
    neg_total = 0
    for chunk in chunks:
        results = classifier(chunk)
        for result in results:
            label = result['label'].lower()
            score = result['score']
            if label == 'positive':
                pos_total += score
            elif label == 'neutral':
                neu_total += score
            elif label == 'negative':
                neg_total += score
    n = len(chunks)
    pos_avg = pos_total / n
    neu_avg = neu_total / n
    neg_avg = neg_total / n
    compound_avg = pos_avg - neg_avg
    subjectivity = 1 - abs(compound_avg)
    print(neg_avg, neu_avg, pos_avg, compound_avg, subjectivity)
    return pd.Series([neg_avg, neu_avg, pos_avg, compound_avg, subjectivity])

df[['Negative', 'Neutral', 'Positive', 'Compound', 'Subjectivity']] = df['TextContent'].apply(get_sentiment_chunked)

print(df.head())

df.to_csv("sentiment-v4.csv",index=False)