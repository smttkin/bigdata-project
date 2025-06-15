import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("annual_reports_dataset.csv")
analyzer = SentimentIntensityAnalyzer()


i=1
def get_sentiment_chunked(text, chunk_size=512):
    global i
    print(i)
    i+=1
    text = str(text)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    neg_total = 0
    neu_total = 0
    pos_total = 0
    compound_total = 0

    for chunk in chunks:
        scores = analyzer.polarity_scores(chunk)
        neg_total += scores['neg']
        neu_total += scores['neu']
        pos_total += scores['pos']
        compound_total += scores['compound']

    n = len(chunks)
    neg_avg = neg_total / n
    neu_avg = neu_total / n
    pos_avg = pos_total / n
    compound_avg = compound_total / n
    
    # Subjectivity: 1 - |compound|
    subjectivity = 1 - abs(compound_avg)
    
    return pd.Series([neg_avg, neu_avg, pos_avg, compound_avg, subjectivity])
df[['Negative', 'Neutral', 'Positive', 'Compound', 'Subjectivity']] = df['TextContent'].apply(get_sentiment_chunked)

df.to_csv("sentiment-v2.csv", index=False)
