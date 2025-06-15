import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment Verileri Okunur
df = pd.read_csv("sentiment-v4.csv")

# Year sütunu integer değere çevrilir
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Sentiment sütunlarının etiketleri
sentiment_columns = ['Negative', 'Neutral', 'Positive', 'Compound', 'Subjectivity']

# Grafik stilini belirle
sns.set(style="whitegrid")
print(df[sentiment_columns].head())

# Her bir sentiment sütunu için ayrı grafik çiz
for sentiment in sentiment_columns:
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df,
        x="Year",
        y=sentiment,
        hue="Company",
        marker="o"
    )
    ax.set_title(f"{sentiment} over Years by Company")
    ax.set_ylabel(f"{sentiment} Score")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 1 if sentiment != "Compound" else -1.1)  # Compound -1 ile 1 arasında, diğerleri 0-1
    plt.legend(title="Company", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
