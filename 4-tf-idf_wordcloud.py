import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#TF-IDF WORDCLOUD GORSELLESTIRME
df = pd.read_csv("annual_reports_dataset_cleaned_tokenized_entitified.csv")  # Replace with your actual file

import ast




vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df["TextContent"])
feature_names = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)


tfidf_df["Company"] = df["Company"]
tfidf_df["Year"] = df["Year"]

print(tfidf_df.head())

# Company sütununa göre gruplama 
grouped = tfidf_df.groupby("Company",sort=False)
n = len(grouped)
cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
axes = axes.flatten() 

for ax in axes[n:]:
    ax.axis('off')  

for i, (company, group_df) in enumerate(grouped):
    word_scores = group_df[feature_names].mean().to_dict()
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white'
    ).generate_from_frequencies(word_scores)

    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(str(company), fontsize=14, pad=0)  

plt.subplots_adjust(hspace=0.8, wspace=0.3) 
plt.tight_layout()
plt.show()
plt.show()
