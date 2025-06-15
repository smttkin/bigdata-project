import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

df = pd.read_csv("annual_reports_dataset_cleaned_tokenized_entitified.csv")

def parse_entities(entities_str):
    if isinstance(entities_str, str):
        return ast.literal_eval(entities_str)
    return entities_str

df["Entities"] = df["Entities"].apply(parse_entities)


# TF-IDF hesapla
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Entities"].apply(lambda ents: " ".join([text for text, label in ents])))
feature_names = vectorizer.get_feature_names_out()

# TF-IDF skorlarını bir dict olarak her satıra ekle
tfidf_vectors = []
for row in tfidf_matrix:
    vector = row.toarray().flatten()
    tfidf_dict = {feature_names[i]: vector[i] for i in range(len(feature_names)) if vector[i] > 0}
    tfidf_vectors.append(tfidf_dict)

df["TFIDF"] = tfidf_vectors

df.to_csv("annual_reports_dataset_cleaned_tokenized_entitified_tfidf.csv", index=False)
