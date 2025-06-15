import spacy
import pandas as pd 
nlp = spacy.load("en_core_web_sm")


df=pd.read_csv("annual_reports_dataset_cleaned_tokenized_entitified.csv")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  

def chunk_text(text, max_chunk_size=100000):
    for i in range(0, len(text), max_chunk_size):
        yield text[i:i + max_chunk_size]



i = 1
def lemm(text):
    global i
    print(f"Processing chunk {i}")
    i += 1
    return " ".join([token.lemma_ for token in nlp(text)])

j=1

def lemmatize_large_text(text):
    global j
    print(j)
    j+=1
    lemmatized_chunks = [lemm(chunk) for chunk in chunk_text(text)]
    return " ".join(lemmatized_chunks)

df['TextContent'] = df['TextContent'].apply(lemmatize_large_text)
print(df.head()["TextContent"])
