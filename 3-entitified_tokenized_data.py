import pandas as pd
import ast
from transformers import pipeline


ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")

df['Tokens'] = df['Tokens'].apply(ast.literal_eval)  # convert string to list




def chunk_tokens(tokens, chunk_size=128):
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]


def extract_ner(tokens):
    all_entities = []
    i=1
    for chunk in chunk_tokens(tokens, chunk_size=512):

        print(i)
        i+=1
        text = " ".join(chunk)
        ner_results = ner_pipeline(text)
        
        all_entities.extend([(ent['entity'],ent['word']) for ent in ner_results])
    return all_entities

df['Entities'] = df['Tokens'].apply(extract_ner)

df.to_csv("annual_reports_dataset_cleaned_tokenized_entitified.csv")
