import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from typing import List
import torch

#получение вектора из запроса
def get_vector_from_request_tfidf(request, vectorizer):
    return vectorizer.transform([request])

def get_vector_from_request_bm25(request: str,
                            vectorizer: CountVectorizer) -> np.array:
    return vectorizer.transform([request])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_vector_from_request_bert(request, model, tokenizer):
    encoded_input = tokenizer(request, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    return mean_pooling(model_output, encoded_input['attention_mask'])