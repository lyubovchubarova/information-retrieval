import numpy as np
import torch
import pickle
from yaml import safe_load
from transformers import AutoTokenizer, AutoModel
import prepare_corpus
from scipy import sparse

with open('config.yml', 'r') as f:
    config = safe_load(f)

documents_names = prepare_corpus.extract_documents(path=config['PARAMS']['path_to_files'],
                                                           number_of_documents=config['PARAMS']['number_of_documents'])

number_of_results = config['PARAMS']['number_of_results']


matrix_tfidf = sparse.load_npz(config['PARAMS']['path_to_matrix_tfidf'])
with open(config['PARAMS']['path_to_vectorizer_tfidf'], 'rb') as f:
    vectorizer_tfidf = pickle.load(f)

matrix_bm25 = sparse.load_npz(config['PARAMS']['path_to_matrix_bm25'])
with open(config['PARAMS']['path_to_vectorizer_bm25'], 'rb') as f:
    vectorizer_bm25 = pickle.load(f)

matrix_bert = torch.load(config['PARAMS']['path_to_matrix_bert'], map_location=torch.device('cpu'))
matrix_bert = np.array([item.numpy() for item in matrix_bert])

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")




