from transformers import AutoTokenizer, AutoModel
from yaml import safe_load
import torch
import numpy as np

import search_methods

def main():

    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    with open('config.yml', 'r') as f:
        config = safe_load(f)

    corpus_indexed = config['PARAMS']['corpus_indexed']

    if not corpus_indexed:
        documents_names, corpus = search_methods.extract_documents(path = config['PARAMS']['path_to_files'],
                                                                number_of_documents=config['PARAMS']['number_of_documents'])

        search_methods.bert_indexing(model=model.to(torch.device('cuda:0')),
                                     tokenizer=tokenizer,
                                     corpus=[search_methods.preprocess(item) for item in corpus],
                                     matrix_file=config['PARAMS']['path_to_matrix'])

        matrix = torch.load(config['PARAMS']['path_to_matrix'])
        matrix = np.array([item.numpy() for item in matrix])

    else:
        documents_names = []
        with open(config['PARAMS']['path_to_names'], encoding='utf-8') as f:
            for line in f:
                documents_names.append(line.replace('\n', ''))

        matrix = torch.load(config['PARAMS']['path_to_matrix'], map_location=torch.device('cpu'))
        matrix = np.array([item.numpy() for item in matrix])

    while True:
        request = input('Введите запрос: ')
        doc_scores = search_methods.bert_similarity(search_methods.preprocess(request),
                                                    model=model,
                                                    tokenizer=tokenizer,
                                                    matrix=matrix)

        doc_mask = np.argsort(doc_scores, axis=0)[::-1]
        ranked = np.array(documents_names)[doc_mask]
        print(*[answer for answer in ranked.reshape(-1)[:config['PARAMS']['number_of_results']]], sep='\n')

if __name__ == '__main__':
    main()