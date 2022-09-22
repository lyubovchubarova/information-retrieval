from prepare_text import *
from process_request import *

import os
from yaml import safe_load
from sklearn.feature_extraction.text import TfidfVectorizer

with open('config.yml', 'r') as f:
  config = safe_load(f)


def create_corpus(path = config['PARAMS']['path_to_files']):
    # создание и предобработка корпуса
    corpus = []
    documents_names = []

    for root, dirs, files in os.walk(path):
        for name in tqdm(files):
            with open(os.path.join(root, name), encoding='utf-8') as f:
                text = f.read()
                corpus.append(preprocess(text))
                documents_names.append(name)

    vectorizer = TfidfVectorizer()
    matrix = create_tfidf_matrix(corpus, vectorizer)

    return matrix, documents_names, vectorizer

def main():

    matrix, document_names, vectorizer = create_corpus()

    while True:
        #берем реквест
        request = input('Введите запрос: ')
        #ищем в готовом корпусе
        search(preprocess(request), vectorizer, matrix, document_names)

if __name__ == "__main__":
    main()