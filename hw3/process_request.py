import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from typing import List

#получение вектора из запроса
def get_vector_from_request(request: str,
                            vectorizer: CountVectorizer) -> np.array:
    return vectorizer.transform([request])

#поиск по запросу в корпусе
def search(request: str,
           vectorizer: CountVectorizer,
           matrix: sparse.csr_matrix):

    #получаем вектор запроса
    request_vector = get_vector_from_request(request, vectorizer)

    #для каждого документа получаем оценку по BM-25
    doc_scores = np.dot(matrix, request_vector.T)

    return doc_scores

def print_results(doc_scores: np.array,
                  documents_names: List[str],
                  number_of_results: int):

    # индексы документов c ненулевой оценкой близости
    documents_indexes = doc_scores.nonzero()[0]

    #массив с оценками
    doc_scores = doc_scores[doc_scores.nonzero()].A.ravel()
    # получаем индексы элементов массива, отсортированные в порядке убывания оценкц
    sorted_indexes = np.argsort(doc_scores)[::-1]

    # массив с индексами документов, в порядке от максимального соответсвия к минимальному
    ranked = documents_indexes[sorted_indexes]

    if number_of_results > len(ranked) and len(ranked) != 0:
        number_of_results = len(ranked)

    if len(ranked) == 0:
        print('Ничего не нашлось!')
    else:
        print(*[documents_names[index] for index in ranked[:number_of_results]], sep='\n')




