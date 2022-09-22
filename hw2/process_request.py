import numpy as np
from scipy.spatial.distance import cdist

#получение вектора из запроса
def get_vector_from_request(request, vectorizer):
    return vectorizer.transform([request]).toarray()

#поиск по запросу в корпусе
def search(request, vectorizer, matrix, documents_names):

    request_vector = get_vector_from_request(request, vectorizer)
    doc_scores = cdist(request_vector, matrix, metric='cosine')
    doc_mask = np.argsort(doc_scores)[::-1]
    ranked = np.array(documents_names)[doc_mask]

    print(*[elem.replace('.ru.txt', '') for elem in ranked[0]], sep = '\n')




