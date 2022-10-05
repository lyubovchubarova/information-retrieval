#препроцессинг
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import pymorphy2
from string import punctuation
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import json
from typing import List, Tuple

punctuation += '«»' + '...'
stop_words = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()


def preprocess(text: str) -> str:
    """
    Препроцессинг текста: очистка от пунктуации, удаление стоп-слов, лемматизация
    """

    text = text.replace('\n\n', '\n')

    # токенизация
    text = nltk.word_tokenize(text, language='russian')

    processed_text = []

    for token in text:
        if token not in punctuation:  # избавление от пунктуации
            token = morph.parse(token.lower())[0].normal_form  # приведение к нижнему регистру и лемматизация
            if token not in stop_words:  # избавление от стоп-слов
                processed_text.append(token)

    return ' '.join(processed_text)



def extract_documents(path: str, number_of_documents: int) -> Tuple[List[str], List[str]]:
    """
    Получение списков названий документов и текстов документов
    """

    with open(path,'r', encoding='utf-8') as f:
        corpus = list(f)[:number_of_documents]
    names_of_documents = []
    texts_of_documents = []

    for item in tqdm(corpus):

        item = json.loads(item)
        answers = item['answers']
        if answers:
            answer = sorted([answer for answer in answers if answer['author_rating']['value']],
                            key = lambda x: x['author_rating']['value'],
                            reverse = True)[0]['text'] #выбор ответа с самым высоким рейтингом у автора ответа
            names_of_documents.append(answer)
            texts_of_documents.append(preprocess(answer))

    return names_of_documents, texts_of_documents




def corpus_indexing(corpus: List[str]) -> Tuple[sparse.csr_matrix, CountVectorizer]:
    """
    Индексация корпуса с помощью BM25.
    Возвращает корпус и обученный векторайзер для запроса
    """

    #константы
    k = 2
    b = 0.75

    # подсчет tf
    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(corpus)
    tf = count

    # подсчет idf
    tfidf_vectorizer = TfidfVectorizer(norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_ #np.expand_dims(idf, axis=0)

    len_d = tf.sum(axis=1)
    avgdl = len_d.mean()
    const_denominator = (k * (1 - b + (b * len_d / avgdl)))

    # формирование sparse матрицы
    for i, j in tqdm(zip(*tf.nonzero())):
        tf[i, j] = (tf[i, j] * idf[j] * (k+1)) / (tf[i, j] + const_denominator[i])

    return tf, count_vectorizer
