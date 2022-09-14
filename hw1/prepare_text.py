#препроцессинг
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import pymorphy2
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from tqdm import tqdm

punctuation += '«»' + '...'
stop_words = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()

vectorizer = CountVectorizer(analyzer='word')

def preprocess(text):
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

#матрица
def inverted_index_matrix(corpus, vectorizer=vectorizer):
    X = vectorizer.fit_transform(corpus)
    return X.toarray(), vectorizer.get_feature_names_out()


# словарь вида слово: *лист кортежей (номер документа, количество вхождений в документе)*
def inverted_index_dict(corpus, vectorizer=vectorizer):
    X = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names_out()
    inverted_index_dict = defaultdict(list)

    for i in range(len(words)):
        arr = X.toarray()[:, i]
        for j in range(len(arr)):
            if arr[j] != 0:
                inverted_index_dict[words[i]].append((j, arr[j]))
    return inverted_index_dict