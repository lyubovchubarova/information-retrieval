#препроцессинг
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import pymorphy2
from string import punctuation
from tqdm import tqdm

punctuation += '«»' + '...'
stop_words = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()


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

def create_tfidf_matrix(corpus, vectorizer):
    X = vectorizer.fit_transform(corpus)
    return X.toarray()