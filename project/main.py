from flask import Flask, request, render_template
import random
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist

import idioms
import process_request
import prepare_corpus

from initialisation import  vectorizer_tfidf,\
                            matrix_tfidf,\
                            vectorizer_bm25,\
                            matrix_bm25, \
                            documents_names, \
                            number_of_results,\
                            model,\
                            tokenizer,\
                            matrix_bert


app = Flask(
    __name__, template_folder="./frontend/templates", static_folder="./frontend/static"
)


@app.route("/")
def my_form():
    return render_template("search.html")


def search_func(request,
           type,
           vectorizer_tfidf=vectorizer_tfidf,
           matrix_tfidf=matrix_tfidf,
           vectorizer_bm25=vectorizer_bm25,
           matrix_bm25=matrix_bm25,
           model=model,
           tokenizer=tokenizer,
           matrix_bert=matrix_bert,
           documents_names=documents_names,
           number_of_results=number_of_results):

    ans = []
    time = 0.1

    if type == 'TfIdf':
        start_time = datetime.now()
        request_vector = process_request.get_vector_from_request_tfidf(request, vectorizer_tfidf)
        doc_scores = np.dot(matrix_tfidf, request_vector.T)
        documents_indexes = doc_scores.nonzero()[0]
        doc_scores = doc_scores[doc_scores.nonzero()].A.ravel()
        sorted_indexes = np.argsort(doc_scores)[::-1]
        ranked = documents_indexes[sorted_indexes]
        ans = [documents_names[index] for index in ranked]
        time = datetime.now() - start_time

        return len(ans), ans, time

    elif type == 'BM-25':
        start_time = datetime.now()
        request_vector = process_request.get_vector_from_request_bm25(request, vectorizer_bm25)
        doc_scores = np.dot(matrix_bm25, request_vector.T)
        documents_indexes = doc_scores.nonzero()[0]
        doc_scores = doc_scores[doc_scores.nonzero()].A.ravel()
        sorted_indexes = np.argsort(doc_scores)[::-1]
        print(sorted_indexes)
        ranked = documents_indexes[sorted_indexes]
        ans = [documents_names[index] for index in ranked]
        time = datetime.now() - start_time

        return len(ans), ans, time


    elif type == 'BERT':
        start_time = datetime.now()
        request_vector = process_request.get_vector_from_request_bert(request, model, tokenizer)
        doc_scores = cdist(matrix_bert, request_vector, metric='cosine')
        doc_mask = np.argsort(doc_scores, axis=0)[::-1]
        ranked = np.array(documents_names)[doc_mask]
        ans = [answer for answer in ranked.reshape(-1)[::-1][:number_of_results]]
        time = datetime.now() - start_time

        return len(ans), ans, time

    return len(ans), ans, time

@app.route("/", methods=["POST"])
def my_form_post():
    search_text = request.form.get("search_text")
    search_type = request.form.get("search_type")
    search_results_len, search_results, search_results_time = search_func(prepare_corpus.preprocess(search_text),
                                                                          search_type)

    if search_results_len:
        return render_template(
            "results.html",
            search_results_len=search_results_len,
            search_results=search_results,
            search_text=search_text,
            search_type=search_type,
            search_results_time=search_results_time
        )
    else:
        idiom, idiom_author = random.choice(list(idioms.idioms.items()))
        return render_template(
            "404.html",
            idiom_author=idiom_author,
            idiom=idiom,
            search_text=search_text,
            search_type=search_type,
            search_results_time=search_results_time
        )

if __name__ == '__main__':
    import os
    app.run(debug=True, port = int(os.environ.get("PORT", 5001)))

