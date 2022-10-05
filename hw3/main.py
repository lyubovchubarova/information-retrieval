from yaml import safe_load

import prepare_documents
import process_request

with open('config.yml', 'r') as f:
  config = safe_load(f)


def create_corpus(path=config['PARAMS']['path_to_files'],
                  number_of_documents=config['PARAMS']['number_of_documents']):
  """
  Создание корпуса и приведение его в готовый для поиска формат матрицы
  """
  documents_names, corpus = prepare_documents.extract_documents(path, number_of_documents)
  corpus, vectorizer = prepare_documents.corpus_indexing(corpus)

  return documents_names, corpus, vectorizer

def main(number_of_results=config['PARAMS']['number_of_results']):

  documents_names, corpus, vectorizer = create_corpus()

  while True:

    query = input('Введите свой вопрос о любви: ')
    doc_scores = process_request.search(prepare_documents.preprocess(query), vectorizer, corpus)
    process_request.print_results(doc_scores, documents_names, number_of_results)


if __name__ == "__main__":
  main()