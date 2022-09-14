from prepare_text import *
from execute_tasks import *

import os
from yaml import safe_load

with open('config.yml', 'r') as f:
  config = safe_load(f)

def main(path = config['PARAMS']['path_to_files']):

    #создание и предобработка корпуса
    corpus = []


    for root, dirs, files in os.walk(path):
        for name in files:
            with open(os.path.join(root, name), encoding = 'utf-8') as f:
                text = f.read()
                corpus.append(preprocess(text))


    #создание матрицы и списка слов
    matrix, words = inverted_index_matrix(corpus)

    #создание словаря вид слово: *список пар (документ, количество вхождений)*
    dictionary = inverted_index_dict(corpus)

    #все с матрицей
    print('МАТРИЦА')
    print('Самое популярное слово: ', most_common_matrix(matrix, words))
    least_common_matrix_list = least_common_matrix(matrix, words)
    least_common_matrix_list_numered = [str(i+1) + '. ' + least_common_matrix_list[i] for i in range(len(least_common_matrix_list))]
    print('Наименее популярные слова: \n', '\n'.join(least_common_matrix_list_numered[:5]) + '\n...\n' + '\n'.join(least_common_matrix_list_numered[-5:]))
    print('Слова, которые встречаются в каждом документе: ', ', '.join(words_in_every_doc_matrix(matrix, words)))
    print('Самый популярный персонаж: ', most_popular_character_matrix(matrix, words))

    #все со словарем
    print('\nСЛОВАРЬ')
    print('Самое популярное слово: ', most_common_dict(dictionary))
    least_common_dict_list = least_common_dict(dictionary)
    least_common_dict_list_numered = [str(i + 1) + '. ' + least_common_matrix_list[i] for i in range(len(least_common_dict_list))]
    print('Наименее популярные слова: \n',
          '\n'.join(least_common_dict_list_numered[:5]) + '\n...\n' + '\n'.join(least_common_dict_list_numered[-5:]))
    print('Слова, которые встречаются в каждом документе: ', ', '.join(words_in_every_doc_dict(dictionary)))
    print('Самый популярный персонаж: ', most_popular_character_dict(dictionary))

if __name__ == '__main__':
    main()
