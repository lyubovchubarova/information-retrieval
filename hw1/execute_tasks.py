import numpy as np
from collections import defaultdict

names = {
    'Моника': ['моника', 'мон'],
    'Рейчел': ['рейчел', 'рейч'],
    'Чендлер': ['чендлер', 'чэндлер', 'чен'],
    'Фиби': ['фиби', 'фибс'],
    'Росс': ['росс'],
    'Джоуи': ['джоуи', 'джои', 'джо']
}

#самое популярное слово - матрица
def most_common_matrix(matrix, words):
    sum_of_occurrences_matrix = np.sum(matrix, axis=0)  # сумма всех вхождений для каждого слова
    list_most_common = []
    # массив с индексами элементов-максимумов
    for elem in np.where(sum_of_occurrences_matrix == sum_of_occurrences_matrix[np.argmax(sum_of_occurrences_matrix)])[0]:
        list_most_common.append(words[elem])
    return list_most_common

#самое популярное слово - словарь
def most_common_dict(dictionary):
    list_most_common = []
    sum_of_occurrences_dict = {word: sum([pair[1] for pair in dictionary[word]]) for word in dictionary}
    max_occurence = max(list(sum_of_occurrences_dict.values()))
    for elem in sum_of_occurrences_dict:
        if sum_of_occurrences_dict[elem] == max_occurence:
            list_most_common.append(elem)
    return list_most_common

#самые редкие слова - матрица
def least_common_matrix(matrix, words):
    sum_of_occurrences_matrix = np.sum(matrix, axis=0)  # сумма всех вхождений для каждого слова
    less_popular_matrix = []
    # массив с индексами элементов-минимумов
    for elem in np.where(sum_of_occurrences_matrix == sum_of_occurrences_matrix[np.argmin(sum_of_occurrences_matrix)])[0]:
        less_popular_matrix.append(words[elem])
    return less_popular_matrix

#самые редкие слова - словарь
def least_common_dict(dictionary):
    sum_of_occurrences_dict = {word: sum([pair[1] for pair in dictionary[word]]) for word in dictionary}
    min_occurence = min(list(sum_of_occurrences_dict.values()))
    less_popular_dict = []
    for elem in sum_of_occurrences_dict:
        if sum_of_occurrences_dict[elem] == min_occurence:
            less_popular_dict.append(elem)
    return less_popular_dict

#набор слов во всех документах коллекции - матрица
def words_in_every_doc_matrix(matrix, words):
    #возвращаем все слова, у которых 165 ненулевых вхождений в матрице
    return [words[elem] for elem in np.where(np.count_nonzero(matrix, axis=0) == 165)[0]]

#набор слов во всех документах коллекции - словарь
def words_in_every_doc_dict(dictionary):
    return [elem for elem in dictionary if len(dictionary[elem]) == 165]

#самый популярный герой - матрица

def most_popular_character_matrix(matrix, words, names = names):
    # словарь персонаж: индексы вариантов имен
    names_indexes = {name: [np.where(np.array(words) == lil_name)[0][0] for lil_name in names[name] if
                            np.where(np.array(words) == lil_name)[0].size > 0] for name in names}
    # словарь имя: сумма количества вхождений вариантов имен
    number_of_occurrences_by_matrix = {
        name: sum([np.sum(matrix[:, name_index]) for name_index in names_indexes[name]]) for name in
        names_indexes}
    return max(number_of_occurrences_by_matrix, key=number_of_occurrences_by_matrix.get)

#самый популярный герой - словарь
def most_popular_character_dict(dictionary, names = names):
    # для каждого имени посчитаем количество вхождений всех имен
    number_of_occurrences_by_dict = defaultdict(int)
    for name in names:
        for lil_name in names[name]:
            number_of_occurrences_by_dict[name] += sum([pair[1] for pair in dictionary[lil_name]])
    return max(number_of_occurrences_by_dict, key=number_of_occurrences_by_dict.get)