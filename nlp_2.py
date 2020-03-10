# coding: utf8
import collections
import math
import string

import nltk
import pandas as pandas
import pymorphy2
from nltk.corpus import stopwords


def normalize_review(review, morph):
    tokens = nltk.word_tokenize(review)
    normalized_tokens = []
    normalized_review = ''
    for token in tokens:
        token = morph.parse(token)[0].normal_form
        normalized_tokens.append(token)
        if (token == ',' or token == '.'
                or token == '!' or token == '?'):
            normalized_review += token
        else:
            normalized_review += ' ' + token

    return normalized_review


def parse_all_reviews_of_one_movie(morph):
    excel_file = pandas.read_excel('reviews.xlsx', 0)
    reviews = excel_file['Отзыв']
    for review in reviews:
        normalized_review = normalize_review(review, morph)
        file = open('Все отзывы фильма ММС.txt', 'a', encoding='utf-8')
        file.write(normalized_review)
        file.write('\n\n\n')
        file.close()


def parse_all_negative_rewiews(morph):
    negative_reviews = []
    # с первого листа эксель таблицы
    negative_reviews.extend(pandas.read_excel('reviews.xlsx', 0)['Отзыв'][0:10:])
    # со второго листа эксель таблицы
    negative_reviews.extend(pandas.read_excel('reviews.xlsx', 1)['Отзыв'][0:10:])
    # с третьего листа эксель таблицы
    negative_reviews.extend(pandas.read_excel('reviews.xlsx', 2)['Отзыв'][0:10:])

    file = open('Все негативные отзывы.txt', 'w', encoding='utf-8')
    for review in negative_reviews:
        normalized_review = normalize_review(review, morph)
        file.write(normalized_review)
        file.write('\n\n\n')
    file.close()


def parse_all_positive_rewiews(morph):
    positive_reviews = []
    # с первого листа эксель таблицы
    positive_reviews.extend(pandas.read_excel('reviews.xlsx', 0)['Отзыв'][20:30:])
    # со второго листа эксель таблицы
    positive_reviews.extend(pandas.read_excel('reviews.xlsx', 1)['Отзыв'][20:30:])
    # с третьего листа эксель таблицы
    positive_reviews.extend(pandas.read_excel('reviews.xlsx', 2)['Отзыв'][20:30:])

    file = open('Все положительные отзывы.txt', 'w', encoding='utf-8')
    for review in positive_reviews:
        normalized_review = normalize_review(review, morph)
        file.write(normalized_review)
        file.write('\n\n\n')
    file.close()


def compute_tf(tokens):
    tokens_dict = collections.Counter(tokens)
    for i in tokens_dict:
        tokens_dict[i] = tokens_dict[i]/len(tokens)
    return tokens_dict


def compute_idf(token, corpus):
    return math.log10(len(corpus) / sum([1.0 for i in corpus if token in i]))


def count_tf_idf(reviews, file_name):
    corpus = []
    for review in reviews:
        tokens = nltk.word_tokenize(review)
        tokens = [token.lower() for token in tokens if (token not in string.punctuation)]
        stop_words = stopwords.words('russian')
        stop_words.extend(['«', '»', '–', '...', '“', '”', '—'])
        tokens = [token for token in tokens if (token not in stop_words)]
        corpus.append(tokens)

    tf_idf_dictionary_for_all_docs = {}

    for review in reviews:
        tokens = nltk.word_tokenize(review)
        tokens = [token.lower() for token in tokens if (token not in string.punctuation)]
        stop_words = stopwords.words('russian')
        stop_words.extend(['«', '»', '–', '...', '“', '”', '—'])
        tokens = [token for token in tokens if (token not in stop_words)]

        tf_idf_dictionary = {}
        computed_tf = compute_tf(tokens)
        for word in computed_tf:

            # cчитаем tf-idf для каждого слова в отзыве
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)

            #  складываем все метрики tf-idf по всем документам
            if word not in tf_idf_dictionary_for_all_docs.keys():
                tf_idf_dictionary_for_all_docs[word] = tf_idf_dictionary[word]
            else:
                tf_idf_dictionary_for_all_docs[word] += tf_idf_dictionary[word]

    sorted_tokens = sorted(tf_idf_dictionary_for_all_docs.items(), key=lambda item: (-item[1], item[0]))
    file = open(file_name, 'w', encoding='utf-8')
    for token, count in sorted_tokens:
        file.write(f'{token}: {count}\n')
    file.close()


morph = pymorphy2.MorphAnalyzer()

# parse_all_reviews_of_one_movie(morph)
# parse_all_negative_rewiews(morph)
# parse_all_positive_rewiews(morph)

# для одного фильма (ММС)
reviews = []
reviews.extend(pandas.read_excel('reviews.xlsx', 0)['Отзыв'])
count_tf_idf(reviews, 'Подсчет метрики для одного фильма.txt')

# для негативных отзывов
negative_reviews = []
negative_reviews.extend(pandas.read_excel('reviews.xlsx', 0)['Отзыв'][0:10:])
negative_reviews.extend(pandas.read_excel('reviews.xlsx', 1)['Отзыв'][0:10:])
negative_reviews.extend(pandas.read_excel('reviews.xlsx', 2)['Отзыв'][0:10:])
count_tf_idf(negative_reviews, 'Подсчет метрики для всех негативных отзывов.txt')

# для позитивных отзывов
positive_reviews = []
positive_reviews.extend(pandas.read_excel('reviews.xlsx', 0)['Отзыв'][20:30:])
positive_reviews.extend(pandas.read_excel('reviews.xlsx', 1)['Отзыв'][20:30:])
positive_reviews.extend(pandas.read_excel('reviews.xlsx', 2)['Отзыв'][20:30:])
count_tf_idf(positive_reviews, 'Подсчет метрики для всех положительных отзывов.txt')
