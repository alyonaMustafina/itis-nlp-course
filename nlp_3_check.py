# coding: utf8
import math
import string

import nltk
import pandas
import pymorphy2
from nltk.corpus import stopwords
import numpy as np


def normalize_review(review, morph, stop_word=None):
    """

    :param stop_words:
    :param review:
    :param morph:
    :return:
    """

    # Токены для исключения из корпуса
    # stop_words = stopwords.words('russian')
    stop_words = ['«', '»', '...', '“', '”', '—', '№']
    stop_words.extend(stopwords.words('russian'))
    if stop_word:
        stop_words.extend(stop_word)

    tokens = nltk.word_tokenize(review)
    normalized_tokens = []
    # normalized_review = ''
    for token in tokens:
        token = morph.parse(token)[0].normal_form
        if token not in stop_words and token not in string.punctuation:
            normalized_tokens.append(token.lower())

    return normalized_tokens


def get_trained_model(morph):
    """
    Обучаем модель на всех ревью из таблицы, кроме тех, что надо будет
    классифицировать.

    :param morph:
    :return model: Ключи - классы, значения - списки с нормализованными ревью,
    относящимися к соответствующему классу
    """

    all_words = []
    model = {
        '1': [],
        '-1': [],
        '0': []
    }
    excel_file = pandas.read_excel('Отзывы кино.xlsx', 0)
    for i, row in excel_file.iterrows():
        if (row['title'] == 'Криминальное чтиво' or
                row['title'] == 'Маленькая Мисс Счастье' or
                row['title'] == 'Амели'):
            continue
        tokens = normalize_review(row['text'], morph)
        model[str(row['label'])].append(tokens)
        all_words.extend(tokens)

    # for row in ['предоставляю услуги бухгалтера', 'спешите купить виагру']:
    #     tokens = normalize_review(row, morph)
    #     model['-1'].append(tokens)
    #     all_words.extend(tokens)
    #
    # for row in ['надо купить молоко']:
    #     tokens = normalize_review(row, morph)
    #     model['1'].append(tokens)
    #     all_words.extend(tokens)

    pos_class_words = []
    pos_class_words_cnt = 0
    for item in model['1']:
        pos_class_words_cnt += len(item)
        pos_class_words.extend(item)

    neg_class_words_cnt = 0
    neg_class_words = []
    for item in model['-1']:
        neg_class_words_cnt += len(item)
        neg_class_words.extend(item)

    neut_class_words_cnt = 0
    neut_class_words = []
    for item in model['0']:
        neut_class_words_cnt += len(item)
        neut_class_words.extend(item)

    return (model, all_words, list(set(all_words)),
            pos_class_words_cnt, neg_class_words_cnt, neut_class_words_cnt,
            pos_class_words, neg_class_words, neut_class_words
            )


def get_reviews_to_classify(morph):
    """
    Получаем ревью из эксель таблицы, которые надо классифицировать.

    :param morph:
    :return tuple: (название фильма, текст ревью, реальный класс ревью)
    """
    reviews = []
    excel_file = pandas.read_excel('Отзывы кино.xlsx', 0)
    for i, row in excel_file.iterrows():
        if (row['title'] == 'Криминальное чтиво' or
                row['title'] == 'Маленькая Мисс Счастье' or
                row['title'] == 'Амели'):
            reviews.append(
                (normalize_review(row['text'], morph,
                                  # stopwords.words('russian')
                                  ),
                 str(row['label'])))
    # reviews = []
    # rows = ['надо купить сигареты']
    # for row in rows:
    #     reviews.append((normalize_review(row, morph,
    #                                      # stopwords.words('russian')
    #                                      ), '-1'))

    return reviews


def classify_review(rev,
                    p_pos, p_neg, p_neut,
                    all_words, all_unique_words_cnt,
                    pos_class_words_cnt, neg_class_words_cnt,
                    neut_class_words_cnt,
                    pos_class_words, neg_class_words, neut_class_words
                    ):
    pos_result = p_pos
    neg_result = p_neg
    neut_result = p_neut

    act_class = rev[1]
    rev_words = rev[0]

    # print('pos cl words')
    # print(pos_class_words)
    #
    # print('neg cl words')
    # print(neg_class_words)

    for word in rev_words:
        # print(word)
        # print('cl 1')
        # print(pos_class_words.count(word)+1)
        # print(all_unique_words_cnt)
        # print(pos_class_words_cnt)

        pos_result += math.log((1.0 + pos_class_words.count(word)) / (
                    all_unique_words_cnt + pos_class_words_cnt))
        # print('cl -1')
        # print(neg_class_words.count(word)+1)
        # print(all_unique_words_cnt)
        # print(neg_class_words_cnt)

        neg_result += math.log((1.0 + neg_class_words.count(word)) / (
                    all_unique_words_cnt + neg_class_words_cnt))

        neut_result += math.log((1.0 + neut_class_words.count(word)) / (
                    all_unique_words_cnt + neut_class_words_cnt))

    label = ''

    results = [pos_result, neg_result, neut_result]
    most_probable = max(results)
    if most_probable == results[0]:
        label = '1'
    elif most_probable == results[1]:
        label = '-1'
    else:
        label = '0'

    # print(
    #     f"Оценка для отзыва, что он:\n"
    #     f"- положительный {pos_result},\n"
    #     f"- отрицательный {neg_result},\n"
    #     f"- нейтральный {neut_result}.\n"
    #     f"Реальная оценка - {act_class}\n"
    #     f"Получившаяся оценка - {label}\n"
    # )


    return act_class, label, act_class==label


morph = pymorphy2.MorphAnalyzer()
trained, all_words, all_unique_words, pos_class_words_cnt, neg_class_words_cnt, neut_class_words_cnt, pos_class_words, neg_class_words, neut_class_words = get_trained_model(
    morph)

all_unique_words_cnt = len(all_unique_words)
test = get_reviews_to_classify(morph)

pos_train_rev_cnt = len(trained['1'])
neg_train_rev_cnt = len(trained['-1'])
neut_train_rev_cnt = len(trained['0'])

docs_cnt = pos_train_rev_cnt + neg_train_rev_cnt + neut_train_rev_cnt

# Априорные вероятности P(c)
# print('apr')
# print(pos_train_rev_cnt)
# print(docs_cnt)
#
# print(neg_train_rev_cnt)
# print(docs_cnt)

p_positive = math.log(pos_train_rev_cnt / docs_cnt)
p_negative = math.log(neg_train_rev_cnt / docs_cnt)
p_neutral = math.log(neut_train_rev_cnt / docs_cnt)

accuracy = 0
act = []
pred = []
for review in test:
    result = classify_review(review,
                             p_positive, p_negative, p_neutral,
                             all_words, all_unique_words_cnt,
                             pos_class_words_cnt, neg_class_words_cnt,
                             neut_class_words_cnt,
                             pos_class_words, neg_class_words, neut_class_words
                             )
    act.append(result[0])
    pred.append(result[1])
    if result[2]:
        accuracy += 1
print(act)
print(pred)
print(f'Accuracy = {accuracy / len(test)}')


# вывод программы
#['-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
#['-1', '-1', '1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '1', '1', '-1', '-1', '1', '1', '-1', '0', '-1', '0', '1', '1', '1', '1', '-1', '1', '1', '1', '1', '1', '-1', '1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '1', '1', '1', '1', '1', '1', '1', '0', '-1', '0', '1', '1', '-1', '-1', '1', '1', '1', '1', '1', '1', '0', '-1', '1', '1', '0', '-1', '-1', '-1', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1']
# Accuracy = 0.6