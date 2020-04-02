# coding: utf8
import math
import string

import nltk
import pandas
import pymorphy2
from nltk.corpus import stopwords

from nlp_2 import normalize_review


def get_trained_model(morph):
    """
    Обучаем модель на всех ревью из таблицы, кроме тех, что надо будет
    классифицировать.

    :param morph:
    :return model: Ключи - классы, значения - списки с нормализованными ревью,
    относящимися к соответствующему классу
    """
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

        model[str(row['label'])].append(normalize_review(row['text'], morph))

    return model


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
            reviews.append((row['title'], normalize_review(row['text'], morph),
                            str(row['label'])))

    return reviews


def get_corpus_and_tokens(texts, stop_words):
    """
    Формируем корпус из уникальных слов по всем ревью.

    :param texts: Список документов (текстов ревью)
    :param stop_words: Список символов, которые не надо
    учитывать при классификации
    :return (corpus, tokens): (список уникальных токенов, список всех токенов)
    """
    corpus = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        tokens = [
            token for token in tokens if (
                    token not in string.punctuation and token not in stop_words)
        ]
        corpus.extend([token for token in tokens])

    return list(set(corpus)), corpus


def classify_review(review, model, p_positive, p_negative, p_neutral):
    # токены ревью, для которого считаем вероятности
    tokens = nltk.word_tokenize(review[1])
    tokens = [token for token in tokens if (token not in string.punctuation)]
    tokens = [token for token in tokens if (token not in stop_tokens)]

    # правдоподобия
    pos = 0
    neg = 0
    netr = 0
    unique_tokens = list(set(tokens))
    for un_token in tokens:
        pos += math.log10(
            (1.0 + sum(
                [1.0 for token in positive_tokens if (token == un_token)])) /
            (len(positive_tokens) + len(corpus))
        )
        neg += math.log10(
            (1.0 + sum(
                [1.0 for token in negative_tokens if (token == un_token)])) /
            (len(negative_tokens) + len(corpus))
        )
        netr += math.log10(
            (1.0 + sum(
                [1.0 for token in neutral_corpus if (token == un_token)])) /
            (len(neutral_tokens) + len(corpus))
        )

    p_positive += pos
    p_negative += neg
    p_neutral += netr

    label = ''
    most_probable = max(p_neutral, p_negative, p_positive)
    if most_probable == p_positive:
        label = '1'
    elif most_probable == p_negative:
        label = '-1'
    else:
        label = '0'

    print(
        f"Оценка для отзыва, что он:\n"
        f"- положительный {p_positive},\n"
        f"- отрицательный {p_negative},\n"
        f"- нейтральный {p_neutral}.\n"
        f"Реальная оценка - {review[2]}\n"
        f"Получившаяся оценка - {label}\n"
    )

    return label == review[2]


morph = pymorphy2.MorphAnalyzer()

# Обучаем модель
trained_model = get_trained_model(morph)

# Делаем выборку из ревью для классификаций
reviews_to_classify = get_reviews_to_classify(morph)

# Токены для исключения из корпуса
stop_tokens = stopwords.words('russian')
stop_tokens.extend(['«', '»', '–', '...', '“', '”', '—', '!',
                    '@', '№', ':', ',', '.', '?', ':', '(', ')'])

# Количество документов в обученной модели
docs_cnt = len(trained_model['-1']) + len(trained_model['1']) + len(
    trained_model['0'])

# Создаем корпуса и отдельно для положительных, отрицательных, нейтральных
# отзывах
positive_corpus, positive_tokens = get_corpus_and_tokens(
    trained_model['1'], stop_tokens)
negative_corpus, negative_tokens = get_corpus_and_tokens(
    trained_model['-1'], stop_tokens)
neutral_corpus, neutral_tokens = get_corpus_and_tokens(
    trained_model['0'], stop_tokens)

# Создаем корпус уникальных слов по всем отзывам
corpus = list(set(positive_corpus + negative_corpus + neutral_corpus))

# Оценка качества классификации
accuracy = 0

for review in reviews_to_classify:

    # Априорные вероятности P(c)
    p_positive = math.log10(len(trained_model['1']) / docs_cnt)
    p_negative = math.log10(len(trained_model['-1']) / docs_cnt)
    p_neutral = math.log10(len(trained_model['0']) / docs_cnt)

    # Классифицируем ревью
    result = classify_review(
        review, trained_model, p_positive, p_negative, p_neutral)
    if result:
        # Увеличиваем счётчик верно классифицированных отзывов
        accuracy += 1

print(f'Accuracy = {accuracy / len(reviews_to_classify)}')
