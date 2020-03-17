# coding: utf8
import math
import string

import nltk
import pandas
import pymorphy2
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy

from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def normalize_review(review, morph):
    """

    :param review:
    :param morph:
    :return:
    """

    tokens = nltk.word_tokenize(review)
    normalized_tokens = []
    # normalized_review = ''
    for token in tokens:
        token = morph.parse(token)[0].normal_form
        if token not in stop_words and token not in string.punctuation:
            normalized_tokens.append(token.lower())

    return normalized_tokens, " ".join(normalized_tokens)


def get_my_reviews(morph):
    """
    Получаем мои ревью из эксель таблицы.

    :param morph:
    :return tuple: (название фильма, текст ревью, реальный класс ревью)
    """
    pos_reviews = []
    neg_reviews = []
    neutr_reviews = []

    excel_file = pandas.read_excel('Отзывы кино.xlsx', 0)
    for i, row in excel_file.iterrows():
        if (row['title'] == 'Криминальное чтиво' or
                row['title'] == 'Маленькая Мисс Счастье' or
                row['title'] == 'Амели'):

            if row['label'] == 1:
                pos_reviews.append(
                    (row['title'], normalize_review(row['text'], morph),
                     str(row['label'])))
            if row['label'] == -1:
                neg_reviews.append(
                    (row['title'], normalize_review(row['text'], morph),
                     str(row['label'])))
            if row['label'] == 0:
                neutr_reviews.append(
                    (row['title'], normalize_review(row['text'], morph),
                     str(row['label'])))

    return pos_reviews, neg_reviews, neutr_reviews


def get_reviews(morph):
    """
    Получаем ревью одногруппников из эксель таблицы.

    :param morph:
    :return tuple: (название фильма, список токенов, реальный класс ревью)
    """
    pos_reviews = []
    neg_reviews = []
    neutr_reviews = []

    excel_file = pandas.read_excel('Отзывы кино.xlsx', 0)
    for i, row in excel_file.iterrows():
        if (row['title'] != 'Криминальное чтиво' and
                row['title'] != 'Маленькая Мисс Счастье' and
                row['title'] != 'Амели'):
            if row['label'] == 1:
                pos_reviews.append(
                    (row['title'], normalize_review(row['text'], morph),
                     str(row['label'])))
            if row['label'] == -1:
                neg_reviews.append(
                    (row['title'], normalize_review(row['text'], morph),
                     str(row['label'])))
            if row['label'] == 0:
                neutr_reviews.append(
                    (row['title'], normalize_review(row['text'], morph),
                     str(row['label'])))

    return pos_reviews, neg_reviews, neutr_reviews


morph = pymorphy2.MorphAnalyzer()

stop_words = stopwords.words('russian')
stop_words.extend(['«', '»', '–', '...', '“', '”', '—', '!',
                   '@', '№', ':', ',', '.', '?', ':', '(', ')'])
stop_words = set(stop_words)

train_pos_reviews, train_neg_reviews, train_neutr_reviews = get_reviews(morph)
test_pos_reviews, test_neg_reviews, test_neutr_reviews = get_my_reviews(morph)

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer="word")

print('Done with parsing')

train_pos_reviews = [item[1][1] for item in train_pos_reviews]
train_neg_reviews = [item[1][1] for item in train_neg_reviews]
train_neutr_reviews = [item[1][1] for item in train_neutr_reviews]

test_pos_reviews = [item[1][1] for item in test_pos_reviews]
test_neg_reviews = [item[1][1] for item in test_neg_reviews]
test_neutr_reviews = [item[1][1] for item in test_neutr_reviews]
model = LogisticRegression(max_iter=1000)

X = train_pos_reviews + train_neg_reviews + train_neutr_reviews
y = [1] * len(train_pos_reviews) + [-1] * len(train_neg_reviews) + [0] * len(
    train_neutr_reviews)

X_test = test_pos_reviews + test_neg_reviews + test_neutr_reviews
y_test = [1] * len(test_pos_reviews) + [-1] * len(test_neg_reviews) + [
    0] * len(test_neutr_reviews)

X = vectorizer.fit_transform(X).toarray()
X_test = vectorizer.transform(X_test).toarray()

model.fit(X, y)
print(model)
print('\n')
expected = y_test
predicted = model.predict(X_test)

print(metrics.classification_report(expected, predicted))
print('\n')
print(metrics.confusion_matrix(expected, predicted))
weights = model.coef_[1]
vectorizer = CountVectorizer(analyzer="word")
features = vectorizer.fit_transform(train_neutr_reviews).toarray()
vocab = vectorizer.get_feature_names()
mapping = zip(weights, vocab)

mapping = sorted(mapping, key=lambda tup: tup[0])

# первые 10
print(mapping[:10])
print('\n\n')
# последние 10
print(mapping[-10:])
weights = model.coef_[0]
vectorizer = CountVectorizer(analyzer="word")
features = vectorizer.fit_transform(train_neg_reviews).toarray()
vocab = vectorizer.get_feature_names()

mapping = zip(weights, vocab)

mapping = sorted(mapping, key=lambda tup: tup[0])
# первые 10
print(mapping[:10])
print('\n\n')
# последние 10
print(mapping[-10:])
weights = model.coef_[2]
vectorizer = CountVectorizer(analyzer="word")
features = vectorizer.fit_transform(train_neg_reviews).toarray()
vocab = vectorizer.get_feature_names()

mapping = zip(weights, vocab)

mapping = sorted(mapping, key=lambda tup: tup[0])
# первые 10
print(mapping[:10])
print('\n\n')
# последние 10
print(mapping[-10:])
