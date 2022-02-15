import psycopg2 # Для работы с БД
from psycopg2 import Error
import pandas as pd #Для работы с табличными данными
from sqlalchemy import engine as sql # Библиотека для работы с СУБД
import pymorphy2 # Для обработки русских слов
from nltk.corpus import stopwords # Стопслова
from nltk import word_tokenize # Токенизация текста
import re # Для исключения из текста ненужных знаков и символов
import numpy as np


DB_HOST = '52.157.159.24'
DB_USER = 'student7'
DB_USER_PASSWORD = 'student7_password'
DB_NAME = 'sql_ex_for_student7'
# Подключаемся к БД
conn = sqlalchemy.create_engine(f'postgresql://{DB_USER}:{DB_USER_PASSWORD}@{DB_HOST}/{DB_NAME}').connect()
# Выгружаем необходимые данные с помощью SQL запроса
sql = 'select name, description, url from public.vacan'
vacancies_descrip = pd.read_sql(sql, conn)# Сохраняем данные в DataFrame

conn.close() # Закрываем соединение с БД


morph = pymorphy2.MorphAnalyzer()
ru_stopwords = stopwords.words('russian')
# Функция для обработки текста.
def preprocess(tokens):
    return [morph.normal_forms(word)[0]
            for word in tokens
                if word not in ru_stopwords]

# В DF добовляем отделеный столбец с минимально обработанным текстом. его будем отображать в резульатах
vacancies_descrip['des'] = vacancies_descrip['description'].apply(lambda x: (re.sub('[^А-Яа-я0-9,. ]+', '', x)))
# Создаем новый столбец с где убираем лишние символы. Удаляем строки с пустым описанием вакансии
vacancies_descrip['descrip'] = vacancies_descrip['description'].apply(lambda x: (re.sub('[^А-Яа-я ]+', '', x)))
vacancies_descrip['descrip'] = vacancies_descrip['descrip'].apply(lambda x: str(x).strip())
vacancies_descrip['descrip'].replace('', np.nan, inplace=True)
vacancies_descrip.dropna(subset=['descrip'], inplace=True)
# Создаем новый столбец с токенизированным инормированным текстом описания
vacancies_descrip['descrip'] = vacancies_descrip['descrip'].apply(lambda x: word_tokenize(str(x).lower()))
vacancies_descrip['token'] = vacancies_descrip['descrip'].apply(lambda x: preprocess(x))
vacancies_descrip['token'] = vacancies_descrip['token'].apply(lambda x: ' '.join(x))
vacancies_descrip['token'].replace('', np.nan, inplace=True)
vacancies_descrip.dropna(subset=['token'], inplace=True)


from fastapi import FastAPI # Для создания API
from typing import List
from pydantic import BaseModel
import spacy # Обработка тескта
from scipy import spatial # Для вычислений
import uvicorn


app = FastAPI()

nlp = spacy.load("ru_core_news_lg")
# Функция для определения косинусною меры сходства двух векторов
def cos_sim(vec1, vec2):
    cosine_similarity = 1 - spatial.distance.cosine(vec1, vec2)
    return cosine_similarity


class Article(BaseModel):
    content: str
    comments: List[str] = []


@app.post("/text_vec/")
#Создаем функцию которая будет ввыводить 10 релевантных вакансий вводимому в нее тексту основываять на Word2vec
async def text_vec(text: str): # Принимаем текст
    my_text = nlp(text) # Обрабатываем текст word2vec
    vec = my_text.vector # Определяем вектор по введеному текту
    df = vacancies_descrip.copy() # Копируем наш DataFrame с вакансиями
    # Сравниваем вектора введеного текста с вектором описания вакансий
    # Создаем новый столбец с мерой сходства двух векторов
    df['sim'] = df.token.apply(lambda x: cos_sim((nlp(x)).vector, vec))
    df = df.sort_values(by='sim', ascending=False) #Сортируем DF по столбцу схожести векторов
    vacan_list = []
    for r in df.head(10).iterrows(): # Обрабатываем 10 наиболее релевантных вакансий
        # создаем словарь в котором укажем имя, описание, ссылку, и величину схожести отобрынных вакансий
        vac = {
            "title": r[1].get('name'),
            "description": r[1].get('des'),
            "URL": r[1].get('url'),
            "score": r[1].get('sim'),
        }

        vacan_list.append(vac) # Словарь сохраняем в список
    del df
    return {"data": vacan_list} # на выходе отображаются данные вакансий схожие по вектору введеному тексту

from sklearn.feature_extraction.text import TfidfVectorizer # Обработка текста эмбедингом TF-IDF
# Параметры обработки
text_transformer = TfidfVectorizer(stop_words=ru_stopwords, ngram_range=(1,1), lowercase=True, max_features=15000 )

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity # для определения косинусной меры схожести векторов
# Функция выдающая косинусную меру схожести 2 весторов
def sim_tfidf(train_set, test_set):
    stopWords = stopwords.words('russian')
    vectorizer = CountVectorizer(stop_words = stopWords)
    transformer = TfidfTransformer()
    trainVectorizerArray = vectorizer.fit_transform([train_set]).toarray()
    testVectorizerArray = vectorizer.transform([test_set]).toarray()
    return cosine_similarity(trainVectorizerArray, testVectorizerArray)

def arrau(a):
    for m in a:
        for v in m:
            return v

#Создаем функцию которая будет ввыводить 10 релевантных вакансий вводимому в нее тексту основываять на TF-IDF
@app.post("/text_tfidf/")
async def text_tfidf(text: str):
    df = vacancies_descrip.copy() # Копируем DF
    # Сравниваем вектора введеного текста с вектором описания вакансий
    # Создаем новый столбец с мерой сходства двух векторов
    df['sim'] = df.token.apply(lambda x: arrau(sim_tfidf(x, text)))
    df = df.sort_values(by='sim', ascending=False) # сортируем DF по величине схожести векторов
    vacan_list = []
    # создаем словарь в котором укажем имя, описание, ссылку, и величину схожести отобрынных вакансий
    for r in df.head(10).iterrows():
        vac = {
            "title": r[1].get('name'),
            "description": r[1].get('des'),
            "URL": r[1].get('url'),
            "score": r[1].get('sim'),
        }

        vacan_list.append(vac)
    del df
    return {"data": vacan_list}


uvicorn.run(app, port=9000) # Параметры API

