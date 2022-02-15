import requests #Библиотека для работы с HTTP запросами
import json #Для работы с данными json
import time
import os #Модуль для работы с операционной системой
from tqdm.auto import tqdm #показывает прогресс выполнения цикла
import psycopg2 # Для работы с БД
import pandas as pd #Для работы с табличными данными
from sqlalchemy import engine as sql # Библиотека для работы с СУБД

#Функция для получения списка вакансий
def getPage(page=0):
    params = {
        'page': page,
        'per_page': 100
    } #Определяем параметры страниц
    req = requests.get('https://api.hh.ru/vacancies', params) #Запрос к API
    data = req.content.decode()#Декодируем для корректного отображения Кириллицы
    req.close()
    return data

#Считываем 2000 вакансий
for page in tqdm(range(0, 20)):

    js = json.loads(getPage(page))#Преобразуем текст в стравочник
    file_name = 'C:/Users/HP/Desktop/for_sber_scool/{}.json'.format(
        len(os.listdir('C:/Users/HP/Desktop/for_sber_scool'))) #Создаем файл на компьютере
    f = open(file_name, mode='w', encoding='utf8')#Открываем созданный файл, записывапем в него ответ запроса и закрываем
    f.write(json.dumps(js, ensure_ascii=False))
    f.close()

    if (js['pages'] - page) <= 1:
        break

    time.sleep(0.25) #Задержка между запросами


# Получаем список ранее созданных файлов со списком вакансий и проходимся по нему в цикле
for fl in tqdm(os.listdir('c:/Users/HP/Desktop/for_sber_scool')):
    #Открываем файл, читаем и закрываем
    f = open('c:/Users/HP/Desktop/for_sber_scool/{}'.format(fl), encoding='utf8')
    jsonText = f.read()
    f.close()

    jsonObj = json.loads(jsonText)
    # Проходимся по списку вакансий
    for v in jsonObj['items']:
        #Делаем запрос к API по конкретной вакансии
        req = requests.get(v['url'])
        data = req.content.decode()
        req.close()
        #Создаем отдельный файл с данными по конкретной вакансии
        fileName = 'c:/Users/HP/Desktop/for_sber_scool/vacancies/{}.json'.format(v['id'])
        f = open(fileName, mode='w', encoding='utf8')
        f.write(data)
        f.close()

        time.sleep(0.5)

print('Вакансии собраны')





#Реквизиты к БД
DB_HOST = '52.157.159.24'
DB_USER = 'student7'
DB_USER_PASSWORD = 'student7_password'
DB_NAME = 'sql_ex_for_student7'


import psycopg2
from psycopg2 import Error
# Создаем таблицу в БД
try:
    connection = psycopg2.connect(host=DB_HOST, user=DB_USER, password=DB_USER_PASSWORD, dbname=DB_NAME)
    cursor = connection.cursor()
    create_table_query = '''CREATE TABLE vacan(
                            id   INT              NOT NULL,
                            name VARCHAR (150)     NOT NULL,
                            description TEXT,
                            url VARCHAR (50),
                            PRIMARY KEY (id));; '''
    cursor.execute(create_table_query)
    connection.commit()
    print("Таблица - vacan успешно создана в PostgreSQL")
    cursor = connection.cursor()

except (Exception, Error) as error:
    print("Ошибка при работе с PostgreSQL", error)
finally:
    if connection:
        cursor.close()
        connection.close()
        print("Соединение с PostgreSQL закрыто")



IDs = [] #Список ID вакансий
names = [] #Список наименовании вакансии
descriptions = [] #Описание вакансии
url = [] #Сылка на вакансию

#Проходимся циклом по каждому файлу с данными по вакансии
for fl in tqdm(os.listdir('c:/Users/HP/Desktop/for_sber_scool/vacancies')):

    f = open('c:/Users/HP/Desktop/for_sber_scool/vacancies/{}'.format(fl), encoding='utf8')
    jsonText = f.read()
    f.close()
    #сохраняем в списки данные по вакансии
    jsonObj = json.loads(jsonText)
    if 'id' in jsonObj:
        IDs.append(jsonObj['id'])
        names.append(jsonObj['name'])
        descriptions.append(jsonObj['description'])
        url.append(jsonObj['alternate_url'])

from IPython import display #Модуль для работы с отображением вывода Jupyter
#Подключаемся к БД
eng = sql.create_engine(f'postgresql://{DB_USER}:{DB_USER_PASSWORD}@{DB_HOST}/{DB_NAME}')
conn = eng.connect()
# Создаем фрейм из списков с данными
df = pd.DataFrame({'id': IDs, 'name': names, 'description': descriptions, 'url': url})
# Сохраняем фрейм в таблицу SQL
df.to_sql('vacan', conn, schema='public', if_exists='append', index=False)

conn.close()

display.clear_output(wait=True)
display.display('Вакансии загружены в БД')


