# -*- coding: utf-8 -*-

import abc
import pickle
import sqlite3
import uuid
import threading
import queue
import pathlib

import pandas as pd
import numpy as np
from scipy import spatial

from unitech import streams

from sklearn import neighbors


class DetectError(Exception):
	pass


class DetectorBase(metaclass=abc.ABCMeta):
    """
    Базовый класс для создания детекторов лиц. Используется для определения общего интерфейса
    для добавления пользователей и получения пользователей из кадров изображений.
    """

    def __init__(self):
        self.current_frame = None
        self.file_mask = ""
        self.video_path = ""
        self.frame_step = 1
        self.db_path = ""
        self.setDebug(False)
        self.useThreading()
        self.useDistanceMetric('cosine', 0.7)

    
    #~ @abc.abstractmethod
    def scanNewUser(self):
        """
        Абстрактный метод для добавления информации о пользователе.
        Должен быть реализован в подклассах.

        :return: Эмбеддинги лица, извлеченные из изображения.
        """

        self.embeddings = []

        # Поток кадров изображения и метаданных.
        for rgb, metadata in self.frame_stream():

            if self.scaningNewUser(rgb):
                break
            
        return self.embeddings


    #~ @abc.abstractmethod
    def recognizeUser(self):
        """
        Абстрактный метод для получения информации о пользователе по изображению.
        Должен быть реализован в подклассах.

        :return: Идентификатор пользователя, если лицо найдено, иначе None.
        """

        # Поток кадров изображения и метаданных.
        for rgb, metadata in self.frame_stream():

            guid = self.recognizingUser(rgb, metadata)
            if guid:
                return guid


    #######################   API Functions   ################################

    # ----------------------------------------------   Start Functions    ----------------------------------------------


    def beginScan(self):
        """
        Функция регистрации пользователя

        Входные параметры:
            отсутствуют
        Выходные параметры:
            string - ID пользователя типа GUID
        Описание:
        ◦ Инициализация видеопотока с камеры с использованием адреса камеры.
        ◦ Захват кадров в реальном времени.
        ◦ Обработка каждого кадра для распознавания лица.
        ◦ Регистрация пользователя в базе данных SQL.
        ◦ Возврат GUID пользователя.
        """
        return self.executeInThread( self.beginScanThread )
          

    def faceRecognitionAll(self):
        """
        Функция распознавания пользователя

        Входные параметры:
            отсутствуют
        Выходные параметры:
            string - ID пользователя типа GUID
        Описание:
        ◦ Инициализация видеопотока с камеры с использованием адреса камеры.
        ◦ Захват кадров в реальном времени.
        ◦ Обработка каждого кадра для распознавания лица.
        ◦ Распознавание лица на основе имеющихся в базе данных значений.
        ◦ Возврат GUID распознанного лица.
        """
        return self.executeInThread( self.faceRecognitionAllThread )


    # ----------------------------------------------   Image Setting Functions   ----------------------------------------------


    def setAdressCamera(self, video_path=0, frame_step=1):
        """
        Функция установки адреса камеры

        Входные параметры:
            string – адрес камеры (индекс устройства или URL).
        Выходные параметры:
            отсутствуют
        Описание:
            установка адреса камеры для захвата
        """
        self.video_path = video_path
        self.frame_step = frame_step
        self.file_mask = ''
        self.frame_streamer = streams.g_from_cv_capture
        if frame_step < 1:
            raise ValueError('Шаг кадров не может быть меньше 1!')

        
    def getVideo(self):
        """
        Функция выдачи видеопотока

        Входные параметры:
            отсутствуют
        Выходные параметры:
            кадр видеопотока
        Описание:
        • Возвращает обработанный кадр видеопотока
        • Должна вызываться в цикле
        """
        return self.current_frame
    
    
    def setCoordinates(self, X, Y, Size):
        """
        Функция установки координат области распознавания

        Входные параметры:
            X, Y -центр области в пикселах
            Size – размер области в пикселах
        Выходные параметры:
            отсутствуют
        Описание:
            Устанавливает новые координаты и размер области для распознавания лица
        """
        pass


    def setAlivePerson(self, seconds):
        """
        Функция установки проверки «живости» лица

        Входные параметры:
            Время в секундах, в течение которых будет анализироваться «живость» лица 
        Выходные параметры:
            Значение типа true/false
        Описание:
            Проверяет, живой человек или это фото
        """
        pass


    def setMethodAlivePerson(self, eye_blink, eye_move, movement, skin):
        """
        Функция установки методов для проверки «живости» лица

        Входные параметры:
            eye_blink – моргание глаз (True/False), 
            eye_move = движение глаз (True/False), 
            movement = движение головы (True/False), 
            skin – текстура кожи (True/False)
        Выходные параметры:
            отсутствуют
        Описание:
            Устанавливает методы определения живости лица. По умолчанию включены все методы
        """
        pass


    def setMethodRecognition(self):
        """
        Функция установки методов детекции и распознавания лица

        Входные параметры:
            string:
                "FACEREC" - FaceRecognition
                "LBPH" – каскады Хаара + LBPHrecognizer(OpenCV)
                "DeepFace" – DeepFace  +VGG-Face
                "DNN" – YuNet (DNN в OpenCV)
        Выходные параметры:
            отсутствуют
        Описание:
            Устанавливает методы детекции и распознавания лица. Методы могут добавляться по мере разработки системы
        """
        pass


    # ----------------------------------------------   Data Base Functions   ----------------------------------------------


    def exportFromData(self, guid):
        """
        Функция экспорта данных

        Входные параметры:
            string - ID пользователя типа GUID
        Выходные параметры:
            Набор векторов
        Описание:
            Экспортирует данные из базы данных sqllite
        """
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            with connection:
                cursor.execute("SELECT embeddings FROM faces WHERE guid=?", (guid,))
                return pickle.loads( cursor.fetchone()[0] )


    def importToData(self, guid, embeddings):
        """
        Функция импорта данных

        Входные параметры:
        • string - ID пользователя типа GUID 
        • Набор векторов
        Выходные параметры:
            отсутствуют
        Описание:
            Занесение данных в базу sqllite
        """
        emb_blob = pickle.dumps(embeddings)
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            with connection:
                cursor.execute("INSERT INTO faces (guid, embeddings) VALUES (?, ?)", (guid, emb_blob))
        self.addEmbeddingsToDataFrame(guid, embeddings)
    
    
    def deleteDataById(self, guid):
        """
        Функция удаления конкретного пользователя

        Входные параметры:
            string - ID пользователя типа GUID
        Выходные параметры:
            отсутствуют
        Описание:
            Удаляет из базы данных SQL пользователей по ID
        """
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            with connection:
                cursor.execute("DELETE FROM faces WHERE guid=?", (guid,))
        self.dataframe = self.dataframe[ self.dataframe.guid != guid ]
        self.refitKNN()


    def deleteAllFromData(self):
        """
        Функция удаления всех данных

        Входные параметры:
            отсутствуют
        Выходные параметры:
            отсутствуют
        Описание:
            Удаление всех пользователей из базы данных SQL
        """
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            with connection:
                cursor.execute("DELETE FROM faces")
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='faces'")  # Сброс автоинкремента
        self.dataframe = pd.DataFrame(columns=['guid', 'embedding'])
        self.knn = None
    
    
    #######################   API Functions End  ################################


    def setDebug(self, enable=True):
        self.debug = enable

    def useThreading(self, enable=True):
        self.use_threading = enable
    
    def setDB(self, db_path):
        self.db_path = pathlib.Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.createDB()
        self.loadDBtoDataFrame()

    def streamFromOneFile(self, video_path):
        self.video_path = video_path
        self.file_mask = ''
        self.frame_streamer = streams.g_from_one_file


    def streamFromDirectory(self, video_path, file_mask='*.*'):
        self.video_path = video_path
        self.file_mask = file_mask
        self.frame_streamer = streams.g_from_directory


    def frame_stream(self):
        if self.file_mask:
            video_stream = self.frame_streamer(self.video_path, self.file_mask)
        else:
            video_stream = self.frame_streamer(self.video_path)
        if self.frame_step != 1:
            video_stream = streams.g_thin_out_frames(video_stream, self.frame_step)
        for rgb, metadata in video_stream:
            self.current_frame = rgb
            yield rgb, metadata


    def createDB(self):
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            with connection:
                cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                            guid TEXT UNIQUE, 
                                            embeddings BLOB)''')
    
    
    def loadDBtoDataFrame(self):
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            with connection:
                cursor.execute("SELECT * FROM faces")
                self.dataframe = pd.DataFrame(columns=['guid', 'embedding'])
                for id, guid, emb_blob in cursor.fetchall():
                    embeddings = pickle.loads( emb_blob )
                    self.addEmbeddingsToDataFrame(guid, embeddings)
    
    
    def addEmbeddingsToDataFrame(self, guid, embeddings):
        if isinstance(embeddings, str):
            return
        new_emb = [ np.array(emb) for emb in embeddings ]
        columns = {'guid': [guid] * len(new_emb), 'embedding': new_emb}
        self.dataframe = pd.concat([ self.dataframe, pd.DataFrame(columns) ], axis=0)
        self.refitKNN()
    
    
    def executeInThread(self, function):
        result_queue = queue.Queue()
        if self.use_threading:
            recognition_thread = threading.Thread(target=function, args=(result_queue, ))
            recognition_thread.start()
            #~ recognition_thread.join()  # Ждем завершения потока
            return result_queue.get()  # Получаем результат из очереди
        else:
            function(result_queue)
            return result_queue.get(False)  # Получаем результат из очереди


    def beginScanThread(self, result_queue):
        embeddings = self.scanNewUser()
        #~ print(embeddings)
        if embeddings:
            guid = str(uuid.uuid4())
            self.importToData(guid, embeddings)
        else:
            guid = ''
            #~ raise DetectError('Пользователь не обнаружен!')
        result_queue.put(guid)

        
    def faceRecognitionAllThread(self, result_queue):
        guid = self.recognizeUser()
        result_queue.put(guid if guid else '')


    def searchGuidInDataframe(self, embedding):
        if self.distance_search_type == 'KNN':
            guid, self.last_distance =  self.searchGuidByKnn(embedding)
        else:
            guid, self.last_distance = self.searchGuidByMin(embedding)      # lambda a,b: np.linalg.norm(a-b)
        if self.last_distance < self.distance_threshold:
            return guid


    def searchGuidByMin(self, embedding ):
        
        distances = []
        for index, row in self.dataframe.iterrows():
            stored_embedding = np.array(row['embedding'])
            distance = spatial.distance.cdist([stored_embedding], [embedding], self.distance_metric)
            distances.append( (row['guid'], distance[0][0]) )

        if not distances:
            return '', 10**10
        return min(distances, key=lambda x: x[1])

    
    def useDistanceMetric(self, distance_metric, distance_threshold, distance_search_type='bruteforce'):
        """
        `distance_metric` can be: 
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, 
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, 
        ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, 
        ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
        """
        self.last_distance = 10**10
        self.distance_metric = distance_metric
        self.distance_threshold = distance_threshold
        self.distance_search_type = distance_search_type
        self.refitKNN()


    def refitKNN(self):
        self.knn = None
        if self.distance_search_type == 'KNN':
            embeddings = list(self.dataframe['embedding'].array)
            # Обучаем модель k-NN на всех эмбеддингах
            if embeddings:
                self.knn = neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto', metric=self.distance_metric).fit( embeddings )

    
    def searchGuidByKnn(self, embedding):
        if not self.knn:
            return '', 10**10
        # Находим ближайшего соседа для нового эмбеддинга
        distances, indices = self.knn.kneighbors( [embedding] )
        # Получаем соответствующий идентификатор
        return self.dataframe.iloc[indices[0][0]]['guid'], distances[0][0]


