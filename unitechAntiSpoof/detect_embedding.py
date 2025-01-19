# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import sys

import cv2



class HaarCascade:


    def __init__(self):
        for path in sys.path:
            if 'site-packages' in path:
                data_path = Path(path) / 'cv2' / 'data'
                break
        self.detect = cv2.CascadeClassifier( data_path / 'haarcascade_frontalface_default.xml')


    def detector(self, frame):
        faces = self.detect.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        ans = []
        for x, y, w, h in faces:
            bbox = np.array( [x, y, x+w, y+h] )
            ans.append( {'bbox': bbox, 'det_score': None } )
        return ans



class Dlib:


    def __init__(self):
        import dlib
        self.detect = dlib.get_frontal_face_detector()       # OK    14% CPU     300Mb
        self.predictor = None
        #~ self.predictor = dlib.shape_predictor("../files/shape_predictor_68_face_landmarks.dat")


    def detector(self, frame):
        ans = []
        rects = self.detect(frame, 0)
        for rect in rects:
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            bbox = np.array( [x, y, x+w, y+h] )
            ans.append( {'bbox': bbox, 'det_score': None } )
        return ans


    def predict(self, frame, rect):
        shape = self.predictor(frame, dlib.rectangle(*rect))
        return np.array([(point.x, point.y) for point in shape.parts()])

    

# для установки insightface выполните:
# python -m pip install insightface onnxruntime
# и если deepface уже был установлен, то выполните:
# python -m pip uninstall opencv-python-headless
# python -m pip install opencv-contrib-python opencv-python

class Insightface:
    """
    Класс для работы с фреймворком InsightFace
    """

    def __init__(self, name=None, allowed_modules=None, ctx_id=-1, threshold=0.4, det_size=(256,256)):
        """
        `name` - имя пакета моделей
        `allowed_modules` - выбор режима работы пакета моделей
        `threshold` - порог обнаружения лиц
        `det_size` - параметр детектора лиц
        """
        from insightface.app import FaceAnalysis
        # Создание экземпляра FaceAnalysis и подготовка модели
        self.app = FaceAnalysis(name=name, allowed_modules=allowed_modules)
        self.app.prepare(ctx_id=ctx_id, det_thresh=threshold, det_size=det_size) # 'ctx id, <0 means using cpu'

    def detector(self, rgb):
        """
        Ищет лица и выдает список словарей.
        Каждый словарь содержит:
            `bbox` - область координат лица
            `det_score` - метрика качества найденного лица
        Может содержать
            `embedding` - вектор embedding лица (если в `allowed_modules` конструктора есть 'recognition')
        """
        return self.app.get(rgb)

    def embedding(self, face):
        """
        Выдает вектор embedding лица
        """
        faces = self.app.get(face)
        try:
            return faces[0]['embedding']
        except IndexError:
            return []



# для установки deepface выполните:
# python -m pip install deepface tf-keras

class Deep_Face:
    """
    Класс для работы с фреймворком DeepFace
    """

    def __init__(self, detector_backend='mediapipe', model_name="DeepID"):
        """
        `detector_backend` - имя модели детектора лиц
        `model_name` - имя модели embedding
        """

        # Так как DeepFace включает множество других библиотек и не редки конфликты
        from deepface import DeepFace 
        self.DeepFace = DeepFace

        self.detector_backend = detector_backend
        self.model_name = model_name

    def detector(self, rgb):
        """
        Ищет лица и выдает список словарей.
        Каждый словарь содержит:
            `bbox` - область координат лица
            `det_score` - метрика качества найденного лица
        """
        result = []
        try:
            faces = self.DeepFace.extract_faces(rgb, detector_backend=self.detector_backend)
        except ValueError:													# Faces could not be detected
            return result
        for face in faces:
            area = face['facial_area']											# area['left_eye'], area['right_eye']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            bbox = np.array( [x, y, x+w, y+h] )
            result.append( {'bbox': bbox, 'det_score': face['confidence'] } )
        return result

    def embedding(self, face):
        """
        Выдает вектор embedding лица
        """
        if not face.size:
            return []
        try:
            faces = self.DeepFace.represent(face, detector_backend='skip', model_name=self.model_name)
        except ValueError:													# Faces could not be detected
            return []
        assert len(faces) == 1
        return faces[0]['embedding']


